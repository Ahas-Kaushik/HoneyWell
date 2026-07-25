"""
baseline.py -- Unsupervised baseline anomaly detector + honest metrics. PHASE 2.
================================================================================

MODEL
-----
An **Isolation Forest** over the 9 incremental features from `features.py`.

Why Isolation Forest is the right *baseline*:
  * Unsupervised -- it is trained WITHOUT labels (it just learns the shape of the
    bulk of the data and isolates outliers). That keeps the whole pipeline
    label-blind: labels are used ONLY to evaluate, never to fit. This is the
    "label hidden at inference" rule from the brief, honoured end to end.
  * Fast, robust, and -- crucially -- our features are already ENTITY-RELATIVE
    (z-scores, novelty, mismatch are all "deviation from THIS entity's normal").
    So a single global Isolation Forest over these relative features behaves like
    a per-entity anomaly detector without training 100 separate models. That is
    the "baseline profiling model" deliverable, done efficiently.

TEMPORAL (STREAMING-HONEST) EVALUATION
--------------------------------------
We split by TIME, not randomly: fit on the first `TRAIN_FRAC` of the timeline,
evaluate on the rest. This mimics reality -- you train on the past and score the
future -- and it is only sound because the feature engine is causal (each row's
features use only earlier events). A random split would leak future information
through the shared per-entity state and flatter the metrics.

METRICS -- THE ANTI-ACCURACY SUITE (brief section 4.2 / Appendix A)
-------------------------------------------------------------------
On 98.6%-normal data, "accuracy" is a trap: predict "all normal" and score ~98.6%
while catching zero attacks. So we DO NOT headline accuracy. We report:
  * PR-AUC (average precision) -- the primary imbalance-robust score.
  * Precision@Top-k% and Recall@Top-k% -- the "analyst alert budget". If an analyst
    can only review the top 1% of events, how many real attacks are in there
    (precision) and what share of all attacks did we surface (recall)?
  * The alert-budget curve -- recall as a function of % of events reviewed.
  * A per-feature PR-AUC diagnostic -- which single signals carry the load (feeds
    Phase 4 explainability).
The base-rate accuracy is printed ONCE, explicitly labelled as the trap, to show
we understand why it's meaningless -- never as a headline number.

OUTPUTS
-------
  models/baseline_if.joblib   the fitted model + transform metadata (for scoring)
  data/scores.parquet         event_id, anomaly_score, risk_score(0-1), split
  data/phase2_metrics.json    all metrics above, machine-readable for the report
  figures/pr_curve.png        precision-recall curve (test split)
  figures/alert_budget.png    recall vs alert budget (test split)
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

try:
    from models.features import FEATURE_COLUMNS, build_features
except ImportError:  # pragma: no cover
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.features import FEATURE_COLUMNS, build_features

SEED = 42
TRAIN_FRAC = 0.70                 # first 70% of the timeline is the fit window
ALERT_BUDGETS = [0.005, 0.01, 0.02, 0.05]   # top-k% an analyst might review
# heavy-tailed features get a log1p so a few enormous values don't dominate the
# forest's split ranges. (Trees are scale-robust but not tail-robust.)
_LOG_FEATURES = ("geo_velocity_kmh", "failed_auth_in_window", "ip_fanin")


def _model_matrix(feats: pd.DataFrame) -> np.ndarray:
    """Build the numeric matrix the forest sees (with log1p on heavy-tailed cols)."""
    X = feats[FEATURE_COLUMNS].copy()
    for c in _LOG_FEATURES:
        X[c] = np.log1p(X[c].clip(lower=0))
    return X.to_numpy(dtype=np.float64)


def _precision_recall_at_k(scores: np.ndarray, y: np.ndarray, k_frac: float) -> Dict[str, float]:
    """Precision/Recall if the analyst reviews the top k_frac of events by score."""
    n = len(scores)
    k = max(1, int(round(n * k_frac)))
    order = np.argsort(-scores)              # most anomalous first
    topk = order[:k]
    tp = int(y[topk].sum())
    total_pos = int(y.sum())
    precision = tp / k
    recall = tp / total_pos if total_pos else 0.0
    return {"budget": k_frac, "k_events": k, "precision": precision,
            "recall": recall, "true_positives": tp}


def _alert_budget_curve(scores: np.ndarray, y: np.ndarray, points: int = 100):
    """Recall as a function of fraction-of-events-reviewed (for the curve plot)."""
    order = np.argsort(-scores)
    y_sorted = y[order]
    cum_tp = np.cumsum(y_sorted)
    total_pos = max(1, int(y.sum()))
    fracs = np.linspace(1.0 / len(scores), 1.0, points)
    recalls = [cum_tp[max(0, int(round(f * len(scores))) - 1)] / total_pos for f in fracs]
    return fracs, np.array(recalls)


def _per_feature_prauc(feats: pd.DataFrame, y: np.ndarray) -> Dict[str, float]:
    """Univariate PR-AUC per feature: which raw signals carry the load."""
    out = {}
    for c in FEATURE_COLUMNS:
        s = feats[c].to_numpy(dtype=np.float64)
        # z-score is signed; deviation in either direction matters -> use |z|
        if c == "duration_zscore":
            s = np.abs(s)
        try:
            out[c] = float(average_precision_score(y, s))
        except Exception:
            out[c] = float("nan")
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _plots(scores_te, y_te, out_dir="figures"):
    """Save PR curve + alert-budget curve. Matplotlib only (no seaborn)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)

    prec, rec, _ = precision_recall_curve(y_te, scores_te)
    ap = average_precision_score(y_te, scores_te)
    base = y_te.mean()
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec, lw=2, label=f"Isolation Forest (PR-AUC={ap:.3f})")
    plt.axhline(base, ls="--", c="gray", lw=1, label=f"random baseline ({base:.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall (test split)"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{out_dir}/pr_curve.png", dpi=130); plt.close()

    fracs, recalls = _alert_budget_curve(scores_te, y_te)
    plt.figure(figsize=(5, 4))
    plt.plot(fracs * 100, recalls * 100, lw=2)
    for b in ALERT_BUDGETS:
        r = _precision_recall_at_k(scores_te, y_te, b)["recall"]
        plt.scatter([b * 100], [r * 100], zorder=5)
        plt.annotate(f"{b*100:.1f}%->{r*100:.0f}%", (b * 100, r * 100),
                     textcoords="offset points", xytext=(6, -2), fontsize=8)
    plt.xlabel("Alert budget: % of events reviewed")
    plt.ylabel("Recall: % of attacks caught")
    plt.title("Alert-budget curve (test split)"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{out_dir}/alert_budget.png", dpi=130); plt.close()


def run(features_path="data/features.parquet", labels_path="data/labels.parquet",
        rebuild_features=False) -> dict:
    # ---- features (label-blind) ----
    if rebuild_features or not os.path.exists(features_path):
        feats = build_features(out_path=features_path)
    else:
        feats = pd.read_parquet(features_path)
    feats = feats.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    # ---- labels: used ONLY from here down, ONLY for evaluation ----
    labels = pd.read_parquet(labels_path)
    y_all = feats.merge(labels, on="event_id")["label"].to_numpy()

    # ---- temporal split ----
    cut = feats["timestamp"].quantile(TRAIN_FRAC)
    is_train = (feats["timestamp"] <= cut).to_numpy()
    is_test = ~is_train
    X = _model_matrix(feats)

    # ---- fit UNSUPERVISED on the train window (no labels) ----
    clf = IsolationForest(n_estimators=300, max_samples="auto",
                          contamination="auto", random_state=SEED, n_jobs=-1)
    clf.fit(X[is_train])

    # higher = more anomalous
    anomaly_score = -clf.score_samples(X)
    # 0-1 risk via rank percentile (calibrated-looking, monotonic with score)
    risk = pd.Series(anomaly_score).rank(pct=True).to_numpy()

    # ---- metrics on the TEST split (primary) ----
    s_te, y_te = anomaly_score[is_test], y_all[is_test]
    prauc = float(average_precision_score(y_te, s_te))
    rocauc = float(roc_auc_score(y_te, s_te))
    budgets = [_precision_recall_at_k(s_te, y_te, b) for b in ALERT_BUDGETS]
    per_feat = _per_feature_prauc(feats[is_test].reset_index(drop=True), y_te)
    base_rate = float(y_te.mean())
    accuracy_trap = 1.0 - base_rate   # "predict all normal" accuracy -- the trap

    # ---- persist artifacts ----
    os.makedirs("models", exist_ok=True)
    joblib.dump({"model": clf, "feature_columns": FEATURE_COLUMNS,
                 "log_features": _LOG_FEATURES, "train_frac": TRAIN_FRAC},
                "models/baseline_if.joblib", compress=3)
    scores_df = feats[["event_id", "entity_id", "timestamp"]].copy()
    scores_df["anomaly_score"] = anomaly_score
    scores_df["risk_score"] = risk
    scores_df["split"] = np.where(is_train, "train", "test")
    scores_df.to_parquet("data/scores.parquet", index=False)

    metrics = {
        "model": "IsolationForest(n=300)",
        "n_events_total": int(len(feats)),
        "n_test": int(is_test.sum()),
        "test_anomaly_rate": base_rate,
        "PR_AUC_test": prauc,
        "ROC_AUC_test": rocauc,
        "alert_budget_table": budgets,
        "per_feature_PR_AUC": per_feat,
        "accuracy_trap_note": {
            "predict_all_normal_accuracy": accuracy_trap,
            "explanation": "This is why accuracy is banned: guessing 'normal' for "
                           "every event scores this high while catching zero attacks."},
    }
    with open("data/phase2_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    _plots(s_te, y_te)

    _print_report(metrics)
    return metrics


def _print_report(m: dict) -> None:
    print("=" * 70)
    print("PHASE 2 -- BASELINE DETECTOR (Isolation Forest)  |  TEST-SPLIT METRICS")
    print("=" * 70)
    print(f"  test events        : {m['n_test']:,}   "
          f"(anomaly rate {m['test_anomaly_rate']*100:.2f}%)")
    print(f"  PR-AUC  (primary)  : {m['PR_AUC_test']:.3f}   "
          f"[random baseline = {m['test_anomaly_rate']:.3f}]")
    print(f"  ROC-AUC (secondary): {m['ROC_AUC_test']:.3f}")
    print("-" * 70)
    print("  ALERT-BUDGET TABLE  (if the analyst reviews the top-k% by risk):")
    print(f"      {'budget':>7} {'#alerts':>8} {'precision':>10} {'recall':>8} {'TPs':>5}")
    for b in m["alert_budget_table"]:
        print(f"      {b['budget']*100:6.1f}% {b['k_events']:8d} "
              f"{b['precision']:10.3f} {b['recall']:8.3f} {b['true_positives']:5d}")
    print("-" * 70)
    print("  PER-FEATURE PR-AUC  (which single signals carry the load):")
    for k, v in m["per_feature_PR_AUC"].items():
        print(f"      {k:<22} {v:.3f}")
    print("-" * 70)
    acc = m["accuracy_trap_note"]["predict_all_normal_accuracy"]
    print(f"  ACCURACY TRAP: predicting 'all normal' scores {acc*100:.2f}% accuracy")
    print("                 while catching ZERO attacks -> that's why we never")
    print("                 headline accuracy; PR-AUC / recall@budget are the truth.")
    print("=" * 70)
    print("  wrote: models/baseline_if.joblib, data/scores.parquet,")
    print("         data/phase2_metrics.json, figures/pr_curve.png, figures/alert_budget.png")


if __name__ == "__main__":
    run()
