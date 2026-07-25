"""
metrics.py -- The full evaluation suite + figures. PHASE 4.
================================================================================

Produces every number and plot the report headlines with -- all the imbalance-
robust metrics the brief's Appendix A demands, plus two operational-maturity
bonuses (calibration, de-duplication) from section 5.

WHAT AND WHY
------------
  * PR-AUC (primary) + ROC-AUC ..... imbalance-robust detection quality.
  * Alert-budget curve ............. recall vs % of events reviewed -- THE SOC
                                     metric: "if I can only look at the top 1-2%,
                                     how many attacks do I catch?"
  * Per-class confusion matrix ..... attack-type attribution quality (criterion #2).
  * FP-rate at budget .............. how much of the analyst's queue is noise.
  * Calibration (bonus, sec 5) ..... make the 0-1 risk a MEANINGFUL probability
                                     (isotonic), with a reliability curve + Brier.
  * Alert de-duplication (bonus) ... collapse a burst (e.g. a 40-event brute force)
                                     into ONE grouped alert -- analysts hate storms.

We never headline accuracy. Figures land in ./figures for the report/slides.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             precision_recall_curve, confusion_matrix,
                             classification_report, brier_score_loss)
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve

try:
    from models.baseline import ALERT_BUDGETS, _precision_recall_at_k, _alert_budget_curve
except ImportError:  # pragma: no cover
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.baseline import ALERT_BUDGETS, _precision_recall_at_k, _alert_budget_curve

FIGDIR = "figures"
DEDUP_GAP_MIN = 30            # alerts of same entity+type within 30 min = one incident
BENIGN_TYPES = {"normal", "insider_drift"}
# fixed class order for a readable confusion matrix
CLASS_ORDER = ["normal", "insider_drift", "brute_force", "credential_stuffing",
               "impossible_travel", "device_spoofing", "lateral_movement",
               "low_and_slow", "novel/unknown"]


def _load():
    scores = pd.read_parquet("data/scores_v2.parquet")   # unified_score, split
    labels = pd.read_parquet("data/labels.parquet")
    alerts = pd.read_parquet("data/alerts.parquet")       # test events + predicted_type
    df = scores.merge(labels, on="event_id")
    return df, alerts


def _detection_metrics(df: pd.DataFrame) -> dict:
    te = df[df["split"] == "test"]
    s = te["unified_score"].to_numpy(); y = te["label"].to_numpy()
    budgets = [_precision_recall_at_k(s, y, b) for b in ALERT_BUDGETS]
    for b in budgets:                                     # add FP rate at each budget
        b["false_positive_rate"] = 1.0 - b["precision"]
    return {"PR_AUC": float(average_precision_score(y, s)),
            "ROC_AUC": float(roc_auc_score(y, s)),
            "test_anomaly_rate": float(y.mean()),
            "alert_budget_table": budgets}, s, y


def _plot_pr_and_budget(s, y):
    os.makedirs(FIGDIR, exist_ok=True)
    prec, rec, _ = precision_recall_curve(y, s)
    ap = average_precision_score(y, s)
    plt.figure(figsize=(5, 4))
    plt.plot(rec, prec, lw=2, color="#c0392b", label=f"Unified detector (PR-AUC={ap:.3f})")
    plt.axhline(y.mean(), ls="--", c="gray", lw=1, label=f"random ({y.mean():.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall (test)")
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(f"{FIGDIR}/pr_curve_final.png", dpi=140); plt.close()

    fracs, recalls = _alert_budget_curve(s, y)
    plt.figure(figsize=(5, 4))
    plt.plot(fracs * 100, recalls * 100, lw=2, color="#2c3e50")
    for b in ALERT_BUDGETS:
        r = _precision_recall_at_k(s, y, b)["recall"]
        plt.scatter([b * 100], [r * 100], zorder=5, color="#c0392b")
        plt.annotate(f"{b*100:.1f}%->{r*100:.0f}%", (b * 100, r * 100),
                     textcoords="offset points", xytext=(6, -3), fontsize=8)
    plt.xlabel("Alert budget: % of events reviewed")
    plt.ylabel("Recall: % of attacks caught")
    plt.title("Alert-budget curve (test)"); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(f"{FIGDIR}/alert_budget_final.png", dpi=140); plt.close()


def _per_type_recall(df: pd.DataFrame, budgets=(0.01, 0.02)) -> dict:
    """Detection recall for EACH attack type at each alert budget.

    The aggregate PR-AUC is dominated by the most frequent attack (brute force),
    so it hides whether whole CATEGORIES are missed. This table is the honest
    view: does the analyst's top-k queue contain at least most of every kind of
    attack? It is where the sequence-model fusion visibly earns its keep.
    """
    te = df[df["split"] == "test"].copy()
    out = {}
    for b in budgets:
        k = max(1, int(round(len(te) * b)))
        top = set(te["unified_score"].nlargest(k).index)
        te["_in"] = te.index.isin(top)
        rec = {}
        for t, g in te[te["label"] == 1].groupby("attack_type"):
            rec[t] = {"recall": round(float(g["_in"].mean()), 3), "n": int(len(g))}
        out[f"top_{b*100:.0f}pct"] = rec
    return out


def _confusion(alerts: pd.DataFrame) -> dict:
    """Attack-type attribution over ALERTS (what the analyst is actually shown)."""
    a = alerts[alerts["is_alert"]].copy()
    y_true = a["attack_type"].astype(str)
    y_pred = a["predicted_type"].astype(str)
    present = [c for c in CLASS_ORDER if c in set(y_true) | set(y_pred)]
    cm = confusion_matrix(y_true, y_pred, labels=present)

    plt.figure(figsize=(7.5, 6.2))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.xticks(range(len(present)), present, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(len(present)), present, fontsize=8)
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(len(present)):
        for j in range(len(present)):
            if cm[i, j]:
                plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                         color="white" if cm[i, j] > thresh else "black")
    plt.ylabel("True type"); plt.xlabel("Predicted type")
    plt.title("Attack-type confusion (alerts shown to analyst)")
    plt.tight_layout(); plt.savefig(f"{FIGDIR}/confusion_matrix.png", dpi=140); plt.close()

    rep = classification_report(y_true, y_pred, labels=present,
                                output_dict=True, zero_division=0)
    return {"labels": present, "matrix": cm.tolist(), "classification_report": rep}


def _calibration(df: pd.DataFrame) -> dict:
    """Isotonic-calibrate the anomaly score into a probability; reliability + Brier."""
    tr = df[df["split"] == "train"]; te = df[df["split"] == "test"]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(tr["unified_score"].to_numpy(), tr["label"].to_numpy())
    p_te = iso.predict(te["unified_score"].to_numpy())
    y_te = te["label"].to_numpy()
    brier_raw = brier_score_loss(y_te, te["risk_score"].to_numpy())
    brier_cal = brier_score_loss(y_te, p_te)

    frac_pos, mean_pred = calibration_curve(y_te, p_te, n_bins=8, strategy="quantile")
    plt.figure(figsize=(5, 4))
    plt.plot([0, 1], [0, 1], ls="--", c="gray", label="perfectly calibrated")
    plt.plot(mean_pred, frac_pos, "o-", color="#16a085", label=f"calibrated (Brier={brier_cal:.4f})")
    plt.xlabel("Predicted P(attack)"); plt.ylabel("Observed attack fraction")
    plt.title("Reliability curve (test)"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f"{FIGDIR}/calibration.png", dpi=140); plt.close()

    # persist calibrated probabilities for the dashboard
    out = te[["event_id"]].copy(); out["calibrated_prob"] = p_te
    out.to_parquet("data/calibrated_scores.parquet", index=False)
    return {"brier_raw_rankscore": float(brier_raw), "brier_calibrated": float(brier_cal)}


def _dedup(alerts: pd.DataFrame) -> dict:
    """Group alerts of the same entity+type within DEDUP_GAP_MIN into one incident."""
    a = alerts[alerts["is_alert"]].sort_values(["entity_id", "predicted_type", "timestamp"]).copy()
    group_ids, gid = [], -1
    prev = None
    for _, r in a.iterrows():
        key = (r["entity_id"], r["predicted_type"])
        if prev is None or key != prev[0] or (r["timestamp"] - prev[1]) > pd.Timedelta(minutes=DEDUP_GAP_MIN):
            gid += 1
        group_ids.append(gid)
        prev = (key, r["timestamp"])
    a["incident_id"] = group_ids
    n_raw, n_grouped = len(a), a["incident_id"].nunique()
    sizes = a.groupby("incident_id").size()
    a.to_parquet("data/alert_groups.parquet", index=False)
    return {"raw_alerts": int(n_raw), "grouped_incidents": int(n_grouped),
            "reduction_factor": round(n_raw / max(n_grouped, 1), 2),
            "largest_incident_events": int(sizes.max())}


def run() -> dict:
    df, alerts = _load()
    det, s, y = _detection_metrics(df)
    _plot_pr_and_budget(s, y)
    per_type = _per_type_recall(df)
    conf = _confusion(alerts)
    cal = _calibration(df)
    dedup = _dedup(alerts)

    metrics = {"detection": det, "per_type_recall": per_type, "confusion": conf,
               "calibration": cal, "deduplication": dedup}
    with open("data/phase4_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    _print(metrics)
    return metrics


def _print(m: dict):
    d = m["detection"]
    print("=" * 72)
    print("PHASE 4B -- FULL METRICS SUITE  (test split, ~1.3% anomalies)")
    print("=" * 72)
    print(f"  PR-AUC (primary) : {d['PR_AUC']:.3f}      ROC-AUC : {d['ROC_AUC']:.3f}")
    print("-" * 72)
    print("  ALERT-BUDGET / FALSE-POSITIVE TABLE:")
    print(f"      {'budget':>7} {'#alerts':>8} {'precision':>10} {'recall':>8} {'FP-rate':>8}")
    for b in d["alert_budget_table"]:
        print(f"      {b['budget']*100:6.1f}% {b['k_events']:8d} {b['precision']:10.3f} "
              f"{b['recall']:8.3f} {b['false_positive_rate']:8.3f}")
    print("-" * 72)
    pt = m["per_type_recall"]["top_2pct"]
    print("  PER-ATTACK-TYPE DETECTION RECALL @ top-2% budget (coverage by category):")
    for t, r in sorted(pt.items(), key=lambda kv: -kv[1]["recall"]):
        print(f"      {t:<20} recall={r['recall']:.2f}  (n={r['n']})")
    print("-" * 72)
    rep = m["confusion"]["classification_report"]
    print("  ATTACK-TYPE ATTRIBUTION (per-class naming, over alerts):")
    for c in m["confusion"]["labels"]:
        if c in rep:
            r = rep[c]
            print(f"      {c:<20} P={r['precision']:.2f} R={r['recall']:.2f} "
                  f"F1={r['f1-score']:.2f}  (n={int(r['support'])})")
    print("-" * 72)
    cal = m["calibration"]
    print(f"  CALIBRATION (bonus): Brier {cal['brier_raw_rankscore']:.4f} (rank) "
          f"-> {cal['brier_calibrated']:.4f} (isotonic)  lower=better")
    dd = m["deduplication"]
    print(f"  DE-DUPLICATION (bonus): {dd['raw_alerts']} raw alerts -> "
          f"{dd['grouped_incidents']} incidents ({dd['reduction_factor']}x fewer; "
          f"largest burst={dd['largest_incident_events']} events -> 1)")
    print("=" * 72)
    print("  figures: pr_curve_final, alert_budget_final, confusion_matrix, calibration (.png)")
    print("  wrote: data/phase4_metrics.json, data/calibrated_scores.parquet, data/alert_groups.parquet")


if __name__ == "__main__":
    run()
