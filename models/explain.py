"""
explain.py -- Per-alert explanations: SHAP + recon error -> plain English. PHASE 4.
================================================================================

WHY THIS IS THE MOST IMPORTANT FILE (brief section 4.1)
-------------------------------------------------------
Explainability is the single most heavily-weighted, most-winnable criterion. A
score alone ("risk 0.94") is useless to a tired analyst at 2am. They need: WHY,
in one sentence, with the numbers -- and ideally "what would have made this
normal" (a counterfactual). Most teams ship an opaque score; we ship a sentence.

HOW WE RANK THE CONTRIBUTING FEATURES (three complementary sources)
-------------------------------------------------------------------
  1. RULE  -- if a high-precision rule named the attack, that rule's feature IS the
     explanation (e.g. impossible_travel -> geo_velocity). Nothing is more faithful
     than the actual decision path.
  2. SHAP  -- for model-named alerts, TreeSHAP on the RandomForest gives the exact
     per-feature contribution to THIS event's predicted class. Exact (not
     approximate) because the classifier is tree-based.
  3. PER-FEATURE RECONSTRUCTION ERROR -- from the GRU autoencoder: which feature did
     the sequence model most fail to reconstruct (i.e. which behaviour was most
     surprising given history). Used for novel/unknown alerts where there is no
     confident class to SHAP-explain.

We then take the TOP-3 drivers, turn each into a templated human phrase filled with
the real values, and add a COUNTERFACTUAL ("would not have alerted if geo-velocity
< 900 km/h"). The templating is deterministic and auditable -- no LLM, no
hallucination risk -- which is exactly what a SOC wants.

Output: data/explanations.parquet  (event_id, predicted_type, risk, reason,
        top_features[json], counterfactual) -- consumed by the Phase 5 dashboard.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import joblib

try:
    from models.features import FEATURE_COLUMNS
    from models.baseline import _model_matrix
except ImportError:  # pragma: no cover
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.features import FEATURE_COLUMNS
    from models.baseline import _model_matrix

# A feature only belongs in an explanation if its value is actually ANOMALOUS,
# not merely non-zero. e.g. ip_fanin==1 is normal and must never be surfaced as
# "evidence". These floors keep the top-3 free of noise filler.
def _informative(feature: str, v: float) -> bool:
    if feature == "ip_fanin":              return v >= 2
    if feature == "failed_auth_in_window": return v >= 1
    if feature == "duration_zscore":       return abs(v) >= 1.0
    if feature == "off_hours":             return v >= 0.15
    if feature == "geo_velocity_kmh":      return v >= 100
    if feature == "cmd_novelty":           return v > 0
    if feature in ("resource_novelty", "device_mismatch", "cold_start"):
        return v >= 1.0
    return abs(v) > 1e-9


# feature -> which rule/attack it is the signature of (for rule-decided alerts)
_RULE_FEATURE = {
    "impossible_travel": "geo_velocity_kmh",
    "device_spoofing": "device_mismatch",
    "lateral_movement": "cmd_novelty",
    "low_and_slow": "off_hours",
    "credential_stuffing": "ip_fanin",
    "brute_force": "failed_auth_in_window",
}


def _short_dev(fp: str) -> str:
    """Pull 'os=...' out of a device_fingerprint for readable messages."""
    if not isinstance(fp, str):
        return "unknown"
    for part in fp.split(";"):
        if part.startswith("os="):
            return part[3:]
    return fp[:16]


def _phrase(feature: str, r: pd.Series) -> str:
    """Templated human phrase for one contributing feature, filled with real values."""
    geo = r.get("geo_location", "?"); prev = r.get("prev_city", None)
    ip = r.get("source_ip", "?"); res = r.get("resource_accessed", "?")
    if feature == "geo_velocity_kmh":
        v = r["geo_velocity_kmh"]; route = f"{prev} -> {geo}" if prev else f"to {geo}"
        mins = (r.get("hours_since_prev") or 0) * 60
        return f"geo-velocity {v:,.0f} km/h ({route}" + (f", {mins:.0f} min apart)" if mins else ")")
    if feature == "failed_auth_in_window":
        return f"{r['failed_auth_in_window']:.0f} failed logins from IP {ip} within 5 min"
    if feature == "ip_fanin":
        return f"IP {ip} hit {r['ip_fanin']:.0f} different accounts within 10 min"
    if feature == "device_mismatch":
        return f"device fingerprint (OS {_short_dev(r.get('device_fingerprint'))}) differs from this entity's known device"
    if feature == "resource_novelty":
        return f"first-ever access to resource '{res}' for this entity"
    if feature == "cmd_novelty":
        toks = r.get("command_sequence", "") or ""
        show = ",".join(toks.split(";")[:4])
        return f"unfamiliar commands ({show}) -- {r['cmd_novelty']*100:.0f}% never seen for this entity"
    if feature == "off_hours":
        return f"active at {int(r.get('event_hour', 0)):02d}:00, an unusual hour for this entity"
    if feature == "duration_zscore":
        return f"session length {r['duration_zscore']:+.1f}sigma from this entity's norm"
    if feature == "cold_start":
        return "new entity with little history (scored on population baseline)"
    return feature


_COUNTERFACTUAL = {
    "geo_velocity_kmh": "would not alert if geo-velocity stayed below 900 km/h (normal travel)",
    "failed_auth_in_window": "would not alert with fewer than 8 failed logins from this IP in 5 min",
    "ip_fanin": "would not alert if this IP touched fewer than 6 accounts in 10 min",
    "device_mismatch": "would not alert if the device fingerprint matched the entity's known device",
    "resource_novelty": "would not alert once this resource is part of the entity's normal set",
    "cmd_novelty": "would not alert if only familiar commands were used",
    "off_hours": "would not alert during this entity's habitual hours",
    "duration_zscore": "would not alert if session length were within the entity's typical range",
    "cold_start": "confidence rises automatically as the entity accrues history",
}


def _shap_matrix(clf, X: np.ndarray):
    """Return |SHAP| per (row, feature) for each row's PREDICTED class. Robust to
    SHAP version differences. Falls back to None if SHAP is unavailable/errors."""
    try:
        import shap
    except Exception:
        return None
    try:
        expl = shap.TreeExplainer(clf)
        sv = expl.shap_values(X)
        classes = list(clf.classes_)
        pred_idx = clf.predict(X)
        pred_pos = [classes.index(p) for p in pred_idx]
        n, f = X.shape
        out = np.zeros((n, f))
        if isinstance(sv, list):                    # [n_classes][n, f]
            for i in range(n):
                out[i] = np.abs(sv[pred_pos[i]][i])
        elif sv.ndim == 3:                          # [n, f, n_classes]
            for i in range(n):
                out[i] = np.abs(sv[i, :, pred_pos[i]])
        else:                                       # [n, f] (binary/edge)
            out = np.abs(sv)
        return out
    except Exception as e:
        print(f"    (SHAP unavailable, using reconstruction error only: {e})")
        return None


# Explain a wider set than the headline 2% alert budget so the dashboard's
# alert-budget slider always has a reason ready as the analyst widens the queue.
EXPLAIN_BUDGET = 0.05


def run(alerts_path="data/alerts.parquet", features_path="data/features.parquet",
        events_path="data/events.parquet", recon_path="data/seq_feat_err.parquet",
        clf_path="models/attack_clf.joblib", top_k=3) -> pd.DataFrame:
    alerts = pd.read_parquet(alerts_path)
    feats = pd.read_parquet(features_path)
    events = pd.read_parquet(events_path)[["event_id", "geo_location", "source_ip",
                                           "resource_accessed", "device_fingerprint"]]
    recon = pd.read_parquet(recon_path)
    bundle = joblib.load(clf_path)
    clf = bundle["model"]

    # explain the events an analyst could see across the budget range (top 5%)
    kexp = max(1, int(round(len(alerts) * EXPLAIN_BUDGET)))
    a = alerts.nlargest(kexp, "unified_score").copy()
    a = a.merge(feats, on=["event_id", "entity_id", "entity_type", "timestamp"], how="left")
    a = a.merge(events, on="event_id", how="left").merge(recon, on="event_id", how="left")
    a = a.reset_index(drop=True)

    X = _model_matrix(a)
    shap_abs = _shap_matrix(clf, X)                       # [n, f] or None
    recon_cols = [f"recon_{c}" for c in FEATURE_COLUMNS]

    rows_out = []
    for i in range(len(a)):
        r = a.iloc[i]
        ptype = r["predicted_type"]
        src = r["decision_source"]

        # ----- rank contributing features -----
        contrib = np.zeros(len(FEATURE_COLUMNS))
        if shap_abs is not None:
            s = shap_abs[i]
            contrib += s / (s.max() + 1e-9)
        rc = a.loc[i, recon_cols].to_numpy(dtype=float)
        contrib += rc / (rc.max() + 1e-9)
        # only surface features whose value is genuinely anomalous (not noise filler)
        active = np.array([_informative(c, float(r[c])) for c in FEATURE_COLUMNS])
        contrib = np.where(active, contrib, -1.0)
        order = list(np.argsort(-contrib))

        drivers: List[str] = []
        # rule-decided: force the rule's own feature to the front (most faithful)
        if src == "rule" and ptype in _RULE_FEATURE:
            rf = _RULE_FEATURE[ptype]
            drivers.append(rf)
        for idx in order:
            f = FEATURE_COLUMNS[idx]
            if f in drivers:
                continue
            if contrib[idx] <= 0:
                break
            drivers.append(f)
            if len(drivers) >= top_k:
                break
        if not drivers:                                   # safety: never empty
            drivers = [FEATURE_COLUMNS[int(np.argmax(rc))]]

        phrases = [_phrase(f, r) for f in drivers]
        top_features = [{"feature": f, "value": float(r[f]) if f in r else None,
                         "phrase": p} for f, p in zip(drivers, phrases)]
        reason = (f"Flagged (risk {r['risk_score']:.2f}, {ptype}): "
                  + "; ".join(phrases) + ".")
        cf = _COUNTERFACTUAL.get(drivers[0], "would not alert within the entity's normal range")
        counterfactual = f"Counterfactual: {cf}."

        rows_out.append({
            "event_id": int(r["event_id"]), "entity_id": r["entity_id"],
            "predicted_type": ptype, "risk_score": float(r["risk_score"]),
            "decision_source": src, "reason": reason,
            "counterfactual": counterfactual,
            "top_features": json.dumps(top_features),
        })

    out = pd.DataFrame(rows_out)
    out.to_parquet("data/explanations.parquet", index=False)
    _print(out, a)
    return out


def _print(out: pd.DataFrame, a: pd.DataFrame):
    print("=" * 74)
    print(f"PHASE 4A -- EXPLANATIONS GENERATED for {len(out)} alerts")
    print("=" * 74)
    # show one example per predicted type (highest risk first)
    a2 = a.copy(); a2["reason"] = out["reason"].values; a2["cf"] = out["counterfactual"].values
    seen = set()
    for _, r in a2.sort_values("risk_score", ascending=False).iterrows():
        t = r["predicted_type"]
        if t in seen:
            continue
        seen.add(t)
        tag = "TRUE-POSITIVE" if r["label"] == 1 else ("benign-drift" if r["attack_type"] == "insider_drift" else "false-positive")
        print(f"\n  [{t}]  entity={r['entity_id']}  ({tag}, truth={r['attack_type']})")
        print(f"    {r['reason']}")
        print(f"    {r['cf']}")
    print("\n" + "=" * 74)
    print("  wrote: data/explanations.parquet")


if __name__ == "__main__":
    run()
