"""
app.py -- Analyst-facing behavioral-anomaly dashboard (Streamlit). PHASE 5.
================================================================================

Two audiences, one app:
  * A SOC ANALYST: a ranked alert queue (worst first), a one-glance risk score, a
    one-sentence reason + counterfactual per alert, per-entity timelines, an
    alert-budget slider, and confirm/dismiss feedback. This is the "would a tired
    analyst at 2am trust and use this?" product (brief criterion #4 / section 4.6).
  * YOU (learning the system): a "How it works" tab and a "Model evaluation" tab
    that show the architecture, the eight attack patterns, and every metric/figure
    -- so the dashboard teaches the project while it demos it.

Run:  streamlit run dashboard/app.py

Design choices tied to the brief:
  * The OPERATIONAL views never rely on the hidden `label` -- they run on the same
    label-blind scores the model produces at inference. A "reveal ground truth"
    toggle exists FOR THE DEMO ONLY and is clearly marked as not-available-in-prod.
  * The alert-budget slider makes the core SOC trade-off tangible: widen the budget
    -> catch more attacks but review more events.
  * Alerts are grouped into INCIDENTS (de-duplication) so a 25-event brute-force
    burst is one row, not twenty-five.
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FIG = os.path.join(ROOT, "figures")
FEEDBACK_CSV = os.path.join(DATA, "analyst_feedback.csv")

ATTACK_INFO = {
    "brute_force": ("Repeated failed logins on ONE account from ONE IP.",
                    "Count of failed logins from a source IP in a short window."),
    "credential_stuffing": ("A few IPs spraying breached passwords across MANY accounts.",
                            "Distinct accounts hit per source IP (fan-in)."),
    "impossible_travel": ("Same account, two logins too far apart to be one person.",
                          "Geo-velocity (km / hours) between consecutive logins > 900 km/h."),
    "lateral_movement": ("A compromised account pivoting across systems it never used.",
                         "Resource-novelty + recon commands (whoami, psexec, dump_creds)."),
    "device_spoofing": ("An attacker impersonating a trusted device.",
                        "Device fingerprint (OS/MAC) differs from the entity's history."),
    "low_and_slow": ("Data stolen in tiny off-hours dribbles to stay under the radar.",
                     "Off-hours + sensitive resource + external-egress commands, accumulated."),
    "insider_drift": ("BENIGN: a legitimate user slowly expanding their footprint.",
                      "In-hours, home geo, internal resources, no egress -> NOT flagged."),
    "novel/unknown": ("Clearly anomalous but matches no known signature -> investigate.",
                      "High anomaly score + classifier can only call it benign/low-confidence."),
    "normal": ("Habitual, expected behavior.", "Matches the entity's learned baseline."),
}

RISK_COLORS = {"critical": "#c0392b", "high": "#e67e22", "medium": "#f1c40f", "low": "#7f8c8d"}


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_all():
    def p(f):
        return os.path.join(DATA, f)
    alerts = pd.read_parquet(p("alerts.parquet"))
    expl = pd.read_parquet(p("explanations.parquet"))
    feats = pd.read_parquet(p("features.parquet"))
    events = pd.read_parquet(p("events.parquet"))
    try:
        cal = pd.read_parquet(p("calibrated_scores.parquet"))
        alerts = alerts.merge(cal, on="event_id", how="left")
    except Exception:
        alerts["calibrated_prob"] = alerts["risk_score"]
    alerts["calibrated_prob"] = alerts["calibrated_prob"].fillna(alerts["risk_score"])
    try:
        profiles = {pr["entity_id"]: pr for pr in json.load(open(p("entity_profiles.json")))}
    except Exception:
        profiles = {}
    try:
        metrics = json.load(open(p("phase4_metrics.json")))
    except Exception:
        metrics = {}
    return alerts, expl, feats, events, profiles, metrics


def risk_band(x: float) -> str:
    return "critical" if x >= 0.9 else "high" if x >= 0.75 else "medium" if x >= 0.5 else "low"


def _log_feedback(event_id, predicted_type, decision):
    """Append an analyst confirm/dismiss decision (human-in-the-loop, brief Section 5)."""
    row = pd.DataFrame([{"ts": datetime.now().isoformat(), "event_id": int(event_id),
                         "predicted_type": predicted_type, "decision": decision}])
    header = not os.path.exists(FEEDBACK_CSV)
    row.to_csv(FEEDBACK_CSV, mode="a", header=header, index=False)


def dedup_incidents(df: pd.DataFrame, gap_min=30) -> pd.DataFrame:
    """Collapse same-entity+type alerts within `gap_min` into one incident row."""
    if df.empty:
        return df.assign(incident_id=[], burst_size=[])
    d = df.sort_values(["entity_id", "predicted_type", "timestamp"]).copy()
    ids, gid, prev = [], -1, None
    for _, r in d.iterrows():
        key = (r["entity_id"], r["predicted_type"])
        if prev is None or key != prev[0] or (r["timestamp"] - prev[1]) > pd.Timedelta(minutes=gap_min):
            gid += 1
        ids.append(gid); prev = (key, r["timestamp"])
    d["incident_id"] = ids
    size = d.groupby("incident_id")["event_id"].transform("size")
    d["burst_size"] = size
    # keep the highest-risk event as the incident representative
    rep = d.sort_values("unified_score", ascending=False).groupby("incident_id").head(1)
    return rep.sort_values("unified_score", ascending=False)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Behavioral Anomaly Detection - SOC Console",
                   layout="wide")

if not os.path.exists(os.path.join(DATA, "alerts.parquet")):
    st.error("No data found. Run the pipeline first: generator -> features -> baseline "
             "-> sequence -> classifier -> explain -> metrics.")
    st.stop()

alerts, expl, feats, events, profiles, metrics = load_all()
alerts = alerts.merge(expl[["event_id", "reason", "counterfactual", "top_features"]],
                      on="event_id", how="left")

st.title("Behavioral Anomaly Detection — SOC Console")
st.caption("Per-entity behavioral baselining · sequence-aware detection · attack "
           "attribution · per-alert explanations — framed for OT / industrial-edge security.")

# ---- sidebar controls ----
with st.sidebar:
    st.header("Controls")
    budget = st.slider("Alert budget (top-% of events an analyst reviews)",
                       0.1, 5.0, 2.0, 0.1,
                       help="The core SOC trade-off: widen to catch more attacks, "
                            "but review more events.") / 100.0
    teach = st.toggle("Teaching mode (extra captions)", value=True)
    reveal = st.toggle("Reveal ground truth (DEMO ONLY)", value=False,
                       help="Shows the hidden label. Not available in production — "
                            "the model never sees it.")
    group = st.toggle("Group bursts into incidents (de-dup)", value=True)
    st.divider()
    n_total = len(alerts)
    k = max(1, int(round(n_total * budget)))
    st.metric("Events in 'live' window", f"{n_total:,}")
    st.metric("Alert budget", f"top {budget*100:.1f}%  ->  {k} events")
    if teach:
        st.info("This window is the temporal **test split** (the 'future' the model "
                "scored after training on the past).")

# ---- current alert set from budget ----
queue = alerts.nlargest(k, "unified_score").copy()
if group:
    shown = dedup_incidents(queue)
else:
    shown = queue.assign(burst_size=1)

tab_q, tab_e, tab_m, tab_h = st.tabs(
    ["Alert Queue", "Entity Investigation", "Model Evaluation", "How it works"])

# ===========================================================================
# TAB 1 — ALERT QUEUE
# ===========================================================================
with tab_q:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Alerts (incidents)" if group else "Alerts", f"{len(shown):,}")
    c2.metric("Entities at risk", f"{shown['entity_id'].nunique():,}")
    crit = int((shown["calibrated_prob"] >= 0.9).sum())
    c3.metric("Critical (P≥0.90)", f"{crit}")
    if reveal:
        tp = int((shown["label"] == 1).sum())
        c4.metric("True attacks in queue", f"{tp}/{len(shown)}",
                  help="Demo only — precision of the current queue.")
    else:
        c4.metric("Largest burst grouped", f"{int(shown['burst_size'].max())}->1" if group else "—")

    if teach:
        st.info("Each row is worst-first by **calibrated risk** (a real probability). "
                "Expand a row for the one-sentence reason, the counterfactual, the "
                "contributing features, and confirm/dismiss.")

    st.subheader("Ranked alert queue")
    for _, r in shown.iterrows():
        band = risk_band(r["calibrated_prob"])
        dot = RISK_COLORS[band]
        burst = f" · ×{int(r['burst_size'])}" if group and r["burst_size"] > 1 else ""
        truth = ""
        if reveal:
            ok = "" if r["label"] == 1 else ("benign-drift" if r["attack_type"] == "insider_drift" else "FP")
            truth = f" — truth: `{r['attack_type']}` {ok}"
        header = (f"**{r['calibrated_prob']*100:.0f}%** · `{r['predicted_type']}` · "
                  f"{r['entity_id']} ({r['entity_type']}) · {r['timestamp']:%Y-%m-%d %H:%M}{burst}{truth}")
        with st.expander(header):
            st.markdown(f"<span style='color:{dot};font-weight:600'>{band.upper()} RISK "
                        f"({r['calibrated_prob']*100:.0f}% probability of attack)</span>",
                        unsafe_allow_html=True)
            st.write(f"**Why:** {r.get('reason', '—')}")
            st.write(f"**{r.get('counterfactual', '')}**")
            st.caption(f"decision path: {r['decision_source']}")

            # contributing features as bars
            tf = r.get("top_features")
            if isinstance(tf, str) and tf:
                try:
                    items = json.loads(tf)
                    tfx = pd.DataFrame([{"feature": it["feature"], "value": it.get("value", 0) or 0}
                                        for it in items]).set_index("feature")
                    st.bar_chart(tfx, height=160)
                    for it in items:
                        st.caption(f"• {it['phrase']}")
                except Exception:
                    pass

            # raw event context
            ev = events[events["event_id"] == r["event_id"]]
            if not ev.empty:
                with st.popover("Raw event"):
                    st.json(ev.iloc[0].to_dict(), expanded=False)

            # analyst feedback (bonus: human-in-the-loop, brief Section 5)
            fb1, fb2, _ = st.columns([1, 1, 4])
            if fb1.button("Confirm", key=f"c{r['event_id']}"):
                _log_feedback(r["event_id"], r["predicted_type"], "confirmed")
                st.success("Logged as confirmed.")
            if fb2.button("Dismiss", key=f"d{r['event_id']}"):
                _log_feedback(r["event_id"], r["predicted_type"], "dismissed")
                st.info("Logged as dismissed (feeds future active learning).")

    st.subheader("Queue table")
    cols = ["timestamp", "entity_id", "entity_type", "predicted_type",
            "calibrated_prob", "decision_source", "burst_size"]
    if reveal:
        cols += ["attack_type", "label"]
    st.dataframe(shown[cols].rename(columns={"calibrated_prob": "risk_prob"}),
                 width="stretch", hide_index=True)


# ===========================================================================
# TAB 2 — ENTITY INVESTIGATION
# ===========================================================================
with tab_e:
    st.subheader("Per-entity timeline")
    ent_options = sorted(alerts["entity_id"].unique())
    default_ent = shown["entity_id"].iloc[0] if len(shown) else ent_options[0]
    ent = st.selectbox("Entity", ent_options, index=ent_options.index(default_ent))

    # profile card
    prof = profiles.get(ent, {})
    if prof:
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Type", prof.get("entity_type", "?"))
        p2.metric("Home geo", prof.get("home_city", "?"))
        hh = prof.get("habitual_hours", [])
        p3.metric("Habitual hours", f"{min(hh):02d}-{max(hh):02d}h" if hh else "?")
        p4.metric("Usual resources", str(len(prof.get("usual_resources", []))))
        if teach:
            st.caption("The model learned this entity's 'normal' from data; the profile "
                       "above is the ground-truth baseline for comparison.")

    ent_ev = alerts[alerts["entity_id"] == ent].sort_values("timestamp")
    if not ent_ev.empty:
        chart = ent_ev[["timestamp", "calibrated_prob"]].set_index("timestamp")
        st.line_chart(chart, height=220)
        st.caption("Risk over time for this entity (calibrated probability).")
        show_cols = ["timestamp", "predicted_type", "calibrated_prob", "reason"]
        if reveal:
            show_cols += ["attack_type", "label"]
        st.dataframe(ent_ev[show_cols].rename(columns={"calibrated_prob": "risk_prob"}),
                     width="stretch", hide_index=True)


# ===========================================================================
# TAB 3 — MODEL EVALUATION
# ===========================================================================
with tab_m:
    st.subheader("How good is it? (offline evaluation — uses hidden labels)")
    if teach:
        st.info("Operational views never use labels. This tab does, to MEASURE the "
                "model — reporting the imbalance-robust metrics the brief requires, "
                "never raw accuracy.")
    det = metrics.get("detection", {})
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PR-AUC (primary)", f"{det.get('PR_AUC', 0):.3f}", help="vs ~0.01 random")
    m2.metric("ROC-AUC", f"{det.get('ROC_AUC', 0):.3f}")
    budget_tbl = det.get("alert_budget_table", [])
    r2 = next((b for b in budget_tbl if abs(b["budget"] - 0.02) < 1e-6), None)
    if r2:
        m3.metric("Recall @ top-2%", f"{r2['recall']*100:.0f}%")
        m4.metric("Precision @ top-2%", f"{r2['precision']*100:.0f}%")

    cal = metrics.get("calibration", {})
    dd = metrics.get("deduplication", {})
    st.caption(f"Calibration Brier {cal.get('brier_raw_rankscore',0):.3f} -> "
               f"{cal.get('brier_calibrated',0):.4f} (isotonic) · "
               f"De-dup {dd.get('raw_alerts','?')} alerts -> {dd.get('grouped_incidents','?')} incidents")

    g1, g2 = st.columns(2)
    for col, f, cap in [(g1, "pr_curve_final.png", "Precision-Recall (imbalance-robust)"),
                        (g2, "alert_budget_final.png", "Alert-budget curve: recall vs % reviewed"),
                        (g1, "confusion_matrix.png", "Attack-type attribution (alerts)"),
                        (g2, "calibration.png", "Reliability: risk score = real probability")]:
        fp = os.path.join(FIG, f)
        if os.path.exists(fp):
            col.image(fp, caption=cap, width="stretch")

    pt = metrics.get("per_type_recall", {}).get("top_2pct", {})
    if pt:
        st.subheader("Detection recall by attack type @ top-2% (every category covered)")
        st.dataframe(pd.DataFrame([{"attack_type": t, "recall": v["recall"], "n": v["n"]}
                                   for t, v in pt.items()]).sort_values("recall", ascending=False),
                     width="stretch", hide_index=True)


# ===========================================================================
# TAB 4 — HOW IT WORKS
# ===========================================================================
with tab_h:
    st.subheader("The pipeline (event -> risk + reason)")
    st.markdown("""
```
 access-log stream
        |
        v
 (1) FEATURE SERVICE  (incremental per-entity state — streaming-ready)
    geo-velocity · failed-auth-in-window · resource-novelty · device-mismatch
    off-hours · duration z-score · command-novelty · IP fan-in · cold-start
        |
        v
 (2) DETECTOR  (unsupervised, label-blind)
    Isolation Forest  +  GRU sequence autoencoder   -> fused anomaly score
        |
        v
 (3) CLASSIFIER  (rules + RandomForest + novel/unknown)  -> attack type
        |
        v
 (4) EXPLAINER  (SHAP + per-feature reconstruction error) -> one-sentence reason
        |                                                   + counterfactual
        v
 risk score · attack type · human reason · analyst feedback loop
```
Every feature is computable from small per-entity/per-IP running state, so the
identical logic runs on a live stream (state cached in e.g. Redis). We demo in
batch; the design streams.
""")
    st.subheader("The eight behaviors it knows")
    for name, (what, how) in ATTACK_INFO.items():
        if name == "normal":
            continue
        tag = "benign edge case" if name == "insider_drift" else (
            "catch-all" if name == "novel/unknown" else "attack")
        st.markdown(f"**{name}** — {tag}  \n{what}  \n*Detection handle:* {how}")
