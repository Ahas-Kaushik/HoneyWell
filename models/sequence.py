"""
sequence.py -- GRU sequence autoencoder over each entity's event stream. PHASE 3.
================================================================================

WHY A SEQUENCE MODEL (the brief's criterion #1)
-----------------------------------------------
The Phase-2 Isolation Forest scores each event in isolation, so it is strong on
attacks that spike a single event (a brute-force burst) but weak on attacks that
only reveal themselves ACROSS A SEQUENCE of events:
  * lateral movement -- novelty ESCALATES hop after hop;
  * low-and-slow exfil -- many individually-boring off-hours reads ACCUMULATE.
A model that respects ORDER and HISTORY is needed. That is what this GRU
autoencoder adds.

HOW IT WORKS (plain version)
----------------------------
For every event we take the window of the entity's last L events (that event plus
its L-1 predecessors) and ask a GRU autoencoder to compress-then-reconstruct that
window. The network is trained only on the (mostly-normal) past, so it becomes
fluent in "what this fleet's normal behaviour looks like." When the current event
does not fit the normal pattern, the reconstruction is poor -> high
RECONSTRUCTION ERROR = behavioural surprise. We use the error at the LAST
(current) position as that event's sequence-anomaly score.

STREAMING-HONEST + LABEL-BLIND
------------------------------
  * Causal: an event's window uses only that event and EARLIER events -- never the
    future. The identical windowing runs on a live stream (keep the last L feature
    vectors per entity in state).
  * Unsupervised & label-blind: the autoencoder trains to reconstruct feature
    vectors; it never sees `label`. Labels are loaded ONLY to measure PR-AUC.
  * Temporal split: we train the network on windows whose current event falls in
    the first TRAIN_FRAC of the timeline, and evaluate on the later part.

THE UNIFIED DETECTOR
--------------------
The sequence error and the Phase-2 Isolation-Forest score catch different things,
so we fuse them (rank-normalise each to [0,1], take a weighted max). We report
PR-AUC for IF-only, sequence-only, and the fused score to show the lift.

Outputs:
  models/seq_ae.pt            trained autoencoder weights + config
  data/scores_v2.parquet      event_id, if_score, seq_error, unified_score, risk, split
  data/phase3_seq_metrics.json  PR-AUC / alert-budget comparison
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

try:
    from models.features import FEATURE_COLUMNS
    from models.baseline import _precision_recall_at_k, ALERT_BUDGETS, _LOG_FEATURES
except ImportError:  # pragma: no cover
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.features import FEATURE_COLUMNS
    from models.baseline import _precision_recall_at_k, ALERT_BUDGETS, _LOG_FEATURES

SEED = 42
SEQ_LEN = 16                 # events of history per window
HIDDEN = 32                  # GRU hidden size
LATENT = 16                  # bottleneck (forces compression, prevents copy)
EPOCHS = 18
BATCH = 256
LR = 1e-3
TRAIN_FRAC = 0.70
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _set_seed(s=SEED):
    np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------------------
# Data prep: per-entity causal windows of standardized feature vectors
# ---------------------------------------------------------------------------
def _standardized_matrix(feats: pd.DataFrame, train_mask: np.ndarray):
    """log1p the heavy-tailed features, then standardize using TRAIN stats only."""
    X = feats[FEATURE_COLUMNS].copy()
    for c in _LOG_FEATURES:
        X[c] = np.log1p(X[c].clip(lower=0))
    X = X.to_numpy(dtype=np.float32)
    mu = X[train_mask].mean(0, keepdims=True)
    sd = X[train_mask].std(0, keepdims=True) + 1e-6
    return (X - mu) / sd, mu, sd


def _build_windows(feats: pd.DataFrame, Xstd: np.ndarray) -> np.ndarray:
    """For each event, the [SEQ_LEN, F] window of the entity's last SEQ_LEN events.

    Left-padded with zeros for entities with fewer than SEQ_LEN prior events.
    Windows are aligned to the feats row order (which is time-sorted).
    """
    F = Xstd.shape[1]
    N = len(feats)
    W = np.zeros((N, SEQ_LEN, F), dtype=np.float32)
    # group rows by entity, preserving time order
    idx_by_entity: Dict[str, List[int]] = {}
    for pos, eid in enumerate(feats["entity_id"].to_numpy()):
        idx_by_entity.setdefault(eid, []).append(pos)
    for eid, positions in idx_by_entity.items():
        for j, pos in enumerate(positions):
            lo = max(0, j - SEQ_LEN + 1)
            ctx = positions[lo:j + 1]
            w = Xstd[ctx]
            W[pos, SEQ_LEN - len(w):, :] = w      # left-pad
    return W


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class GRUAutoencoder(nn.Module):
    """Encode a window with a GRU, bottleneck to a latent, decode back the window."""
    def __init__(self, n_features: int, hidden=HIDDEN, latent=LATENT, seq_len=SEQ_LEN):
        super().__init__()
        self.seq_len = seq_len
        self.enc = nn.GRU(n_features, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.dec = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, n_features)

    def forward(self, x):
        _, h = self.enc(x)                     # h: [1, B, hidden]
        z = self.to_latent(h[-1])              # [B, latent]
        d0 = torch.tanh(self.from_latent(z))   # [B, hidden]
        dec_in = d0.unsqueeze(1).repeat(1, self.seq_len, 1)  # broadcast latent over time
        y, _ = self.dec(dec_in)
        return self.out(y)                     # [B, seq_len, F]


def _train(model, Wtr, epochs=EPOCHS):
    model.to(DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    lossf = nn.MSELoss()
    Wtr_t = torch.from_numpy(Wtr)
    n = len(Wtr_t)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, BATCH):
            b = Wtr_t[perm[i:i + BATCH]].to(DEVICE)
            opt.zero_grad()
            rec = model(b)
            loss = lossf(rec, b)
            loss.backward(); opt.step()
            tot += loss.item() * len(b)
        if ep == 0 or (ep + 1) % 6 == 0 or ep == epochs - 1:
            print(f"    epoch {ep+1:2d}/{epochs}  train MSE {tot/n:.4f}")


@torch.no_grad()
def _per_event_error(model, W):
    """Reconstruction error at the LAST (current) position of each window.

    Returns (errs, per_feature_errs):
      errs             [N]    mean squared error of the current event (scalar score)
      per_feature_errs [N,F]  squared error broken down BY FEATURE -- Phase 4 uses
                              this to say WHICH feature the model failed to explain
                              (per-feature reconstruction error = an explanation).
    """
    model.eval()
    F = W.shape[2]
    errs = np.zeros(len(W), dtype=np.float64)
    per_feat = np.zeros((len(W), F), dtype=np.float64)
    Wt = torch.from_numpy(W)
    for i in range(0, len(Wt), 1024):
        b = Wt[i:i + 1024].to(DEVICE)
        rec = model(b)
        sq = (rec[:, -1, :] - b[:, -1, :]) ** 2            # [B, F]
        errs[i:i + len(b)] = sq.mean(dim=1).cpu().numpy()
        per_feat[i:i + len(b)] = sq.cpu().numpy()
    return errs, per_feat


def _rank01(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).rank(pct=True).to_numpy()


def _cdf_map(train_vals: np.ndarray, all_vals: np.ndarray) -> np.ndarray:
    """Map values to their percentile under the TRAIN distribution (empirical CDF)."""
    xs = np.sort(np.asarray(train_vals, dtype=np.float64))
    return np.searchsorted(xs, np.asarray(all_vals, dtype=np.float64), side="right") / max(len(xs), 1)


def run(features_path="data/features.parquet", labels_path="data/labels.parquet",
        scores_path="data/scores.parquet", fuse_weight=0.5) -> dict:
    _set_seed()
    feats = pd.read_parquet(features_path).sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    labels = pd.read_parquet(labels_path)
    y = feats.merge(labels, on="event_id")["label"].to_numpy()

    cut = feats["timestamp"].quantile(TRAIN_FRAC)
    train_mask = (feats["timestamp"] <= cut).to_numpy()
    test_mask = ~train_mask

    # ---- build standardized causal windows ----
    Xstd, mu, sd = _standardized_matrix(feats, train_mask)
    print("  building causal per-entity windows ...")
    W = _build_windows(feats, Xstd)

    # ---- train autoencoder unsupervised on train-window slice ----
    print(f"  training GRU autoencoder on {int(train_mask.sum()):,} windows (device={DEVICE}) ...")
    model = GRUAutoencoder(n_features=len(FEATURE_COLUMNS))
    _train(model, W[train_mask])

    # ---- per-event sequence reconstruction error (+ per-feature breakdown) ----
    seq_err, per_feat_err = _per_event_error(model, W)
    pfe = pd.DataFrame(per_feat_err, columns=[f"recon_{c}" for c in FEATURE_COLUMNS])
    pfe.insert(0, "event_id", feats["event_id"].to_numpy())
    pfe.to_parquet("data/seq_feat_err.parquet", index=False)

    # ---- fuse with the Phase-2 Isolation-Forest score ----
    if os.path.exists(scores_path):
        sc = pd.read_parquet(scores_path)[["event_id", "anomaly_score"]]
        merged = feats[["event_id"]].merge(sc, on="event_id", how="left")
        if_score = merged["anomaly_score"].to_numpy()
    else:
        if_score = np.zeros(len(feats))
    # Normalise each detector against the TRAIN-split distribution (a fixed
    # reference CDF). This is streaming-honest -- in production you calibrate each
    # detector against accumulated history, not against the current batch -- and
    # unlike global rank-normalisation it maps every event consistently, so the
    # fused ordering is stable regardless of which window you evaluate on.
    if_c = _cdf_map(if_score[train_mask], if_score)
    seq_c = _cdf_map(seq_err[train_mask], seq_err)
    # Fuse by elementwise MAX: an attack that EITHER detector nails stays flagged,
    # rather than being diluted by the other missing it (single-signal spikes vs
    # sequential attacks are exactly complementary). MAX preserves a brute-force
    # spike AND a lateral-movement sequence; averaging would dilute both.
    if_r, seq_r = if_c, seq_c
    unified = np.maximum(if_c, seq_c)

    # ---- metrics on TEST split ----
    def _prauc(s): return float(average_precision_score(y[test_mask], s[test_mask]))
    m = {
        "PR_AUC": {"isolation_forest": _prauc(if_r), "sequence_ae": _prauc(seq_r),
                   "unified_max": _prauc(unified)},
        "alert_budget_unified": [_precision_recall_at_k(unified[test_mask], y[test_mask], b)
                                 for b in ALERT_BUDGETS],
        "config": {"seq_len": SEQ_LEN, "hidden": HIDDEN, "latent": LATENT,
                   "epochs": EPOCHS, "device": DEVICE},
    }

    # ---- persist ----
    torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd,
                "feature_columns": FEATURE_COLUMNS, "seq_len": SEQ_LEN,
                "hidden": HIDDEN, "latent": LATENT}, "models/seq_ae.pt")
    out = feats[["event_id", "entity_id", "timestamp"]].copy()
    out["if_score"] = if_score
    out["seq_error"] = seq_err
    out["unified_score"] = unified
    out["risk_score"] = _rank01(unified)
    out["split"] = np.where(train_mask, "train", "test")
    out.to_parquet("data/scores_v2.parquet", index=False)
    with open("data/phase3_seq_metrics.json", "w") as f:
        json.dump(m, f, indent=2)

    _print(m)
    return m


def _print(m: dict):
    print("=" * 70)
    print("PHASE 3A -- SEQUENCE AUTOENCODER + UNIFIED DETECTOR  |  TEST-SPLIT PR-AUC")
    print("=" * 70)
    p = m["PR_AUC"]
    print(f"  Isolation Forest (Phase 2) : {p['isolation_forest']:.3f}")
    print(f"  Sequence AE  (Phase 3)     : {p['sequence_ae']:.3f}")
    print(f"  UNIFIED (max of the two)   : {p['unified_max']:.3f}   <-- detection model")
    lift = (p['unified_max'] - p['isolation_forest'])
    print(f"  lift over Phase-2 baseline : {lift:+.3f}")
    print("-" * 70)
    print("  UNIFIED alert-budget table:")
    print(f"      {'budget':>7} {'#alerts':>8} {'precision':>10} {'recall':>8} {'TPs':>5}")
    for b in m["alert_budget_unified"]:
        print(f"      {b['budget']*100:6.1f}% {b['k_events']:8d} "
              f"{b['precision']:10.3f} {b['recall']:8.3f} {b['true_positives']:5d}")
    print("=" * 70)
    print("  wrote: models/seq_ae.pt, data/scores_v2.parquet, data/phase3_seq_metrics.json")


if __name__ == "__main__":
    run()
