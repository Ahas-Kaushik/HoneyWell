"""
features.py -- Incremental (streaming-ready) behavioral feature layer. PHASE 2.
================================================================================

WHAT THIS DOES
--------------
Turns the raw access-log stream (`data/events.parquet`) into a per-event numeric
feature vector that describes *how far this event deviates from what is normal for
THIS entity*. The detector in `baseline.py` then scores those vectors.

THE ONE IDEA THAT MAKES THIS WORK
---------------------------------
Every feature is computed RELATIVE TO THE ENTITY'S OWN HISTORY, and that history
is maintained as a small running state that is updated one event at a time, in
timestamp order. Two big consequences:

  1. Streaming-ready (brief criterion #6). Nothing here needs a full-table scan or
     a look at the future. Each event is processed with O(1)-ish work against a
     tiny per-entity / per-IP state object. The exact same code could run on a live
     Kafka stream; we just happen to replay a batch here. Where production would
     put that state (Redis, per-entity) is noted per feature.

  2. Label-blind (brief failure mode: "reading the label at inference"). This file
     imports ONLY `data/events.parquet`. It never touches `labels.parquet`. The
     feature layer literally cannot leak the answer.

THE NINE FEATURES (exactly the set the brief's cheat-sheet asks for)
--------------------------------------------------------------------
  geo_velocity_kmh      km / hours between this and the entity's previous login.
                        Huge => "impossible travel". State: last (city, time).
  failed_auth_in_window count of failed logins from THIS source_ip in the last
                        5 min. Spikes => brute force / stuffing pressure.
                        State: per-IP sliding window of failed-auth timestamps.
  resource_novelty      1 if this resource is new to the entity (else 0). Bursts
                        of novelty => lateral movement. State: per-entity resource set.
  device_mismatch       1 if device_fingerprint isn't one the entity has used
                        before. => device spoofing. State: per-entity device set.
  off_hours             how *rare* this hour is for the entity (0=normal hour,
                        ->1=never-seen hour). State: per-entity 24-bin hour histogram.
  duration_zscore       (session_duration - entity mean) / entity std, computed
                        online (Welford). Big |z| => unusual session length.
  cmd_novelty           fraction of this event's command tokens the entity has
                        never run. Recon/egress commands => lateral / exfil.
                        State: per-entity command-token set.
  ip_fanin              # of DISTINCT entities that used this source_ip in the last
                        10 min. High => credential stuffing (few IPs, many users).
                        State: per-IP sliding window of (time, entity).
  cold_start            1 if the entity has < MIN_HISTORY prior events (no reliable
                        personal baseline yet). Drives the cold-start policy below.

THE COLD-START POLICY (brief criterion #5 -- "don't blanket-flag new entities")
-------------------------------------------------------------------------------
A brand-new entity has no personal history, so *everything* looks novel. Naively,
that flags every new user/device as an attack -- a named failure mode. Our policy:
while an entity is in cold-start (fewer than MIN_HISTORY events), we NEUTRALISE the
features that are only meaningful relative to personal history
(resource_novelty, device_mismatch, off_hours, duration_zscore, cmd_novelty -> 0).
The features that are physical or cross-entity and still valid with no personal
history (geo_velocity, failed_auth_in_window, ip_fanin) stay live. Net effect: a
new entity is scored on universally-valid signals only and is NOT auto-flagged.
The `cold_start` flag is emitted so the model, the metrics, and the dashboard can
all see "this was a low-confidence, new-entity score" -- and Phase 3 can swap in a
peer-group baseline here.

Failed logins (session_duration == 0 on `auth/login`, see generator) are excluded
from the duration statistics and do NOT advance the entity's physical location --
a failed attempt isn't evidence the user is really there.
"""

from __future__ import annotations

from collections import deque, defaultdict
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    from generator.geo import geo_velocity_kmh, CITY_COORDS
except ImportError:  # pragma: no cover
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from generator.geo import geo_velocity_kmh, CITY_COORDS

# ---- knobs -----------------------------------------------------------------
MIN_HISTORY = 5                       # events before an entity leaves cold-start
FAILED_AUTH_WINDOW_MIN = 5            # sliding window for failed-auth count
IP_FANIN_WINDOW_MIN = 10             # sliding window for distinct-entities-per-IP
OFF_HOURS_RARE_THRESHOLD = 0.03      # an hour rarer than 3% of history is "off-hours"
AUTH_RESOURCE = "auth/login"

# The 9 model features, in a fixed order (baseline.py depends on this order).
FEATURE_COLUMNS: List[str] = [
    "geo_velocity_kmh",
    "failed_auth_in_window",
    "resource_novelty",
    "device_mismatch",
    "off_hours",
    "duration_zscore",
    "cmd_novelty",
    "ip_fanin",
    "cold_start",
]

# Extra human-readable columns carried for the dashboard / explanations (NOT fed
# to the model). They make an alert legible to an analyst.
DISPLAY_COLUMNS: List[str] = ["is_failed_auth", "event_hour", "prev_city", "hours_since_prev"]

# Features that are only meaningful relative to personal history -> suppressed
# during cold-start (see policy in the module docstring).
_HISTORY_RELATIVE = ("resource_novelty", "device_mismatch", "off_hours",
                     "duration_zscore", "cmd_novelty")


class _EntityState:
    """Tiny per-entity running state -- the entire 'memory' of one entity.

    In production this object is what you'd cache in Redis keyed by entity_id and
    update on each incoming event. It is deliberately small and O(1) to update.
    """
    __slots__ = ("count", "last_city", "last_ts", "resources", "devices",
                 "hour_hist", "dur_n", "dur_mean", "dur_m2", "cmd_tokens")

    def __init__(self):
        self.count = 0
        self.last_city = None
        self.last_ts = None
        self.resources: set = set()
        self.devices: set = set()
        self.hour_hist = np.zeros(24, dtype=np.float64)
        self.dur_n = 0            # Welford online-variance accumulators
        self.dur_mean = 0.0
        self.dur_m2 = 0.0
        self.cmd_tokens: set = set()

    def duration_z(self, x: float) -> float:
        """z-score of x against the running mean/std BEFORE incorporating x."""
        if self.dur_n < 2:
            return 0.0
        var = self.dur_m2 / (self.dur_n - 1)
        std = np.sqrt(var)
        if std < 1e-9:
            return 0.0
        return float((x - self.dur_mean) / std)

    def update_duration(self, x: float) -> None:
        """Welford online mean/variance update (successful sessions only)."""
        self.dur_n += 1
        d = x - self.dur_mean
        self.dur_mean += d / self.dur_n
        self.dur_m2 += d * (x - self.dur_mean)


def _tokens(cmd: str) -> List[str]:
    if not cmd:
        return []
    return [t for t in cmd.split(";") if t]


class FeatureEngine:
    """Stateful, single-pass feature extractor.

    Call `process_event(row_dict)` for each event IN TIMESTAMP ORDER; it returns a
    dict of the 9 features + display columns and mutates internal state. The batch
    helper `transform(events_df)` does the whole stream and returns a DataFrame.
    """

    def __init__(self):
        self.entities: Dict[str, _EntityState] = defaultdict(_EntityState)
        # per-IP sliding windows (deques of timestamps / (timestamp, entity))
        self.ip_failed: Dict[str, deque] = defaultdict(deque)
        self.ip_fanin: Dict[str, deque] = defaultdict(deque)
        self._failed_td = pd.Timedelta(minutes=FAILED_AUTH_WINDOW_MIN)
        self._fanin_td = pd.Timedelta(minutes=IP_FANIN_WINDOW_MIN)

    # -- per-IP windowed helpers ------------------------------------------
    def _failed_auth_count(self, ip: str, ts, is_failed: bool) -> int:
        dq = self.ip_failed[ip]
        if is_failed:
            dq.append(ts)
        while dq and (ts - dq[0]) > self._failed_td:
            dq.popleft()
        return len(dq)

    def _ip_fanin(self, ip: str, ts, entity: str) -> int:
        dq = self.ip_fanin[ip]
        dq.append((ts, entity))
        while dq and (ts - dq[0][0]) > self._fanin_td:
            dq.popleft()
        return len({e for _, e in dq})

    # -- main --------------------------------------------------------------
    def process_event(self, r: dict) -> dict:
        eid = r["entity_id"]
        ts = r["timestamp"]
        ip = r["source_ip"]
        city = r["geo_location"]
        resource = r["resource_accessed"]
        dur = float(r["session_duration"])
        cmd = r["command_sequence"] if isinstance(r["command_sequence"], str) else ""

        st = self.entities[eid]
        is_failed = (dur == 0.0 and resource == AUTH_RESOURCE)
        cold = st.count < MIN_HISTORY

        # ---- geo-velocity (valid even in cold-start: physical constraint) ----
        if st.last_city is not None and city in CITY_COORDS and st.last_city in CITY_COORDS:
            hours = max((ts - st.last_ts).total_seconds() / 3600.0, 1.0 / 3600.0)
            geo_v = geo_velocity_kmh(st.last_city, city, hours)
            hours_since = hours
            prev_city = st.last_city
        else:
            geo_v, hours_since, prev_city = 0.0, np.nan, None

        # ---- per-IP windowed features (cross-entity: valid in cold-start) ----
        failed_cnt = self._failed_auth_count(ip, ts, is_failed)
        fanin = self._ip_fanin(ip, ts, eid)

        # ---- history-relative features (suppressed during cold-start) --------
        resource_novelty = 0.0 if resource in st.resources else 1.0
        device_mismatch = 0.0 if (len(st.devices) == 0 or r["device_fingerprint"] in st.devices) else 1.0
        # off-hours rarity: 0 for a habitual hour, ->1 for a never/rarely-seen hour
        total = st.hour_hist.sum()
        hour = int(ts.hour)
        if total > 0:
            freq = st.hour_hist[hour] / total
            off_hours = float(max(0.0, 1.0 - freq / OFF_HOURS_RARE_THRESHOLD))
            off_hours = min(off_hours, 1.0)
        else:
            off_hours = 0.0
        dur_z = 0.0 if is_failed else st.duration_z(dur)
        toks = _tokens(cmd)
        if toks:
            new_toks = sum(1 for t in toks if t not in st.cmd_tokens)
            cmd_novelty = new_toks / len(toks)
        else:
            cmd_novelty = 0.0

        feat = {
            "geo_velocity_kmh": geo_v,
            "failed_auth_in_window": float(failed_cnt),
            "resource_novelty": resource_novelty,
            "device_mismatch": device_mismatch,
            "off_hours": off_hours,
            "duration_zscore": dur_z,
            "cmd_novelty": cmd_novelty,
            "ip_fanin": float(fanin),
            "cold_start": 1.0 if cold else 0.0,
            # display-only
            "is_failed_auth": 1.0 if is_failed else 0.0,
            "event_hour": hour,
            "prev_city": prev_city,
            "hours_since_prev": hours_since,
        }

        # ---- COLD-START POLICY: neutralise personal-history features ----------
        if cold:
            for k in _HISTORY_RELATIVE:
                feat[k] = 0.0

        # ---- UPDATE STATE (after scoring, so features are causal/past-only) ---
        st.count += 1
        st.hour_hist[hour] += 1.0
        st.resources.add(resource)
        st.devices.add(r["device_fingerprint"])
        for t in toks:
            st.cmd_tokens.add(t)
        if not is_failed:                       # a failed login is not physical presence
            st.update_duration(dur)
            st.last_city = city if city in CITY_COORDS else st.last_city
            st.last_ts = ts
        return feat

    def transform(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """Process an entire event stream (must be time-sortable) -> feature frame."""
        df = events_df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        rows = []
        for r in df.to_dict("records"):
            f = self.process_event(r)
            f["event_id"] = r["event_id"]
            f["entity_id"] = r["entity_id"]
            f["entity_type"] = r["entity_type"]
            f["timestamp"] = r["timestamp"]
            rows.append(f)
        out = pd.DataFrame(rows)
        cols = ["event_id", "entity_id", "entity_type", "timestamp"] + FEATURE_COLUMNS + DISPLAY_COLUMNS
        return out[cols]


def build_features(events_path: str = "data/events.parquet",
                   out_path: str = "data/features.parquet") -> pd.DataFrame:
    """Load events, run the streaming feature engine, persist + return features."""
    events = pd.read_parquet(events_path)
    assert "label" not in events.columns, "events.parquet must be label-blind"
    feats = FeatureEngine().transform(events)
    if out_path:
        feats.to_parquet(out_path, index=False)
    return feats


if __name__ == "__main__":
    f = build_features()
    print(f"features built: {f.shape[0]:,} rows x {len(FEATURE_COLUMNS)} model features")
    print("wrote data/features.parquet")
    print(f.head())
