"""
geo.py -- Geographic reference data + distance math shared across the project.

Why this lives in its own module
---------------------------------
`geo_location` in the event schema is stored as a human-readable "City, CC"
string (what a SOC analyst actually wants to read in an alert). But the single
most explainable anomaly feature -- *geo-velocity* (km travelled / hours
elapsed between an entity's consecutive logins) -- needs latitude/longitude.

Keeping the city->coordinate table and the haversine function here means the
GENERATOR (Phase 1) and the FEATURE LAYER (Phase 2) agree on exactly the same
coordinates. If they disagreed, geo-velocity would be silently wrong.

Everything here is pure/deterministic -- no randomness, no I/O -- so it is safe
to import from both the batch generator and a streaming feature service.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# City coordinate table (lat, lon) in decimal degrees.
#
# The mix is deliberate: a cluster of Indian metros (short, plausible domestic
# travel), plus far-flung global cities so that "impossible travel" injections
# produce physically absurd velocities (e.g. Berlin -> Mumbai in 8 minutes).
# Houston / Phoenix are included for Honeywell's US industrial/OT footprint.
# ---------------------------------------------------------------------------
CITY_COORDS: Dict[str, Tuple[float, float]] = {
    "Mumbai, IN":     (19.0760,  72.8777),
    "Bengaluru, IN":  (12.9716,  77.5946),
    "Delhi, IN":      (28.6139,  77.2090),
    "Hyderabad, IN":  (17.3850,  78.4867),
    "Chennai, IN":    (13.0827,  80.2707),
    "Pune, IN":       (18.5204,  73.8567),
    "London, GB":     (51.5074,  -0.1278),
    "Berlin, DE":     (52.5200,  13.4050),
    "Frankfurt, DE":  (50.1109,   8.6821),
    "New York, US":   (40.7128, -74.0060),
    "Houston, US":    (29.7604, -95.3698),
    "Phoenix, US":    (33.4484,-112.0740),
    "Singapore, SG":  (1.3521,  103.8198),
    "Tokyo, JP":      (35.6895, 139.6917),
    "Sydney, AU":     (-33.8688,151.2093),
    "Dubai, AE":      (25.2048,  55.2708),
    "Sao Paulo, BR":  (-23.5505,-46.6333),
}

# Convenience groupings used by the generator to build realistic profiles.
INDIA_CITIES = [c for c in CITY_COORDS if c.endswith(", IN")]
GLOBAL_CITIES = [c for c in CITY_COORDS if not c.endswith(", IN")]

# Radius of the Earth in kilometres (mean).
_EARTH_RADIUS_KM = 6371.0088


def haversine_km(city_a: str, city_b: str) -> float:
    """Great-circle distance in km between two known cities.

    Returns 0.0 when the two cities are identical (the common case for a
    stationary entity), which keeps geo-velocity at 0 without special-casing.
    """
    if city_a == city_b:
        return 0.0
    lat1, lon1 = CITY_COORDS[city_a]
    lat2, lon2 = CITY_COORDS[city_b]
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def geo_velocity_kmh(city_a: str, city_b: str, hours_elapsed: float) -> float:
    """Implied travel speed (km/h) to get from city_a to city_b in the given time.

    This is the core "impossible travel" signal. A commercial flight tops out
    around ~900 km/h, so anything materially above that between two *successful*
    consecutive logins of the same entity is physically implausible and a strong
    account-compromise indicator.

    `hours_elapsed` is clamped to a small positive floor so two near-simultaneous
    logins from different continents yield a huge (but finite) velocity rather
    than a divide-by-zero. This clamp is intentionally streaming-safe: it needs
    only the previous event's city + timestamp held in per-entity state.
    """
    dist = haversine_km(city_a, city_b)
    if dist == 0.0:
        return 0.0
    hours = max(hours_elapsed, 1.0 / 3600.0)  # floor at 1 second
    return dist / hours
