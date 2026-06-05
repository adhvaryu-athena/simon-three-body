"""
didymos_dimorphos_horizons_feasibility.py

Feasibility check for a real NASA/JPL Horizons three-body experiment:
    Sun + Didymos + Dimorphos

Purpose:
    1. Fetch initial Cartesian state vectors from JPL Horizons.
    2. Compute the Didymos-Dimorphos initial separation.
    3. Compute their relative velocity.
    4. Estimate a rough two-body orbital period from separation and masses.
    5. Check whether the binary separation is near SIMON's NN activation threshold.

This is NOT a SIMON rollout yet. It only checks whether the real-system setup is usable.

Run:
    python didymos_dimorphos_horizons_feasibility.py

Optional examples:
    python didymos_dimorphos_horizons_feasibility.py --epoch 2022-09-25
    python didymos_dimorphos_horizons_feasibility.py --epoch 2026-01-01
    python didymos_dimorphos_horizons_feasibility.py --didymos-mass-kg 5.24e11 --dimorphos-mass-kg 4.8e9

Requirements:
    pip install astroquery astropy numpy

Notes:
    - Horizons object names for small bodies can vary. This script tries multiple
      candidate identifiers for Didymos and Dimorphos.
    - If Dimorphos is not available as a separate Horizons target, the script will
      report that clearly. In that case, a true JPL-state-vector experiment is not
      straightforward.
    - Default asteroid masses are approximate scale values for feasibility only.
      Use literature values if you want a publishable orbital-period estimate.
"""

import argparse
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

try:
    from astroquery.jplhorizons import Horizons
except ImportError as exc:
    raise SystemExit(
        "[ERROR] astroquery is not installed. Install with:\n"
        "    pip install astroquery astropy numpy\n"
    ) from exc


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
AU_KM = 149_597_870.700
AU_M = AU_KM * 1000.0
DAY_S = 86400.0
YEAR_DAYS = 365.25
G_SI = 6.67430e-11
M_SUN_KG = 1.98847e30

# SIMON thresholds from the paper/code.
SIMON_EPS_AU = 3e-4
SIMON_NN_THRESHOLD_AU = 500.0 * SIMON_EPS_AU       # 0.15 AU
SIMON_ADAPT_THRESHOLD_AU = 0.05
SIMON_R_SOFT_MIN_AU = 5e-4


@dataclass
class HorizonsState:
    label: str
    query_id: str
    id_type: Optional[str]
    pos_au: np.ndarray
    vel_au_per_day: np.ndarray
    raw_targetname: str

    @property
    def vel_au_per_year(self) -> np.ndarray:
        return self.vel_au_per_day * YEAR_DAYS


def fetch_vector_for_candidate(
    label: str,
    query_id: str,
    id_type: Optional[str],
    epoch: str,
    location: str = "@0",
) -> HorizonsState:
    """Fetch one Horizons vector for one candidate object identifier."""
    epochs = {"start": epoch, "stop": _next_day(epoch), "step": "1d"}

    kwargs = {"id": query_id, "location": location, "epochs": epochs}
    if id_type is not None:
        kwargs["id_type"] = id_type

    obj = Horizons(**kwargs)
    vec = obj.vectors()

    pos = np.array(
        [float(vec["x"][0]), float(vec["y"][0]), float(vec["z"][0])],
        dtype=np.float64,
    )
    vel = np.array(
        [float(vec["vx"][0]), float(vec["vy"][0]), float(vec["vz"][0])],
        dtype=np.float64,
    )

    targetname = str(vec["targetname"][0]) if "targetname" in vec.colnames else "unknown"

    return HorizonsState(
        label=label,
        query_id=query_id,
        id_type=id_type,
        pos_au=pos,
        vel_au_per_day=vel,
        raw_targetname=targetname,
    )


def fetch_first_working(
    label: str,
    candidates: List[Tuple[str, Optional[str]]],
    epoch: str,
    location: str = "@0",
) -> HorizonsState:
    """Try candidate identifiers until one returns a vector."""
    errors = []
    for query_id, id_type in candidates:
        try:
            state = fetch_vector_for_candidate(label, query_id, id_type, epoch, location)
            print(f"  [OK] {label}: query_id={query_id!r}, id_type={id_type!r}")
            print(f"       Horizons targetname: {state.raw_targetname}")
            return state
        except Exception as exc:  # Horizons raises several different exception types.
            msg = f"query_id={query_id!r}, id_type={id_type!r}: {type(exc).__name__}: {exc}"
            errors.append(msg)
            print(f"  [try failed] {label}: {msg}")

    joined = "\n    ".join(errors)
    raise RuntimeError(
        f"Could not fetch {label} from Horizons using the candidate identifiers.\n"
        f"Tried:\n    {joined}"
    )


def _next_day(epoch: str) -> str:
    """Return a simple next-day string for YYYY-MM-DD input."""
    # Avoid adding pandas/dateutil dependency.
    import datetime as _dt

    try:
        d = _dt.datetime.strptime(epoch, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("Use epoch format YYYY-MM-DD, e.g. 2022-09-25") from exc
    return (d + _dt.timedelta(days=1)).isoformat()


def norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))


def estimate_two_body_period_days(separation_au: float, mass1_kg: float, mass2_kg: float) -> float:
    """
    Rough circular two-body period estimate from instantaneous separation.

    T = 2*pi*sqrt(r^3 / (G*(m1+m2)))

    This is only a feasibility-scale diagnostic. It is not a fitted binary orbit.
    """
    r_m = separation_au * AU_M
    mu = G_SI * (mass1_kg + mass2_kg)
    if r_m <= 0 or mu <= 0:
        return float("nan")
    return 2.0 * math.pi * math.sqrt(r_m**3 / mu) / DAY_S


def classify_against_simon_thresholds(separation_au: float) -> List[str]:
    notes = []
    if separation_au < SIMON_R_SOFT_MIN_AU:
        notes.append("below SIMON safety threshold: NN would be bypassed by fallback")
    elif separation_au < SIMON_NN_THRESHOLD_AU:
        notes.append("inside SIMON NN activation threshold r < 0.15 AU")
    else:
        notes.append("outside SIMON NN activation threshold r >= 0.15 AU")

    if separation_au < SIMON_ADAPT_THRESHOLD_AU:
        notes.append("inside adaptive sub-stepping threshold r < 0.05 AU")
    else:
        notes.append("outside adaptive sub-stepping threshold r >= 0.05 AU")

    if SIMON_R_SOFT_MIN_AU <= separation_au <= 0.01:
        notes.append("near-softening regime: NN correction could be numerically meaningful")
    elif separation_au < SIMON_NN_THRESHOLD_AU:
        notes.append("NN may activate, but softening correction may be small if r >> eps")

    return notes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--epoch",
        default="2022-09-25",
        help="Epoch in YYYY-MM-DD format. Default is 2022-09-25, near the DART encounter period.",
    )
    parser.add_argument(
        "--didymos-mass-kg",
        type=float,
        default=5.24e11,
        help="Approximate Didymos mass in kg. Used only for rough period estimate.",
    )
    parser.add_argument(
        "--dimorphos-mass-kg",
        type=float,
        default=4.8e9,
        help="Approximate Dimorphos mass in kg. Used only for rough period estimate.",
    )
    args = parser.parse_args()

    Horizons.TIMEOUT = 120

    print("DIDYMOS-DIMORPHOS + SUN JPL HORIZONS FEASIBILITY CHECK")
    print("=" * 78)
    print(f"Epoch: {args.epoch}")
    print("Location: @0 Solar System barycentre")
    print()

    # Sun is reliable as numeric major-body ID 10.
    sun_candidates = [("10", None)]

    # Horizons naming can vary. Try names first for readability, then numbered designations.
    didymos_candidates = [
        ("920065803", None),       # Didymos primary center, reconstructed JPL solution
        ("20065803", None),       # Didymos system barycenter
        ("Didymos", "smallbody"),
        ("65803", "smallbody"),
        ("1996 GT", "smallbody"),
    ]

    dimorphos_candidates = [
        ("120065803", None),       # Dimorphos, satellite of Didymos, reconstructed JPL solution
        ("Dimorphos", None),
        ("Dimorphos", "smallbody"),
        ("Didymos I", "smallbody"),
        ("65803 I", "smallbody"),
        ("S/2003 (65803) 1", "smallbody"),
        ("(65803) Didymos I Dimorphos", "smallbody"),
    ]

    print("[fetching Horizons vectors]")
    sun = fetch_first_working("Sun", sun_candidates, args.epoch)
    didymos = fetch_first_working("Didymos", didymos_candidates, args.epoch)
    dimorphos = fetch_first_working("Dimorphos", dimorphos_candidates, args.epoch)

    print()
    print("[state vectors: AU and AU/day]")
    for state in [sun, didymos, dimorphos]:
        print(f"  {state.label} ({state.raw_targetname})")
        print(f"    r = [{state.pos_au[0]: .12e}, {state.pos_au[1]: .12e}, {state.pos_au[2]: .12e}] AU")
        print(
            f"    v = [{state.vel_au_per_day[0]: .12e}, "
            f"{state.vel_au_per_day[1]: .12e}, {state.vel_au_per_day[2]: .12e}] AU/day"
        )

    # Relative Didymos-Dimorphos diagnostics.
    rel_pos_au = dimorphos.pos_au - didymos.pos_au
    rel_vel_au_day = dimorphos.vel_au_per_day - didymos.vel_au_per_day

    sep_au = norm(rel_pos_au)
    sep_km = sep_au * AU_KM
    rel_speed_au_day = norm(rel_vel_au_day)
    rel_speed_km_s = rel_speed_au_day * AU_KM / DAY_S
    rel_speed_m_s = rel_speed_km_s * 1000.0

    period_days = estimate_two_body_period_days(
        sep_au,
        args.didymos_mass_kg,
        args.dimorphos_mass_kg,
    )

    print()
    print("[Didymos-Dimorphos relative diagnostics]")
    print(f"  separation = {sep_au:.12e} AU")
    print(f"  separation = {sep_km:.6f} km")
    print(f"  relative speed = {rel_speed_au_day:.12e} AU/day")
    print(f"  relative speed = {rel_speed_km_s:.9f} km/s = {rel_speed_m_s:.6f} m/s")
    print(f"  rough circular two-body period estimate = {period_days:.6f} days")
    print(f"  rough circular two-body period estimate = {period_days / YEAR_DAYS:.9f} yr")

    print()
    print("[SIMON threshold check]")
    print(f"  eps = {SIMON_EPS_AU:.8e} AU")
    print(f"  r_soft_min = {SIMON_R_SOFT_MIN_AU:.8e} AU")
    print(f"  adaptive threshold = {SIMON_ADAPT_THRESHOLD_AU:.8e} AU")
    print(f"  NN threshold = {SIMON_NN_THRESHOLD_AU:.8e} AU")
    print(f"  separation / eps = {sep_au / SIMON_EPS_AU:.6e}")
    print(f"  separation / NN_threshold = {sep_au / SIMON_NN_THRESHOLD_AU:.6e}")
    for note in classify_against_simon_thresholds(sep_au):
        print(f"  - {note}")

    print()
    print("[mass values used only for rough period estimate]")
    print(f"  Didymos mass = {args.didymos_mass_kg:.6e} kg = {args.didymos_mass_kg / M_SUN_KG:.6e} solar masses")
    print(f"  Dimorphos mass = {args.dimorphos_mass_kg:.6e} kg = {args.dimorphos_mass_kg / M_SUN_KG:.6e} solar masses")

    print()
    print("[next decision]")
    print("  If Dimorphos was fetched successfully and the separation is reasonable, the next script can run:")
    print("    ias15 reference first, then SIMON adaptive ON, then optional No-NN baseline.")
    print("  If Dimorphos could not be fetched, we should not force this experiment from incomplete data.")


if __name__ == "__main__":
    main()
