"""Fast Lambert-based transfer calculations.

Provides functions for computing faster transfers that use more delta-V
than the minimum-energy Hohmann transfer.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
from scipy.optimize import brentq

from ..constants import GM_SUN, AU_TO_M, KM_TO_M
from ..lambert import lambert_solve
from ..porkchop import TransferTrajectory
from .base import hohmann_tof as _hohmann_tof, OrbitGeometry
from .hohmann import hohmann_delta_v, hohmann_trajectory

logger = logging.getLogger(__name__)


@dataclass
class TransferSearchResult:
    best_i: Optional[int]
    best_j: Optional[int]
    launch_date: Optional[datetime]
    tof_days: Optional[float]


def _find_best_transfer_from_porkchop(porkchop_result, target_dv_km_s):
    """Find minimum-TOF transfer within delta-V budget from porkchop grid.

    Uses numpy masking instead of Python loop iteration.

    Returns
    -------
    TransferSearchResult
        Contains best_i, best_j, launch_date, tof_days or None values if not found
    """
    grid = porkchop_result.grid
    tof_days = porkchop_result.tof_days
    date_labels = porkchop_result.date_labels

    feasible = np.isfinite(grid) & (grid <= target_dv_km_s)
    if not np.any(feasible):
        return TransferSearchResult(None, None, None, None)

    tof_grid = np.broadcast_to(tof_days, grid.shape)
    masked_tof = np.where(feasible, tof_grid, np.inf)
    min_idx = np.unravel_index(np.argmin(masked_tof), grid.shape)
    return TransferSearchResult(min_idx[0], min_idx[1], date_labels[min_idx[0]], tof_days[min_idx[1]])


def search_transfer_ecliptic(
    dep_body_name,
    arr_body_name,
    target_dv,
    points,
):
    """Search for fastest transfer using actual ephemeris positions.

    Parameters
    ----------
    dep_body_name : str
        Name of departure body (e.g., "Mars", "15312 Wandt").
    arr_body_name : str
        Name of arrival body.
    target_dv : float
        Available delta-V budget in m/s.
    points : int
        Number of trajectory points.

    Returns
    -------
    TransferTrajectory or None
        The computed transfer trajectory or None if no solution found.
    float
        Hohmann transfer time in seconds.
    """
    from ..porkchop import compute_porkchop, compute_lambert_trajectory
    from .visualization.core import get_body_ecliptic

    dep_r, _ = get_body_ecliptic(dep_body_name)
    arr_r, _ = get_body_ecliptic(arr_body_name)
    ht = _hohmann_tof(dep_r * AU_TO_M, arr_r * AU_TO_M)

    target_dv_km_s = target_dv / 1000.0

    result = compute_porkchop(
        dep_body_name,
        arr_body_name,
        start_date=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
        num_dates=146,
        date_step=5,
        tof_min=50,
        tof_max=400,
        num_tofs=71,
    )

    search_result = _find_best_transfer_from_porkchop(result, target_dv_km_s)

    if search_result.launch_date is None:
        return None, ht

    traj = compute_lambert_trajectory(
        dep_body_name, arr_body_name, search_result.launch_date, search_result.tof_days, points=points
    )

    tof = search_result.tof_days * 86400.0
    return TransferTrajectory(traj.x, traj.y, traj.dep_burn, traj.arr_burn, np.pi, tof), ht


def calc_dv_for_factor(
    r1, r2, factor, dep_geom: OrbitGeometry = None, arr_geom: OrbitGeometry = None
):
    """Compute total delta-V for a transfer orbit scaled by a factor."""
    from .base import OrbitalBody

    if dep_geom is None:
        dep_geom = OrbitGeometry()
    if arr_geom is None:
        arr_geom = OrbitGeometry()

    a = ((r1 + r2) / 2) * factor
    vt1 = np.sqrt(GM_SUN * (2 / r1 - 1 / a))
    v_arr_mag = np.sqrt(GM_SUN * (2 / r2 - 1 / a))
    v_theta = r1 * vt1 / r2
    v_radial_sq = v_arr_mag**2 - v_theta**2
    v_radial = np.sqrt(v_radial_sq) if v_radial_sq > 0 else 0.0

    dep_body = OrbitalBody(r1 / KM_TO_M, r1 / KM_TO_M, eccentricity=dep_geom.eccentricity)
    arr_body = OrbitalBody(r2 / KM_TO_M, r2 / KM_TO_M, eccentricity=arr_geom.eccentricity)

    v_planet_dep = dep_body.velocity_at(-dep_geom.rotation, 0.0)
    v_transfer_dep = np.array([0.0, vt1, 0.0])
    dv_dep = np.linalg.norm(v_transfer_dep - v_planet_dep)

    dnu = np.pi
    v_transfer_arr = np.array(
        [
            v_radial * np.cos(dnu) - v_theta * np.sin(dnu),
            v_radial * np.sin(dnu) + v_theta * np.cos(dnu),
            0.0,
        ]
    )
    arr_nu = dnu - arr_geom.rotation
    v_planet_arr = arr_body.velocity_at(arr_nu, dnu)
    dv_arr = np.linalg.norm(v_transfer_arr - v_planet_arr)

    return dv_dep + dv_arr


def find_factor_for_dv(
    r1,
    r2,
    target_dv,
    max_factor=50.0,
    dep_geom: OrbitGeometry = None,
    arr_geom: OrbitGeometry = None,
):
    """Find the energy factor that uses a given delta-V budget."""
    if dep_geom is None:
        dep_geom = OrbitGeometry()
    if arr_geom is None:
        arr_geom = OrbitGeometry()

    _, _, dv_total = hohmann_delta_v(r1, r2)
    if target_dv < dv_total:
        return 1.0, dv_total

    inward = r2 < r1

    def residual(factor):
        return (
            calc_dv_for_factor(r1, r2, factor, dep_geom, arr_geom)
            - target_dv
        )

    try:
        if inward:
            min_factor = r1 / (r1 + r2) * 1.001
            factor = brentq(residual, min_factor, 1.0, xtol=1e-10, rtol=1e-12)
        else:
            factor = brentq(residual, 1.0, max_factor, xtol=1e-10, rtol=1e-12)
    except ValueError:
        factor = 1.0 if inward else max_factor

    return factor, calc_dv_for_factor(r1, r2, factor, dep_geom, arr_geom)


def search_transfer(
    r1, r2, target_dv, points, dep_geom: OrbitGeometry, arr_geom: OrbitGeometry
):
    """Search for the fastest transfer within a delta-V budget.

    Uses coarse-to-fine grid refinement: coarse sweep over (tof, dnu),
    then refine around the best candidate.

    Parameters
    ----------
    r1 : float
        Semi-major axis of departure orbit in meters.
    r2 : float
        Semi-major axis of arrival orbit in meters.
    target_dv : float
        Available delta-V budget in m/s.
    points : int
        Number of trajectory points.
    dep_geom : OrbitGeometry
        Departure orbit geometry (eccentricity, rotation).
    arr_geom : OrbitGeometry
        Arrival orbit geometry (eccentricity, rotation).

    Returns
    -------
    tuple
        (best, hohmann_tof_s) where best is (tof, dnu, v1, r1_vec, r2_actual)
        or None if no solution found.
    """
    from .base import OrbitalBody

    ht = _hohmann_tof(r1, r2)

    dep_body = OrbitalBody(r1 / KM_TO_M, r1 / KM_TO_M, eccentricity=dep_geom.eccentricity)
    arr_body = OrbitalBody(r2 / KM_TO_M, r2 / KM_TO_M, eccentricity=arr_geom.eccentricity)

    dep_nu = -dep_geom.rotation
    r1_actual = dep_body.radius_at(dep_nu)
    r1_vec = np.array([r1_actual, 0.0, 0.0])
    v_planet_dep = dep_body.velocity_at(dep_nu, 0.0)
    r1_mag = r1_actual

    best = None
    best_dv = float("inf")
    min_tof = None

    def _evaluate(tof, dnu):
        nonlocal best, best_dv, min_tof
        orbit_angle = dnu - arr_geom.rotation
        r2_actual = arr_body.radius_at(orbit_angle)
        v_planet_arr = arr_body.velocity_at(orbit_angle, dnu)
        r2_vec = np.array([r2_actual * np.cos(dnu), r2_actual * np.sin(dnu), 0.0])
        try:
            v1, v2 = lambert_solve(r1_vec, r2_vec, tof)
        except ValueError:
            return
        dv_dep = np.linalg.norm(v1 - v_planet_dep)
        dv_arr = np.linalg.norm(v2 - v_planet_arr)
        total_dv = dv_dep + dv_arr
        if total_dv > target_dv:
            return
        v1_mag = np.linalg.norm(v1)
        specific_energy = v1_mag**2 / 2.0 - GM_SUN / r1_mag
        if specific_energy >= 0:
            return
        a_transfer = -GM_SUN / (2.0 * specific_energy)
        if a_transfer <= 0:
            return
        if total_dv < best_dv:
            best = (tof, dnu, v1, r1_vec, r2_actual)
            best_dv = total_dv
            min_tof = tof

    coarse_tof = np.linspace(0.01, 1.0, 150)
    coarse_dnu = np.linspace(0.01, np.pi, 80)
    for tof_frac in coarse_tof:
        tof = ht * tof_frac
        for dnu in coarse_dnu:
            _evaluate(tof, dnu)

    if best is not None and min_tof is not None:
        dnu_values_fine = np.linspace(
            max(0.01, best[1] - 0.15), min(np.pi, best[1] + 0.15), 80
        )
        tof_fine = np.linspace(
            max(0.01, min_tof / ht - 0.05), min(1.0, min_tof / ht + 0.05), 40
        )
        for tof_frac in tof_fine:
            tof = ht * tof_frac
            for dnu in dnu_values_fine:
                _evaluate(tof, dnu)

    if best is None:
        logger.debug(
            "search_transfer exhausted: r1=%.3e, r2=%.3e, target_dv=%.3e — "
            "no Lambert solution with valid semi-major axis found within budget",
            r1,
            r2,
            target_dv,
        )

    return best, ht


def _hohmann_fallback(r1, r2, points, target_ecc, target_rot):
    """Fallback to Hohmann trajectory when search_transfer fails.

    Returns the same shape as compute_transfer_trajectory.
    """
    x, y = hohmann_trajectory(r1, r2, points)
    ht = _hohmann_tof(r1, r2)
    return TransferTrajectory(
        x=x,
        y=y,
        dep_burn=np.array([r1 / AU_TO_M, 0.0]),
        arr_burn=np.array([-r2 / AU_TO_M, 0.0]),
        dnu=np.pi,
        tof=ht,
    )
