"""Fast Lambert-based transfer calculations.

Provides functions for computing faster transfers that use more delta-V
than the minimum-energy Hohmann transfer.
"""

import logging

import numpy as np
from scipy.optimize import brentq

from ..constants import GM_SUN, AU_TO_M
from ..lambert import lambert_solve
from .base import compute_r2_actual, planet_velocity, hohmann_tof as _hohmann_tof

from .hohmann import hohmann_delta_v

logger = logging.getLogger(__name__)


def _find_best_transfer_from_porkchop(porkchop_result, target_dv_km_s):
    """Find minimum-TOF transfer within delta-V budget from porkchop grid.

    Uses numpy masking instead of Python loop iteration.

    Returns
    -------
    tuple
        (best_i, best_j, launch_date, tof_days) or (None, None, None, None) if not found
    """
    grid = porkchop_result.grid
    tof_days = porkchop_result.tof_days
    date_labels = porkchop_result.date_labels

    feasible = np.isfinite(grid) & (grid <= target_dv_km_s)
    if not np.any(feasible):
        return None, None, None, None

    tof_grid = np.broadcast_to(tof_days, grid.shape)
    masked_tof = np.where(feasible, tof_grid, np.inf)
    min_idx = np.unravel_index(np.argmin(masked_tof), grid.shape)
    return min_idx[0], min_idx[1], date_labels[min_idx[0]], tof_days[min_idx[1]]


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
    tuple
        (x, y, dep_burn, arr_burn, dnu, tof) or None if no solution found.
    """
    from datetime import datetime
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

    _, _, launch_date, tof_days = _find_best_transfer_from_porkchop(
        result, target_dv_km_s
    )

    if launch_date is None:
        return None, ht

    traj = compute_lambert_trajectory(
        dep_body_name, arr_body_name, launch_date, tof_days, points=points
    )

    tof = tof_days * 86400.0
    return (traj.x, traj.y, traj.dep_burn, traj.arr_burn, np.pi, tof), ht


def calc_dv_for_factor(
    r1, r2, factor, dep_ecc=0.0, dep_rot=0.0, arr_ecc=0.0, arr_rot=0.0
):
    """Compute total delta-V for a transfer orbit scaled by a factor."""
    a = ((r1 + r2) / 2) * factor
    vt1 = np.sqrt(GM_SUN * (2 / r1 - 1 / a))
    v_arr_mag = np.sqrt(GM_SUN * (2 / r2 - 1 / a))
    v_theta = r1 * vt1 / r2
    v_radial_sq = v_arr_mag**2 - v_theta**2
    v_radial = np.sqrt(v_radial_sq) if v_radial_sq > 0 else 0.0

    v_planet_dep = planet_velocity(r1, dep_ecc, -dep_rot, 0.0)
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
    arr_nu = dnu - arr_rot
    v_planet_arr = planet_velocity(r2, arr_ecc, arr_nu, dnu)
    dv_arr = np.linalg.norm(v_transfer_arr - v_planet_arr)

    return dv_dep + dv_arr


def find_factor_for_dv(
    r1,
    r2,
    target_dv,
    max_factor=50.0,
    dep_ecc=0.0,
    dep_rot=0.0,
    arr_ecc=0.0,
    arr_rot=0.0,
):
    """Find the energy factor that uses a given delta-V budget."""
    _, _, dv_total = hohmann_delta_v(r1, r2)
    if target_dv < dv_total:
        return 1.0, dv_total

    inward = r2 < r1

    def residual(factor):
        return (
            calc_dv_for_factor(r1, r2, factor, dep_ecc, dep_rot, arr_ecc, arr_rot)
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

    return factor, calc_dv_for_factor(
        r1, r2, factor, dep_ecc, dep_rot, arr_ecc, arr_rot
    )


def search_transfer(
    r1, r2, target_dv, points, target_ecc, target_rot, dep_ecc, dep_rot
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
    target_ecc : float
        Eccentricity of the target orbit.
    target_rot : float
        Rotation angle of the target orbit in radians.
    dep_ecc : float
        Eccentricity of the departure orbit.
    dep_rot : float
        Rotation angle of the departure orbit in radians.

    Returns
    -------
    tuple
        (best, hohmann_tof_s) where best is (tof, dnu, v1, r1_vec, r2_actual)
        or None if no solution found.
    """
    ht = _hohmann_tof(r1, r2)

    dep_nu = -dep_rot
    r1_actual = compute_r2_actual(r1, dep_ecc, dep_nu)
    r1_vec = np.array([r1_actual, 0.0, 0.0])
    v_planet_dep = planet_velocity(r1, dep_ecc, dep_nu, 0.0)
    r1_mag = r1_actual

    best = None
    best_dv = float("inf")
    min_tof = None

    def _evaluate(tof, dnu):
        nonlocal best, best_dv, min_tof
        orbit_angle = dnu - target_rot
        r2_actual = compute_r2_actual(r2, target_ecc, orbit_angle)
        v_planet_arr = planet_velocity(r2, target_ecc, orbit_angle, dnu)
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
