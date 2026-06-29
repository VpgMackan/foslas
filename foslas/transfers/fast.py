"""Fast Lambert-based transfer calculations.

Provides functions for computing faster transfers that use more delta-V
than the minimum-energy Hohmann transfer.
"""

import numpy as np
from scipy.optimize import brentq

from ..constants import GM_SUN, AU_TO_M
from ..lambert import lambert_solve
from ..integrator import integrate_trajectory
from .base import compute_r2_actual, planet_velocity


def calc_dv_for_factor(
    r1, r2, factor, dep_ecc=0.0, dep_rot=0.0, arr_ecc=0.0, arr_rot=0.0
):
    """Compute total delta-V for a transfer orbit scaled by a factor.

    Parameters
    ----------
    r1 : float
        Semi-major axis of departure orbit in meters.
    r2 : float
        Semi-major axis of arrival orbit in meters.
    factor : float
        Energy factor scaling the semi-major axis.
    dep_ecc : float, optional
        Eccentricity of the departure orbit (default: 0.0).
    dep_rot : float, optional
        Rotation angle of the departure orbit in radians (default: 0.0).
    arr_ecc : float, optional
        Eccentricity of the arrival orbit (default: 0.0).
    arr_rot : float, optional
        Rotation angle of the arrival orbit in radians (default: 0.0).

    Returns
    -------
    float
        Total delta-V in m/s.
    """
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
    """Find the energy factor that uses a given delta-V budget.

    Parameters
    ----------
    r1 : float
        Semi-major axis of departure orbit in meters.
    r2 : float
        Semi-major axis of arrival orbit in meters.
    target_dv : float
        Available delta-V budget in m/s.
    max_factor : float, optional
        Maximum energy factor to search (default: 50.0).
    dep_ecc : float, optional
        Eccentricity of the departure orbit (default: 0.0).
    dep_rot : float, optional
        Rotation angle of the departure orbit in radians (default: 0.0).
    arr_ecc : float, optional
        Eccentricity of the arrival orbit (default: 0.0).
    arr_rot : float, optional
        Rotation angle of the arrival orbit in radians (default: 0.0).

    Returns
    -------
    tuple of float
        (factor, actual_delta_v) where factor scales the Hohmann
        semi-major axis and actual_delta_v is the delta-V consumed.
    """
    from .hohmann import hohmann_delta_v

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

    Brute-force searches over time-of-flight fractions and arrival angles
    to find Lambert solutions that fit within the delta-V budget.

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
        (best, hohmann_tof) where best is (tof, dnu, v1, r1_vec, r2_actual)
        or None if no solution found.
    """
    hohmann_tof = np.pi * np.sqrt(((r1 + r2) / 2) ** 3 / GM_SUN)

    tof_fractions = np.linspace(0.05, 0.95, 60)
    dnu_values = np.linspace(0.3, np.pi - 0.05, 40)

    dep_nu = -dep_rot
    r1_actual = compute_r2_actual(r1, dep_ecc, dep_nu)
    r1_vec = np.array([r1_actual, 0.0, 0.0])
    v_planet_dep = planet_velocity(r1, dep_ecc, dep_nu, 0.0)

    best = None
    for tof_frac in tof_fractions:
        tof = hohmann_tof * tof_frac
        for dnu in dnu_values:
            orbit_angle = dnu - target_rot
            r2_actual = compute_r2_actual(r2, target_ecc, orbit_angle)
            v_planet_arr = planet_velocity(r2, target_ecc, orbit_angle, dnu)
            r2_vec = np.array([r2_actual * np.cos(dnu), r2_actual * np.sin(dnu), 0.0])
            try:
                v1, v2 = lambert_solve(r1_vec, r2_vec, tof)
            except ValueError:
                continue
            dv_dep = np.linalg.norm(v1 - v_planet_dep)
            dv_arr = np.linalg.norm(v2 - v_planet_arr)
            total_dv = dv_dep + dv_arr
            if total_dv <= target_dv:
                if best is None or tof < best[0]:
                    best = (tof, dnu, v1, r1_vec, r2_actual)
        if best is not None:
            break

    return best, hohmann_tof


def compute_fast_trajectory(r1, r2, target_dv, target_ecc, target_rot, dep_ecc, dep_rot, points=500):
    """Compute a fast transfer trajectory using Lambert's solver.

    Parameters
    ----------
    r1 : float
        Semi-major axis of departure orbit in meters.
    r2 : float
        Semi-major axis of arrival orbit in meters.
    target_dv : float
        Available delta-V budget in m/s.
    target_ecc : float
        Eccentricity of the target orbit.
    target_rot : float
        Rotation angle of the target orbit in radians.
    dep_ecc : float
        Eccentricity of the departure orbit.
    dep_rot : float
        Rotation angle of the departure orbit in radians.
    points : int, optional
        Number of trajectory points (default: 500).

    Returns
    -------
    tuple
        (x, y, dep_burn, arr_burn, dnu, tof) where x, y are trajectory
        coordinates in AU.
    """
    hohmann_tof = np.pi * np.sqrt(((r1 + r2) / 2) ** 3 / GM_SUN)

    best, _ = search_transfer(
        r1, r2, target_dv, points, target_ecc, target_rot, dep_ecc, dep_rot
    )

    if best is None:
        tof = hohmann_tof * 0.5
        dnu = np.pi * 0.75
        orbit_angle = dnu - target_rot
        r2_actual = compute_r2_actual(r2, target_ecc, orbit_angle)
        dep_nu = -dep_rot
        r1_actual = compute_r2_actual(r1, dep_ecc, dep_nu)
        r1_vec = np.array([r1_actual, 0.0, 0.0])
        r2_vec = np.array([r2_actual * np.cos(dnu), r2_actual * np.sin(dnu), 0.0])
        try:
            v1, v2 = lambert_solve(r1_vec, r2_vec, tof)
        except ValueError:
            from .hohmann import hohmann_trajectory
            x, y = hohmann_trajectory(r1, r2, points)
            return (
                x,
                y,
                np.array([r1 / AU_TO_M, 0.0]),
                np.array([-r2 / AU_TO_M, 0.0]),
                np.pi,
                hohmann_tof,
            )
        best = (tof, dnu, v1, r1_vec, r2_actual)

    tof, dnu, v1, r1_vec, r2_actual = best
    positions, _ = integrate_trajectory(r1_vec, v1, tof, points)

    x = positions[:, 0] / AU_TO_M
    y = positions[:, 1] / AU_TO_M
    dep_nu = -dep_rot
    r1_actual = compute_r2_actual(r1, dep_ecc, dep_nu)
    dep_burn = np.array([r1_actual / AU_TO_M, 0.0])
    arr_burn = np.array(
        [r2_actual * np.cos(dnu) / AU_TO_M, r2_actual * np.sin(dnu) / AU_TO_M]
    )

    return x, y, dep_burn, arr_burn, dnu, tof
