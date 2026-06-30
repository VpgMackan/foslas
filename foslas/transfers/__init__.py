"""Transfer trajectory computations.

Provides the main entry point for computing transfer trajectories,
coordinating between Hohmann and fast Lambert-based transfers.
"""

import numpy as np

from ..constants import GM_SUN, AU_TO_M, HOHMANN_DV_TOLERANCE, CIRCULAR_ECC_TOLERANCE
from ..integrator import integrate_trajectory
from .base import compute_r2_actual
from .base import OrbitalBody, transfer_time
from .hohmann import hohmann_delta_v, hohmann_trajectory
from .fast import find_factor_for_dv, calc_dv_for_factor, search_transfer, compute_fast_trajectory, search_transfer_ecliptic


__all__ = [
    "compute_transfer_trajectory",
    "hohmann_delta_v",
    "transfer_time",
    "find_factor_for_dv",
    "OrbitalBody",
    "hohmann_trajectory",
    "calc_dv_for_factor",
    "search_transfer",
    "compute_fast_trajectory",
    "search_transfer_ecliptic",
]


def compute_transfer_trajectory(
    r1,
    r2,
    target_dv,
    points=500,
    target_ecc=0.0,
    target_rot=0.0,
    dep_ecc=0.0,
    dep_rot=0.0,
):
    """Compute a transfer trajectory between two orbits.

    For delta-V budgets close to Hohmann, returns a simple Hohmann arc.
    For larger budgets, searches for faster Lambert-based transfers and
    numerically integrates the trajectory.

    Parameters
    ----------
    r1 : float
        Semi-major axis of departure orbit in meters.
    r2 : float
        Semi-major axis of arrival orbit in meters.
    target_dv : float
        Available delta-V budget in m/s.
    points : int, optional
        Number of trajectory points (default: 500).
    target_ecc : float, optional
        Eccentricity of the target orbit (default: 0.0).
    target_rot : float, optional
        Rotation angle of the target orbit in radians (default: 0.0).
    dep_ecc : float, optional
        Eccentricity of the departure orbit (default: 0.0).
    dep_rot : float, optional
        Rotation angle of the departure orbit in radians (default: 0.0).

    Returns
    -------
    tuple
        (x, y, dep_burn, arr_burn, dnu, tof) where x, y are trajectory
        coordinates in AU, dep_burn and arr_burn are burn points in AU,
        dnu is the arrival true anomaly, and tof is the transfer time in seconds.
    """
    dv_dep_h, dv_arr_h, dv_total_h = hohmann_delta_v(r1, r2)
    hohmann_tof = np.pi * np.sqrt(((r1 + r2) / 2) ** 3 / GM_SUN)

    if abs(target_dv - dv_total_h) < HOHMANN_DV_TOLERANCE and target_ecc < CIRCULAR_ECC_TOLERANCE and dep_ecc < CIRCULAR_ECC_TOLERANCE:
        x, y = hohmann_trajectory(r1, r2, points)
        return (
            x,
            y,
            np.array([r1 / AU_TO_M, 0.0]),
            np.array([-r2 / AU_TO_M, 0.0]),
            np.pi,
            hohmann_tof,
        )

    best, _ = search_transfer(
        r1, r2, target_dv, points, target_ecc, target_rot, dep_ecc, dep_rot
    )

    if best is None:
        from .fast import _hohmann_fallback
        return _hohmann_fallback(r1, r2, points, target_ecc, target_rot)

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