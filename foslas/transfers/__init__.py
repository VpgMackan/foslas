"""Transfer trajectory computations.

Provides the main entry point for computing transfer trajectories,
coordinating between Hohmann and fast Lambert-based transfers.
"""

from ..constants import KM_TO_M
from .base import OrbitalBody, OrbitGeometry
from .fast import find_factor_for_dv, calc_dv_for_factor, search_transfer, search_transfer_ecliptic
from .hohmann import hohmann_delta_v
from .porkchop import TransferTrajectory
from .strategy import TransferStrategy, HohmannTransfer, FastLambertTransfer


def transfer_time(r1, r2, factor):
    """Compute transfer time for a scaled transfer orbit.

    Parameters
    ----------
    r1 : float
        Radius of departure orbit in meters.
    r2 : float
        Radius of arrival orbit in meters.
    factor : float
        Energy factor scaling the semi-major axis (1.0 = Hohmann).

    Returns
    -------
    float
        Transfer time in days.
    """
    r1_km = r1 / KM_TO_M
    r2_km = r2 / KM_TO_M
    dep = OrbitalBody(r1_km, r1_km)
    arr = OrbitalBody(r2_km, r2_km)
    return dep.transfer_time_to(arr, factor)


__all__ = [
    "compute_transfer_trajectory",
    "hohmann_delta_v",
    "transfer_time",
    "find_factor_for_dv",
    "OrbitalBody",
    "OrbitGeometry",
    "hohmann_trajectory",
    "calc_dv_for_factor",
    "search_transfer",
    "search_transfer_ecliptic",
    "TransferTrajectory",
    "TransferStrategy",
    "HohmannTransfer",
    "FastLambertTransfer",
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
    TransferTrajectory
        The computed transfer trajectory with x, y, dep_burn, arr_burn, dnu, tof fields.
    float
        Hohmann transfer time in seconds.
    """
    dv_dep_h, dv_arr_h, dv_total_h = hohmann_delta_v(r1, r2)

    dep = OrbitalBody(r1 / KM_TO_M, r1 / KM_TO_M, eccentricity=dep_ecc)
    arr = OrbitalBody(r2 / KM_TO_M, r2 / KM_TO_M, eccentricity=target_ecc)

    if abs(target_dv - dv_total_h) < 1.0:
        strategy = HohmannTransfer()
    else:
        strategy = FastLambertTransfer()

    kwargs = {
        "points": points,
        "target_ecc": target_ecc,
        "target_rot": target_rot,
        "dep_ecc": dep_ecc,
        "dep_rot": dep_rot,
    }
    result, ht = strategy.compute(dep, arr, target_dv, **kwargs)
    return result, ht