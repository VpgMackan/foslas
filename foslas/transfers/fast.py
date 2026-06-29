"""Fast Lambert-based transfer calculations.

Provides functions for computing faster transfers that use more delta-V
than the minimum-energy Hohmann transfer.
"""

import logging

import numpy as np
from scipy.optimize import brentq

from ..constants import GM_SUN, AU_TO_M
from ..lambert import lambert_solve
from ..integrator import integrate_trajectory
from .base import compute_r2_actual, planet_velocity

from .hohmann import hohmann_trajectory, hohmann_delta_v

logger = logging.getLogger(__name__)


def search_transfer_ecliptic(
    dep_body_name,
    arr_body_name,
    target_dv,
    points,
):
    """Search for fastest transfer using actual ephemeris positions.

    Performs the entire search in the ecliptic frame using actual
    planetary positions from ephemeris, avoiding the frame-mismatch bug.
    Finds the minimum-TOF solution within the delta-V budget.

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
        (x, y, dep_burn, arr_burn, dnu, tof) where x, y are trajectory
        coordinates in AU, dep_burn and arr_burn are burn points in AU,
        dnu is the arrival true anomaly, and tof is transfer time in seconds.
        Returns None if no solution found.
    """
    from .visualization.core import get_body_ecliptic

    dep_r, dep_lon = get_body_ecliptic(dep_body_name)
    r1_vec = np.array([dep_r * np.cos(dep_lon), dep_r * np.sin(dep_lon), 0.0]) * AU_TO_M
    v_dep, _ = _get_planet_state(dep_body_name, 0)

    arr_r, _ = get_body_ecliptic(arr_body_name)
    hohmann_tof = np.pi * np.sqrt(((dep_r + arr_r) / 2 * AU_TO_M) ** 3 / GM_SUN)

    max_search_tof = max(hohmann_tof * 3.0, 30 * 86400)
    tof_values = np.linspace(10 * 86400, max_search_tof, 200)

    best = None
    best_tof = None

    for tof in tof_values:
        future_r, future_lon = get_body_ecliptic(arr_body_name, time_offset_days=tof / 86400.0)
        r2_vec = np.array([future_r * np.cos(future_lon), future_r * np.sin(future_lon), 0.0]) * AU_TO_M
        v_arr, _ = _get_planet_state(arr_body_name, tof / 86400.0)

        try:
            v1, v2 = lambert_solve(r1_vec, r2_vec, tof)
        except ValueError:
            continue

        dv_dep = np.linalg.norm(v1 - v_dep)
        dv_arr = np.linalg.norm(v2 - v_arr)
        total_dv = dv_dep + dv_arr

        if total_dv <= target_dv:
            r1_mag = np.linalg.norm(r1_vec)
            v1_mag = np.linalg.norm(v1)
            specific_energy = v1_mag**2 / 2.0 - GM_SUN / r1_mag
            if specific_energy >= 0:
                continue
            a_transfer = -GM_SUN / (2.0 * specific_energy)
            if a_transfer <= 0:
                continue
            best = (tof, v1, r1_vec, r2_vec)
            best_tof = tof
            break

    if best is None:
        return None, hohmann_tof

    tof, v1, r1_vec, r2_vec = best
    positions, _ = integrate_trajectory(r1_vec, v1, tof, points)

    x = positions[:, 0] / AU_TO_M
    y = positions[:, 1] / AU_TO_M
    dep_burn = np.array([r1_vec[0] / AU_TO_M, r1_vec[1] / AU_TO_M])
    arr_burn = np.array([r2_vec[0] / AU_TO_M, r2_vec[1] / AU_TO_M])

    dnu = np.pi
    return (x, y, dep_burn, arr_burn, dnu, tof), hohmann_tof


def _get_planet_state(body_name, time_offset_days):
    """Get planetary position and velocity at a given time offset.

    Parameters
    ----------
    body_name : str
        Name of the body.
    time_offset_days : float
        Days from now.

    Returns
    -------
    tuple
        (velocity_vector, position_vector) in m/s and m.
    """
    from ..bodies import ASTEROID_CATALOG, keplerian_to_state

    body_name_lower = body_name.lower().replace(" ", "_")
    if body_name_lower in ASTEROID_CATALOG:
        dt_days = time_offset_days
        ast = ASTEROID_CATALOG[body_name_lower]
        t_jd = ast["epoch_jd"] + dt_days
        r_au, lon = keplerian_to_state(
            ast["a_au"], ast["ecc"], ast["inc_deg"],
            ast["omega_deg"], ast["w_deg"], ast["M0_deg"],
            ast["epoch_jd"], t_jd
        )
        a = ast["a_au"] * AU_TO_M
        e = ast["ecc"]
        inc = np.radians(ast["inc_deg"])
        omega = np.radians(ast["omega_deg"])
        w = np.radians(ast["w_deg"])
        mean_motion = np.sqrt(GM_SUN / a**3) * 86400.0
        M = np.radians(ast["M0_deg"]) + mean_motion * dt_days
        E = M
        for _ in range(50):
            f = E - e * np.sin(E) - M
            fp = 1 - e * np.cos(E)
            delta = -f / fp
            E += delta
            if abs(delta) < 1e-12:
                break
        r = a * (1 - e * np.cos(E))
        nu = 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2.0), np.sqrt(1 - e) * np.cos(E / 2.0))
        h = np.sqrt(GM_SUN * a * (1 - e**2))
        v_r = (GM_SUN / h) * e * np.sin(nu)
        v_theta = (GM_SUN / h) * (1 + e * np.cos(nu))
        cos_lon = np.cos(lon)
        sin_lon = np.sin(lon)
        vx = v_r * cos_lon - v_theta * sin_lon
        vy = v_r * sin_lon + v_theta * cos_lon
        return np.array([vx, vy, 0.0]), np.array([r_au * cos_lon, r_au * sin_lon, 0.0]) * AU_TO_M

    try:
        import astropy.units as u
        from astropy.coordinates import get_body_barycentric_posvel
        from astropy.time import Time, TimeDelta

        now = Time.now() + TimeDelta(time_offset_days, format='jd')
        body_pos, body_vel = get_body_barycentric_posvel(body_name, now)
        sun_pos, sun_vel = get_body_barycentric_posvel("sun", now)

        r_vec = (body_pos.xyz.to(u.m).value - sun_pos.xyz.to(u.m).value)
        v_vec = (body_vel.xyz.to(u.m / u.s).value - sun_vel.xyz.to(u.m / u.s).value)

        return np.array([v_vec[0], v_vec[1], v_vec[2]]), np.array([r_vec[0], r_vec[1], r_vec[2]])
    except Exception:
        return np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0])


def _hohmann_fallback(r1, r2, points, target_ecc=0.0, target_rot=0.0):
    """Return Hohmann transfer as fallback trajectory.

    Returns
    -------
    tuple
        (x, y, dep_burn, arr_burn, dnu, tof) for Hohmann transfer.
    """
    x, y = hohmann_trajectory(r1, r2, points)
    hohmann_tof = np.pi * np.sqrt(((r1 + r2) / 2) ** 3 / GM_SUN)
    dnu = np.pi
    orbit_angle = dnu - target_rot
    r2_actual = r2 * (1 - target_ecc**2) / (1 + target_ecc * np.cos(orbit_angle))
    return (
        x,
        y,
        np.array([r1 / AU_TO_M, 0.0]),
        np.array(
            [r2_actual * np.cos(dnu) / AU_TO_M, r2_actual * np.sin(dnu) / AU_TO_M]
        ),
        dnu,
        hohmann_tof,
    )


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

    tof_fractions = np.linspace(0.01, 1.0, 200)
    dnu_values = np.linspace(0.01, np.pi, 100)

    dep_nu = -dep_rot
    r1_actual = compute_r2_actual(r1, dep_ecc, dep_nu)
    r1_vec = np.array([r1_actual, 0.0, 0.0])
    v_planet_dep = planet_velocity(r1, dep_ecc, dep_nu, 0.0)

    best = None
    best_dv = float('inf')
    min_tof = None
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
                r1_mag = np.linalg.norm(r1_vec)
                v1_mag = np.linalg.norm(v1)
                specific_energy = v1_mag**2 / 2.0 - GM_SUN / r1_mag
                if specific_energy >= 0:
                    continue
                a_transfer = -GM_SUN / (2.0 * specific_energy)
                if a_transfer <= 0:
                    continue
                if total_dv < best_dv:
                    best = (tof, dnu, v1, r1_vec, r2_actual)
                    best_dv = total_dv
                    min_tof = tof

    if best is not None and min_tof is not None:
        dnu_values_fine = np.linspace(
            max(0.01, best[1] - 0.1), min(np.pi, best[1] + 0.1), 50
        )
        for dnu in dnu_values_fine:
            orbit_angle = dnu - target_rot
            r2_actual = compute_r2_actual(r2, target_ecc, orbit_angle)
            v_planet_arr = planet_velocity(r2, target_ecc, orbit_angle, dnu)
            r2_vec = np.array([r2_actual * np.cos(dnu), r2_actual * np.sin(dnu), 0.0])
            try:
                v1, v2 = lambert_solve(r1_vec, r2_vec, min_tof)
            except ValueError:
                continue
            dv_dep = np.linalg.norm(v1 - v_planet_dep)
            dv_arr = np.linalg.norm(v2 - v_planet_arr)
            total_dv = dv_dep + dv_arr
            if total_dv <= target_dv and total_dv < best_dv:
                r1_mag = np.linalg.norm(r1_vec)
                v1_mag = np.linalg.norm(v1)
                specific_energy = v1_mag**2 / 2.0 - GM_SUN / r1_mag
                if specific_energy >= 0:
                    continue
                a_transfer = -GM_SUN / (2.0 * specific_energy)
                if a_transfer <= 0:
                    continue
                best = (min_tof, dnu, v1, r1_vec, r2_actual)
                best_dv = total_dv

    if best is None:
        logger.debug(
            "search_transfer exhausted: r1=%.3e, r2=%.3e, target_dv=%.3e — "
            "no Lambert solution with valid semi-major axis found within budget",
            r1, r2, target_dv,
        )

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
