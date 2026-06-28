"""Orbital mechanics functions for interplanetary transfers.

Provides Hohmann transfer calculations, energy factor search for faster
transfers, and full trajectory computation using Lambert's solver.
"""

import numpy as np
from scipy.optimize import brentq

from .constants import GM_SUN, KM_TO_M, AU_TO_M, SEC_TO_DAY
from .lambert import lambert_solve
from .integrator import integrate_trajectory


class OrbitalBody:
    """Represents a celestial body with an elliptical orbit.

    Parameters
    ----------
    aphelion_km : float
        Aphelion distance in kilometers.
    perihelion_km : float
        Perihelion distance in kilometers.

    Attributes
    ----------
    aphelion : float
        Aphelion distance in meters.
    perihelion : float
        Perihelion distance in meters.
    sma : float
        Semi-major axis in meters.
    """

    def __init__(self, aphelion_km, perihelion_km):
        self.aphelion = aphelion_km * KM_TO_M
        self.perihelion = perihelion_km * KM_TO_M
        self.sma = (self.aphelion + self.perihelion) / 2

    @property
    def eccentricity(self):
        """Compute orbital eccentricity from aphelion and perihelion.

        Returns
        -------
        float
            Eccentricity (0 for circular, 0-1 for elliptical).
        """
        return (self.aphelion - self.perihelion) / (self.aphelion + self.perihelion)


def hohmann_delta_v(r1, r2):
    """Compute delta-V for a Hohmann transfer between two circular orbits.

    Parameters
    ----------
    r1 : float
        Radius of departure orbit in meters.
    r2 : float
        Radius of arrival orbit in meters.

    Returns
    -------
    tuple of float
        (departure_burn, arrival_burn, total_delta_v) in m/s.
    """
    v1_circ = np.sqrt(GM_SUN / r1)
    v2_circ = np.sqrt(GM_SUN / r2)
    a_t = (r1 + r2) / 2
    v_dep = np.sqrt(GM_SUN * (2 / r1 - 1 / a_t)) - v1_circ
    v_arr = v2_circ - np.sqrt(GM_SUN * (2 / r2 - 1 / a_t))
    return v_dep, v_arr, v_dep + v_arr


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
    a = ((r1 + r2) / 2) * factor
    e = 1 - r1 / a
    if e < 1e-10:
        return 0.0
    cos_E = np.clip((1 - r2 / a) / e, -1.0, 1.0)
    E = np.arccos(cos_E)
    M = E - e * np.sin(E)
    return M * np.sqrt(a ** 3 / GM_SUN) / SEC_TO_DAY


def _calc_dv_for_factor(r1, r2, factor):
    """Compute total delta-V for a transfer orbit scaled by a factor.

    Parameters
    ----------
    r1 : float
        Radius of departure orbit in meters.
    r2 : float
        Radius of arrival orbit in meters.
    factor : float
        Energy factor scaling the semi-major axis.

    Returns
    -------
    float
        Total delta-V in m/s.
    """
    a = ((r1 + r2) / 2) * factor
    v1_circ = np.sqrt(GM_SUN / r1)
    v2_circ = np.sqrt(GM_SUN / r2)
    vt1 = np.sqrt(GM_SUN * (2 / r1 - 1 / a))
    v_theta = r1 * vt1 / r2
    v_arr_mag = np.sqrt(GM_SUN * (2 / r2 - 1 / a))
    v_radial_sq = v_arr_mag ** 2 - v_theta ** 2
    v_radial = np.sqrt(v_radial_sq) if v_radial_sq > 0 else 0.0
    dv_dep = abs(vt1 - v1_circ)
    dv_arr = np.hypot(v_radial, v2_circ - v_theta)
    return dv_dep + dv_arr


def find_factor_for_dv(r1, r2, target_dv, max_factor=50.0):
    """Find the energy factor that uses a given delta-V budget.

    Parameters
    ----------
    r1 : float
        Radius of departure orbit in meters.
    r2 : float
        Radius of arrival orbit in meters.
    target_dv : float
        Available delta-V budget in m/s.
    max_factor : float, optional
        Maximum energy factor to search (default: 50.0).

    Returns
    -------
    tuple of float
        (factor, actual_delta_v) where factor scales the Hohmann
        semi-major axis and actual_delta_v is the delta-V consumed.
    """
    dv_hoh, _, _ = hohmann_delta_v(r1, r2)
    if target_dv < dv_hoh:
        return 1.0, dv_hoh

    def residual(factor):
        return _calc_dv_for_factor(r1, r2, factor) - target_dv

    try:
        factor = brentq(residual, 1.0, max_factor, xtol=1e-10, rtol=1e-12)
    except ValueError:
        factor = max_factor

    return factor, _calc_dv_for_factor(r1, r2, factor)


def _hohmann_trajectory(r1, r2, points):
    """Compute a Hohmann transfer arc between two circular orbits.

    Parameters
    ----------
    r1 : float
        Radius of departure orbit in meters.
    r2 : float
        Radius of arrival orbit in meters.
    points : int
        Number of points on the trajectory.

    Returns
    -------
    tuple of numpy.ndarray
        (x, y) coordinates in AU.
    """
    a = (r1 + r2) / 2
    e = (r2 - r1) / (r2 + r1)
    thetas = np.linspace(0, np.pi, points)
    radii = (a * (1 - e ** 2)) / (1 + e * np.cos(thetas))
    x = radii * np.cos(thetas) / AU_TO_M
    y = radii * np.sin(thetas) / AU_TO_M
    return x, y


def _compute_r2_actual(r2, target_ecc, orbit_angle):
    """Compute actual radius at a given true anomaly for an eccentric orbit.

    Parameters
    ----------
    r2 : float
        Nominal radius in meters.
    target_ecc : float
        Eccentricity of the target orbit.
    orbit_angle : float
        True anomaly in radians.

    Returns
    -------
    float
        Actual radius at the given angle.
    """
    return r2 * (1 - target_ecc ** 2) / (1 + target_ecc * np.cos(orbit_angle))


def _search_transfer(r1, r2, target_dv, points, target_ecc, target_rot, v1_circ):
    """Search for the fastest transfer within a delta-V budget.

    Brute-force searches over time-of-flight fractions and arrival angles
    to find Lambert solutions that fit within the delta-V budget.

    Parameters
    ----------
    r1 : float
        Radius of departure orbit in meters.
    r2 : float
        Radius of arrival orbit in meters.
    target_dv : float
        Available delta-V budget in m/s.
    points : int
        Number of trajectory points.
    target_ecc : float
        Eccentricity of the target orbit.
    target_rot : float
        Rotation angle of the target orbit in radians.
    v1_circ : float
        Circular velocity at departure orbit in m/s.

    Returns
    -------
    tuple
        (best, hohmann_tof) where best is (tof, dnu, v1, r1_vec, r2_actual)
        or None if no solution found.
    """
    hohmann_tof = np.pi * np.sqrt(((r1 + r2) / 2) ** 3 / GM_SUN)

    tof_fractions = np.linspace(0.05, 0.95, 60)
    dnu_values = np.linspace(0.3, np.pi - 0.05, 40)

    r1_vec = np.array([r1, 0.0, 0.0])

    best = None
    for tof_frac in tof_fractions:
        tof = hohmann_tof * tof_frac
        for dnu in dnu_values:
            orbit_angle = dnu - target_rot
            r2_actual = _compute_r2_actual(r2, target_ecc, orbit_angle)
            v2_circ = np.sqrt(GM_SUN / r2_actual)
            r2_vec = np.array([r2_actual * np.cos(dnu), r2_actual * np.sin(dnu), 0.0])
            try:
                v1, v2 = lambert_solve(r1_vec, r2_vec, tof)
            except ValueError:
                continue
            dv_dep = np.sqrt(v1[0] ** 2 + (v1[1] - v1_circ) ** 2 + v1[2] ** 2)
            dv_arr = np.sqrt(v2[0] ** 2 + (v2[1] - v2_circ) ** 2 + v2[2] ** 2)
            total_dv = dv_dep + dv_arr
            if total_dv <= target_dv:
                if best is None or tof < best[0]:
                    best = (tof, dnu, v1, r1_vec, r2_actual)
        if best is not None:
            break

    return best, hohmann_tof


def compute_transfer_trajectory(r1, r2, target_dv, points=500, target_ecc=0.0, target_rot=0.0):
    """Compute a transfer trajectory between two orbits.

    For delta-V budgets close to Hohmann, returns a simple Hohmann arc.
    For larger budgets, searches for faster Lambert-based transfers and
    numerically integrates the trajectory.

    Parameters
    ----------
    r1 : float
        Radius of departure orbit in meters.
    r2 : float
        Radius of arrival orbit in meters.
    target_dv : float
        Available delta-V budget in m/s.
    points : int, optional
        Number of trajectory points (default: 500).
    target_ecc : float, optional
        Eccentricity of the target orbit (default: 0.0).
    target_rot : float, optional
        Rotation angle of the target orbit in radians (default: 0.0).

    Returns
    -------
    tuple
        (x, y, dep_burn, arr_burn, dnu) where x, y are trajectory
        coordinates in AU, dep_burn and arr_burn are burn points in AU,
        and dnu is the arrival true anomaly.
    """
    dv_dep_h, dv_arr_h, dv_total_h = hohmann_delta_v(r1, r2)
    hohmann_tof = np.pi * np.sqrt(((r1 + r2) / 2) ** 3 / GM_SUN)

    if abs(target_dv - dv_total_h) < 1.0:
        x, y = _hohmann_trajectory(r1, r2, points)
        return (
            x,
            y,
            np.array([r1 / AU_TO_M, 0.0]),
            np.array([-r2 / AU_TO_M, 0.0]),
            np.pi,
        )

    v1_circ = np.sqrt(GM_SUN / r1)

    best, _ = _search_transfer(r1, r2, target_dv, points, target_ecc, target_rot, v1_circ)

    if best is None:
        tof = hohmann_tof * 0.5
        dnu = np.pi * 0.75
        orbit_angle = dnu - target_rot
        r2_actual = _compute_r2_actual(r2, target_ecc, orbit_angle)
        r1_vec = np.array([r1, 0.0, 0.0])
        r2_vec = np.array([r2_actual * np.cos(dnu), r2_actual * np.sin(dnu), 0.0])
        try:
            v1, v2 = lambert_solve(r1_vec, r2_vec, tof)
        except ValueError:
            x, y = _hohmann_trajectory(r1, r2, points)
            return (
                x,
                y,
                np.array([r1 / AU_TO_M, 0.0]),
                np.array([-r2 / AU_TO_M, 0.0]),
                np.pi,
            )
        best = (tof, dnu, v1, r1_vec, r2_actual)

    tof, dnu, v1, r1_vec, r2_actual = best
    positions, _ = integrate_trajectory(r1_vec, v1, tof, points)

    x = positions[:, 0] / AU_TO_M
    y = positions[:, 1] / AU_TO_M
    dep_burn = np.array([r1 / AU_TO_M, 0.0])
    arr_burn = np.array([r2_actual * np.cos(dnu) / AU_TO_M, r2_actual * np.sin(dnu) / AU_TO_M])

    return x, y, dep_burn, arr_burn, dnu
