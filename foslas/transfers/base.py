"""Orbital mechanics base utilities for interplanetary transfers.

Provides shared classes and functions used by both Hohmann and fast
transfer calculations.
"""

import numpy as np

from ..constants import GM_SUN, KM_TO_M, AU_TO_M, SEC_TO_DAY


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
    e = abs(1 - r1 / a)
    if e < 1e-10:
        return 0.0
    cos_E1 = np.clip((1 - r1 / a) / e, -1.0, 1.0)
    E1 = np.arccos(cos_E1)
    cos_E2 = np.clip((1 - r2 / a) / e, -1.0, 1.0)
    E2 = np.arccos(cos_E2)
    M1 = E1 - e * np.sin(E1)
    M2 = E2 - e * np.sin(E2)
    dM = abs(M2 - M1)
    return dM * np.sqrt(a**3 / GM_SUN) / SEC_TO_DAY


def compute_r2_actual(r2, target_ecc, orbit_angle):
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
    return r2 * (1 - target_ecc**2) / (1 + target_ecc * np.cos(orbit_angle))


def planet_velocity(r_sma, ecc, nu, angle):
    """Compute planet velocity vector in the transfer frame.

    Parameters
    ----------
    r_sma : float
        Semi-major axis of the orbit in meters.
    ecc : float
        Orbital eccentricity.
    nu : float
        True anomaly in radians.
    angle : float
        Angle of the planet in the transfer frame in radians.

    Returns
    -------
    numpy.ndarray
        Velocity vector [vx, vy, 0] in m/s in the transfer frame.
    """
    h = np.sqrt(GM_SUN * r_sma * (1 - ecc**2))
    v_r = (GM_SUN / h) * ecc * np.sin(nu)
    v_theta = (GM_SUN / h) * (1 + ecc * np.cos(nu))
    vx = v_r * np.cos(angle) - v_theta * np.sin(angle)
    vy = v_r * np.sin(angle) + v_theta * np.cos(angle)
    return np.array([vx, vy, 0.0])
