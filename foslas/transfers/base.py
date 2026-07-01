"""Orbital mechanics base utilities for interplanetary transfers.

Provides shared classes and functions used by both Hohmann and fast
transfer calculations.
"""

import numpy as np
from dataclasses import dataclass

from ..constants import GM_SUN, KM_TO_M, SEC_TO_DAY


def hohmann_tof(r1, r2):
    """Compute Hohmann transfer time in seconds."""
    return np.pi * np.sqrt(((r1 + r2) / 2) ** 3 / GM_SUN)


@dataclass
class OrbitGeometry:
    """Orbital geometry parameters for a celestial body.

    Attributes
    ----------
    eccentricity : float
        Orbital eccentricity (0 for circular).
    rotation : float
        Rotation angle in radians (true anomaly or argument of latitude).
    """
    eccentricity: float = 0.0
    rotation: float = 0.0


class OrbitalBody:
    """Represents a celestial body or orbit with an elliptical orbit.

    Parameters
    ----------
    aphelion_km : float
        Aphelion distance in kilometers.
    perihelion_km : float
        Perihelion distance in kilometers.
    eccentricity : float, optional
        Orbital eccentricity. If not provided, computed from aphelion/perihelion.

    Attributes
    ----------
    aphelion : float
        Aphelion distance in meters.
    perihelion : float
        Perihelion distance in meters.
    sma : float
        Semi-major axis in meters.
    """

    def __init__(self, aphelion_km, perihelion_km, eccentricity=None):
        self.aphelion = aphelion_km * KM_TO_M
        self.perihelion = perihelion_km * KM_TO_M
        self.sma = (self.aphelion + self.perihelion) / 2
        self._ecc = eccentricity

    @property
    def eccentricity(self):
        """Compute orbital eccentricity from aphelion and perihelion.

        Returns
        -------
        float
            Eccentricity (0 for circular, 0-1 for elliptical).
        """
        if self._ecc is not None:
            return self._ecc
        return (self.aphelion - self.perihelion) / (self.aphelion + self.perihelion)

    def velocity_at(self, nu, angle):
        """Compute velocity vector at a given true anomaly.

        Parameters
        ----------
        nu : float
            True anomaly in radians.
        angle : float
            Angle of the planet in the transfer frame in radians.

        Returns
        -------
        numpy.ndarray
            Velocity vector [vx, vy, 0] in m/s in the transfer frame.
        """
        ecc = self.eccentricity
        r_sma = self.sma
        h = np.sqrt(GM_SUN * r_sma * (1 - ecc**2))
        v_r = (GM_SUN / h) * ecc * np.sin(nu)
        v_theta = (GM_SUN / h) * (1 + ecc * np.cos(nu))
        vx = v_r * np.cos(angle) - v_theta * np.sin(angle)
        vy = v_r * np.sin(angle) + v_theta * np.cos(angle)
        return np.array([vx, vy, 0.0])

    def radius_at(self, true_anomaly):
        """Compute actual radius at a given true anomaly.

        Parameters
        ----------
        true_anomaly : float
            True anomaly in radians.

        Returns
        -------
        float
            Actual radius at the given angle in meters.
        """
        ecc = self.eccentricity
        r2 = self.sma
        return r2 * (1 - ecc**2) / (1 + ecc * np.cos(true_anomaly))

    def transfer_time_to(self, other, factor=1.0):
        """Compute transfer time to another orbit with a given energy factor.

        Parameters
        ----------
        other : OrbitalBody
            The target orbit.
        factor : float, optional
            Energy factor scaling the semi-major axis (1.0 = Hohmann).

        Returns
        -------
        float
            Transfer time in days.
        """
        r1 = self.sma
        r2 = other.sma
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
