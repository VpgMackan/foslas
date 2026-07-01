"""Ephemeris providers for celestial body positions.

Provides a common interface for different ephemeris implementations,
making the duplication between pykep and hand-rolled Keplerian propagation
explicit and testable.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Protocol

import numpy as np


@dataclass
class PositionVelocity:
    position: np.ndarray
    velocity: np.ndarray


class EphemerisProvider(Protocol):
    """Protocol for celestial body ephemeris providers."""

    def position_velocity(self, body_name: str, epoch: datetime) -> PositionVelocity:
        """Get position and velocity for a body at a given epoch."""
        ...


class PykepEphemeris(EphemerisProvider):
    """Ephemeris provider using pykep library."""

    def __init__(self):
        import pykep as pk
        self._pk = pk

    def position_velocity(self, body_name: str, epoch: datetime) -> PositionVelocity:
        from ...constants import J2000_JD, JD_EPOCH_OFFSET
        
        jd = epoch.timestamp() / 86400.0 + JD_EPOCH_OFFSET
        pk_epoch = self._pk.epoch(jd - J2000_JD)
        planet = self._pk.planet(self._pk.udpla.jpl_lp(body_name))
        r, v = planet.eph(pk_epoch)
        return PositionVelocity(
            position=np.array(r),
            velocity=np.array(v)
        )


class KeplerianEphemeris(EphemerisProvider):
    """Ephemeris provider using hand-rolled Keplerian propagation."""

    def __init__(self):
        from ...bodies import ASTEROID_CATALOG, keplerian_to_state
        from ...constants import JD_EPOCH_OFFSET
        self._catalog = ASTEROID_CATALOG
        self._keplerian_to_state = keplerian_to_state
        self._jd_offset = JD_EPOCH_OFFSET

    def position_velocity(self, body_name: str, epoch: datetime) -> PositionVelocity:
        body_name_lower = body_name.lower().replace(" ", "_")
        
        if body_name_lower in self._catalog:
            ast = self._catalog[body_name_lower]
            t_jd = epoch.timestamp() / 86400.0 + self._jd_offset
            r_au, lon = self._keplerian_to_state(
                ast["a_au"], ast["ecc"], ast["inc_deg"],
                ast["omega_deg"], ast["w_deg"], ast["M0_deg"],
                ast["epoch_jd"], t_jd
            )
            return PositionVelocity(
                position=np.array([r_au]),
                velocity=np.array([0.0])
            )
        
        raise ValueError(f"Body '{body_name}' not found in catalog")


def get_default_ephemeris() -> EphemerisProvider:
    """Return the default ephemeris provider (pykep-based)."""
    return PykepEphemeris()