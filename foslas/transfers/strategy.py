"""Transfer strategy pattern for swappable transfer algorithms.

Provides a clean abstraction for different transfer computation methods,
allowing explicit strategy selection and fallback behavior.
"""

from abc import ABC, abstractmethod

import numpy as np

from ..constants import AU_TO_M
from .base import OrbitalBody, OrbitGeometry
from .fast import search_transfer
from ..porkchop import TransferTrajectory


class TransferStrategy(ABC):
    """Abstract base class for transfer computation strategies."""

    @abstractmethod
    def compute(self, dep: OrbitalBody, arr: OrbitalBody, target_dv: float, **kwargs):
        """Compute a transfer trajectory.

        Parameters
        ----------
        dep : OrbitalBody
            Departure orbit body.
        arr : OrbitalBody
            Arrival orbit body.
        target_dv : float
            Available delta-V budget in m/s.
        **kwargs
            Additional strategy-specific parameters.

        Returns
        -------
        TransferTrajectory
            The computed transfer trajectory.
        """
        ...


class HohmannTransfer(TransferStrategy):
    """Hohmann transfer strategy - minimum energy transfer for circular orbits."""

    def compute(self, dep: OrbitalBody, arr: OrbitalBody, target_dv: float, **kwargs):
        from ..constants import AU_TO_M
        from .hohmann import hohmann_trajectory

        points = kwargs.get("points", 500)
        x, y = hohmann_trajectory(dep.sma, arr.sma, points)
        ht = dep.transfer_time_to(arr)
        return TransferTrajectory(
            x=x,
            y=y,
            dep_burn=np.array([dep.sma / AU_TO_M, 0.0]),
            arr_burn=np.array([-arr.sma / AU_TO_M, 0.0]),
            dnu=np.pi,
            tof=ht,
        )


class FastLambertTransfer(TransferStrategy):
    """Fast Lambert-based transfer strategy with explicit Hohmann fallback."""

    def __init__(self, tolerance: float = 1.0):
        self.tolerance = tolerance

    def compute(self, dep: OrbitalBody, arr: OrbitalBody, target_dv: float, **kwargs):
        points = kwargs.get("points", 500)
        target_geom = OrbitGeometry(
            eccentricity=kwargs.get("target_ecc", 0.0),
            rotation=kwargs.get("target_rot", 0.0)
        )
        dep_geom = OrbitGeometry(
            eccentricity=kwargs.get("dep_ecc", 0.0),
            rotation=kwargs.get("dep_rot", 0.0)
        )

        from ..integrator import integrate_trajectory
        from ..constants import KM_TO_M

        r1 = dep.sma
        r2 = arr.sma

        best, _ = search_transfer(r1, r2, target_dv, points, target_geom, dep_geom)

        if best is None:
            return HohmannTransfer().compute(dep, arr, target_dv, **kwargs)

        tof, dnu, v1, r1_vec, r2_actual = best
        positions, _ = integrate_trajectory(r1_vec, v1, tof, points)

        x = positions[:, 0] / AU_TO_M
        y = positions[:, 1] / AU_TO_M
        dep_nu = -dep_geom.rotation
        r1_actual = dep.radius_at(dep_nu)
        dep_burn = np.array([r1_actual / AU_TO_M, 0.0])
        arr_burn = np.array(
            [r2_actual * np.cos(dnu) / AU_TO_M, r2_actual * np.sin(dnu) / AU_TO_M]
        )

        return TransferTrajectory(
            x=x, y=y, dep_burn=dep_burn, arr_burn=arr_burn, dnu=dnu, tof=tof
        )