"""foslas - Fast Orbital Solar System Lambert Solver.

A Python library for computing interplanetary transfer trajectories
between celestial bodies in the solar system.
"""

from .constants import G, M_SUN, GM_SUN, KM_TO_M, AU_TO_M, AU_TO_KM, SEC_TO_DAY
from .lambert import lambert_solve, stumpff_S, stumpff_C
from .integrator import two_body_ode, integrate_trajectory
from .transfers.base import OrbitalBody, transfer_time
from .transfers.hohmann import hohmann_delta_v, hohmann_trajectory
from .transfers.fast import find_factor_for_dv, compute_fast_trajectory
from .transfers import compute_transfer_trajectory
from .transfers.visualization import (
    visualize,
    animate_transfer,
    plot_orbit,
    plot_transfer,
)

__all__ = [
    "G", "M_SUN", "GM_SUN", "KM_TO_M", "AU_TO_M", "AU_TO_KM", "SEC_TO_DAY",
    "lambert_solve", "stumpff_S", "stumpff_C",
    "two_body_ode", "integrate_trajectory",
    "OrbitalBody", "hohmann_delta_v", "transfer_time", "find_factor_for_dv",
    "compute_transfer_trajectory", "compute_fast_trajectory",
    "plot_orbit", "plot_transfer", "visualize", "animate_transfer",
]
