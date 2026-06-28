"""foslas - Fast Orbital Solar System Lambert Solver.

A Python library for computing interplanetary transfer trajectories
between celestial bodies in the solar system.
"""

from .constants import G, M_SUN, GM_SUN, KM_TO_M, AU_TO_M, AU_TO_KM, SEC_TO_DAY
from .lambert import lambert_solve, stumpff_S, stumpff_C
from .integrator import two_body_ode, integrate_trajectory
from .orbital import (
    OrbitalBody,
    hohmann_delta_v,
    transfer_time,
    find_factor_for_dv,
    compute_transfer_trajectory,
)
from .viz import plot_orbit, plot_transfer, visualize

__all__ = [
    "G", "M_SUN", "GM_SUN", "KM_TO_M", "AU_TO_M", "AU_TO_KM", "SEC_TO_DAY",
    "lambert_solve", "stumpff_S", "stumpff_C",
    "two_body_ode", "integrate_trajectory",
    "OrbitalBody", "hohmann_delta_v", "transfer_time", "find_factor_for_dv",
    "compute_transfer_trajectory",
    "plot_orbit", "plot_transfer", "visualize",
]
