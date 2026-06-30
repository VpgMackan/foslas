"""foslas - Fast Orbital Solar System Lambert Solver.

A Python library for computing interplanetary transfer trajectories
between celestial bodies in the solar system.
"""

from .bodies import load_planet_bodies, load_asteroid_body
from .constants import G, M_SUN, GM_SUN, KM_TO_M, AU_TO_M, AU_TO_KM, SEC_TO_DAY
from .lambert import lambert_solve
from .integrator import two_body_ode, integrate_trajectory
from .porkchop import (
    compute_porkchop,
    plot_porkchop,
    plot_porkchop_budget,
    summarize,
    compute_lambert_trajectory,
    plot_lambert_trajectory,
    get_feasible_windows,
    PorkchopResult,
    LambertTrajectory,
)
from .transfers.base import OrbitalBody, transfer_time
from .transfers.hohmann import hohmann_delta_v, hohmann_trajectory
from .transfers.fast import find_factor_for_dv, compute_fast_trajectory
from .transfers import compute_transfer_trajectory
from .transfers.visualization import (
    visualize,
    plot_orbit,
    plot_transfer,
)

__all__ = [
    "load_planet_bodies", "load_asteroid_body",
    "G", "M_SUN", "GM_SUN", "KM_TO_M", "AU_TO_M", "AU_TO_KM", "SEC_TO_DAY",
    "lambert_solve",
    "two_body_ode", "integrate_trajectory",
    "compute_porkchop", "plot_porkchop", "plot_porkchop_budget", "summarize",
    "compute_lambert_trajectory", "plot_lambert_trajectory", "get_feasible_windows",
    "PorkchopResult", "LambertTrajectory",
    "OrbitalBody", "hohmann_delta_v", "transfer_time", "find_factor_for_dv",
    "compute_transfer_trajectory", "compute_fast_trajectory",
    "plot_orbit", "plot_transfer", "visualize",
]
