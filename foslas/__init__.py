"""foslas - Fast Orbital Solar System Lambert Solver.

A Python library for computing interplanetary transfer trajectories
between celestial bodies in the solar system.
"""

from .bodies import load_planet_bodies, load_asteroid_body, Body
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
from .transfers.base import OrbitalBody, OrbitGeometry
from .transfers import transfer_time
from .transfers.hohmann import hohmann_delta_v
from .transfers.fast import find_factor_for_dv
from .transfers import compute_transfer_trajectory, TransferTrajectory
from .transfers.strategy import TransferStrategy, HohmannTransfer, FastLambertTransfer
from .transfers.visualization import (
    visualize,
    plot_orbit,
    plot_transfer,
)
from .utils import find_body, find_asteroid, resolve_bodies, resolve_body_data, orbit_params
from .ephemeris import EphemerisProvider, PykepEphemeris, KeplerianEphemeris, PositionVelocity

__all__ = [
    "load_planet_bodies", "load_asteroid_body", "Body",
    "G", "M_SUN", "GM_SUN", "KM_TO_M", "AU_TO_M", "AU_TO_KM", "SEC_TO_DAY",
    "lambert_solve",
    "two_body_ode", "integrate_trajectory",
    "compute_porkchop", "plot_porkchop", "plot_porkchop_budget", "summarize",
    "compute_lambert_trajectory", "plot_lambert_trajectory", "get_feasible_windows",
    "PorkchopResult", "LambertTrajectory",
    "OrbitalBody", "OrbitGeometry", "hohmann_delta_v", "transfer_time", "find_factor_for_dv",
    "compute_transfer_trajectory", "TransferTrajectory",
    "plot_orbit", "plot_transfer", "visualize",
    "find_body", "find_asteroid", "resolve_bodies", "resolve_body_data", "orbit_params",
    "TransferStrategy", "HohmannTransfer", "FastLambertTransfer",
    "EphemerisProvider", "PykepEphemeris", "KeplerianEphemeris", "PositionVelocity",
]
