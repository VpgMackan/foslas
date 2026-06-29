"""Numerical trajectory integrator for the two-body problem.

Solves the equations of motion for a spacecraft under the Sun's gravity
using scipy's ODE integrator with the DOP853 method.
"""

import numpy as np
from scipy.integrate import solve_ivp

from .constants import GM_SUN


def two_body_ode(t, state):
    """Equations of motion for the two-body problem under solar gravity.

    Parameters
    ----------
    t : float
        Time (not used explicitly, but required by solve_ivp).
    state : numpy.ndarray
        6-element state vector [x, y, z, vx, vy, vz].

    Returns
    -------
    list
        Derivatives [vx, vy, vz, ax, ay, az].
    """
    r3 = (state[0] ** 2 + state[1] ** 2 + state[2] ** 2) ** 1.5
    return [
        state[3],
        state[4],
        state[5],
        -GM_SUN * state[0] / r3,
        -GM_SUN * state[1] / r3,
        -GM_SUN * state[2] / r3,
    ]


def integrate_trajectory(r_vec, v_vec, tof, points=500):
    """Numerically integrate a trajectory under solar gravity.

    Parameters
    ----------
    r_vec : array-like
        Initial position vector [x, y, z] in meters.
    v_vec : array-like
        Initial velocity vector [vx, vy, vz] in m/s.
    tof : float
        Time of flight in seconds.
    points : int, optional
        Number of output points (default: 500).

    Returns
    -------
    tuple of numpy.ndarray
        (positions, times) where positions is (N, 3) array of [x, y, z]
        and times is (N,) array of time values.

    Raises
    ------
    RuntimeError
        If the integrator fails to converge.
    """
    initial_state = np.empty(6)
    initial_state[:3] = r_vec
    initial_state[3:] = v_vec
    t_eval = np.linspace(0, tof, points)
    sol = solve_ivp(
        two_body_ode,
        [0, tof],
        initial_state,
        t_eval=t_eval,
        method="DOP853",
        rtol=1e-8,
        atol=1e-6,
    )
    if not sol.success:
        raise RuntimeError(f"Trajectory integration failed: {sol.message}")
    return sol.y[:3].T, sol.t
