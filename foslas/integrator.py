import numpy as np
from scipy.integrate import solve_ivp

from .constants import GM_SUN


def two_body_ode(t, state):
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
    return sol.y[:3].T, sol.t
