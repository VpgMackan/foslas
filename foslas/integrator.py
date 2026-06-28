import numpy as np
from scipy.integrate import solve_ivp

from .constants import GM_SUN


def two_body_ode(t, state):
    r = np.linalg.norm(state[:3])
    return np.array(
        [
            state[3],
            state[4],
            state[5],
            -GM_SUN * state[0] / r**3,
            -GM_SUN * state[1] / r**3,
            -GM_SUN * state[2] / r**3,
        ]
    )


def integrate_trajectory(r_vec, v_vec, tof, points=500):
    initial_state = np.concatenate([r_vec, v_vec])
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
