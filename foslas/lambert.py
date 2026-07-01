"""Lambert's problem solver for orbital transfers.

Solves Lambert's problem: given two position vectors and a time of flight,
find the velocity vectors at both positions. Delegates to pykep's Lambert
solver implementation.
"""

import numpy as np
import pykep as pk

from .constants import GM_SUN


def lambert_solve(r1_vec, r2_vec, tof, mu=GM_SUN):
    """Solve Lambert's problem using pykep.

    Parameters
    ----------
    r1_vec : array-like
        Departure position vector in meters.
    r2_vec : array-like
        Arrival position vector in meters.
    tof : float
        Time of flight in seconds.
    mu : float, optional
        Gravitational parameter (default: GM_SUN).

    Returns
    -------
    tuple
        (v1, v2) velocity vectors at departure and arrival in m/s.
    """
    lp = pk.lambert_problem(r1_vec, r2_vec, tof, mu)
    v1 = np.array(lp.v0[0])
    v2 = np.array(lp.v1[0])
    return v1, v2
