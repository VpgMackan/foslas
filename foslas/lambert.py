"""Lambert's problem solver for orbital transfers.

Solves Lambert's problem: given two position vectors and a time of flight,
find the velocity vectors at both positions. Uses the universal variable
formulation with Stumpff functions and Brent's root-finding method.
"""

import numpy as np
import pykep as pk

from .constants import GM_SUN


def lambert_solve(r1_vec, r2_vec, tof, mu=GM_SUN):
    r1 = list(r1_vec)
    r2 = list(r2_vec)
    lp = pk.lambert_problem(r1, r2, tof, mu)
    # Index 0 = prograde, single-rev solution
    v1 = np.array(lp.v0[0])
    v2 = np.array(lp.v1[0])
    return v1, v2
