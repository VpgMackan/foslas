"""Hohmann transfer calculations.

Provides functions for computing minimum-energy Hohmann transfers
between two circular orbits.
"""

import numpy as np

from ..constants import GM_SUN, AU_TO_M


def hohmann_delta_v(r1, r2):
    """Compute delta-V for a Hohmann transfer between two circular orbits.

    Parameters
    ----------
    r1 : float
        Radius of departure orbit in meters.
    r2 : float
        Radius of arrival orbit in meters.

    Returns
    -------
    tuple of float
        (departure_burn, arrival_burn, total_delta_v) in m/s.
    """
    v1_circ = np.sqrt(GM_SUN / r1)
    v2_circ = np.sqrt(GM_SUN / r2)
    a_t = (r1 + r2) / 2
    v_dep = np.sqrt(GM_SUN * (2 / r1 - 1 / a_t)) - v1_circ
    v_arr = v2_circ - np.sqrt(GM_SUN * (2 / r2 - 1 / a_t))
    return abs(v_dep), abs(v_arr), abs(v_dep) + abs(v_arr)


def hohmann_trajectory(r1, r2, points):
    """Compute a Hohmann transfer arc between two circular orbits.

    Parameters
    ----------
    r1 : float
        Radius of departure orbit in meters.
    r2 : float
        Radius of arrival orbit in meters.
    points : int
        Number of points on the trajectory.

    Returns
    -------
    tuple of numpy.ndarray
        (x, y) coordinates in AU.
    """
    a = (r1 + r2) / 2
    e = (r2 - r1) / (r2 + r1)
    thetas = np.linspace(0, np.pi, points)
    radii = (a * (1 - e**2)) / (1 + e * np.cos(thetas))
    x = radii * np.cos(thetas) / AU_TO_M
    y = radii * np.sin(thetas) / AU_TO_M
    return x, y
