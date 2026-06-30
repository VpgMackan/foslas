"""Mathematical utilities for orbital mechanics calculations.

Provides shared numerical functions used across the codebase.
"""

import numpy as np

from .constants import GM_SUN, KEPLER_MAX_ITERATIONS, KEPLER_TOLERANCE


def solve_kepler(M, e, max_iterations=KEPLER_MAX_ITERATIONS, tolerance=KEPLER_TOLERANCE):
    """Solve Kepler's equation M = E - e*sin(E) for E.

    Parameters
    ----------
    M : float
        Mean anomaly in radians.
    e : float
        Eccentricity (0 <= e < 1 for elliptical orbits).
    max_iterations : int, optional
        Maximum number of iterations (default: KEPLER_MAX_ITERATIONS).
    tolerance : float, optional
        Convergence tolerance (default: KEPLER_TOLERANCE).

    Returns
    -------
    float
        Eccentric anomaly E in radians.
    """
    E = M
    for _ in range(max_iterations):
        f = E - e * np.sin(E) - M
        fp = 1 - e * np.cos(E)
        delta = -f / fp
        E += delta
        if abs(delta) < tolerance:
            break
    return E


def true_anomaly_from_eccentric_anomaly(E, e):
    """Convert eccentric anomaly to true anomaly.

    Parameters
    ----------
    E : float
        Eccentric anomaly in radians.
    e : float
        Eccentricity.

    Returns
    -------
    float
        True anomaly nu in radians.
    """
    return 2.0 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E / 2.0),
        np.sqrt(1 - e) * np.cos(E / 2.0),
    )


def orbital_velocity_components(a, e, nu, mu=GM_SUN):
    """Compute orbital velocity components for an elliptical orbit.

    Parameters
    ----------
    a : float
        Semi-major axis in meters.
    e : float
        Eccentricity.
    nu : float
        True anomaly in radians.
    mu : float, optional
        Gravitational parameter (default: GM_SUN).

    Returns
    -------
    tuple
        (v_r, v_theta) radial and transverse velocity components in m/s.
    """
    h = np.sqrt(mu * a * (1 - e**2))
    v_r = (mu / h) * e * np.sin(nu)
    v_theta = (mu / h) * (1 + e * np.cos(nu))
    return v_r, v_theta