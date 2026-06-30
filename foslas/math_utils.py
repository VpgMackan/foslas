"""Mathematical utilities for orbital mechanics calculations.

Provides shared numerical functions used across the codebase.
"""

import numpy as np
from scipy.optimize import newton

from .constants import GM_SUN


def solve_kepler(M, e, tol=1e-12):
    """Solve Kepler's equation M = E - e*sin(E) for E.

    Parameters
    ----------
    M : float
        Mean anomaly in radians.
    e : float
        Eccentricity (0 <= e < 1 for elliptical orbits).
    tol : float, optional
        Convergence tolerance (default: 1e-12).

    Returns
    -------
    float
        Eccentric anomaly E in radians.
    """
    return newton(lambda E: E - e * np.sin(E) - M, M, tol=tol)


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