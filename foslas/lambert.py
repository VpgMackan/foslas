"""Lambert's problem solver for orbital transfers.

Solves Lambert's problem: given two position vectors and a time of flight,
find the velocity vectors at both positions. Uses the universal variable
formulation with Stumpff functions and Brent's root-finding method.
"""

import numpy as np
from scipy.optimize import brentq

from .constants import GM_SUN


def stumpff_S(z):
    """Compute the Stumpff S function.

    Parameters
    ----------
    z : float
        Universal variable.

    Returns
    -------
    float
        Value of the Stumpff S function.
    """
    if abs(z) < 1e-6:
        return 1.0 / 6.0
    elif z > 0:
        sz = np.sqrt(z)
        return (sz - np.sin(sz)) / (sz ** 3)
    else:
        sz = np.sqrt(-z)
        return (np.sinh(sz) - sz) / ((-z) ** 1.5)


def stumpff_C(z):
    """Compute the Stumpff C function.

    Parameters
    ----------
    z : float
        Universal variable.

    Returns
    -------
    float
        Value of the Stumpff C function.
    """
    if abs(z) < 1e-6:
        return 0.5
    elif z > 0:
        return (1 - np.cos(np.sqrt(z))) / z
    else:
        return (np.cosh(np.sqrt(-z)) - 1) / (-z)


def _stumpff_C_vec(z):
    """Vectorized version of stumpff_C for array inputs.

    Parameters
    ----------
    z : numpy.ndarray
        Array of universal variables.

    Returns
    -------
    numpy.ndarray
        Array of Stumpff C values.
    """
    result = np.empty_like(z)
    small = np.abs(z) < 1e-6
    pos = (z > 0) & ~small
    neg = (z < 0) & ~small
    result[small] = 0.5
    sz = np.sqrt(z[pos])
    result[pos] = (1 - np.cos(sz)) / z[pos]
    sz = np.sqrt(-z[neg])
    result[neg] = (np.cosh(sz) - 1) / (-z[neg])
    return result


def _stumpff_S_vec(z):
    """Vectorized version of stumpff_S for array inputs.

    Parameters
    ----------
    z : numpy.ndarray
        Array of universal variables.

    Returns
    -------
    numpy.ndarray
        Array of Stumpff S values.
    """
    result = np.empty_like(z)
    small = np.abs(z) < 1e-6
    pos = (z > 0) & ~small
    neg = (z < 0) & ~small
    result[small] = 1.0 / 6.0
    sz = np.sqrt(z[pos])
    result[pos] = (sz - np.sin(sz)) / (z[pos] ** 1.5)
    sz = np.sqrt(-z[neg])
    result[neg] = (np.sinh(sz) - sz) / ((-z[neg]) ** 1.5)
    return result


def _tof_residual_vec(z, r1_mag, r2_mag, A, mu, tof):
    """Compute time-of-flight residual for vectorized z values.

    Parameters
    ----------
    z : numpy.ndarray
        Array of universal variables to evaluate.
    r1_mag : float
        Magnitude of the first position vector.
    r2_mag : float
        Magnitude of the second position vector.
    A : float
        Geometry constant from Lambert's problem.
    mu : float
        Standard gravitational parameter.
    tof : float
        Desired time of flight.

    Returns
    -------
    numpy.ndarray
        Residual values (computed_tof - target_tof).
    """
    C = _stumpff_C_vec(z)
    S = _stumpff_S_vec(z)
    denom = np.sqrt(C)
    result = np.full_like(z, 1e20)
    safe = denom > 1e-30
    y = r1_mag + r2_mag + A * (z * S - 1) / denom
    safe &= y > 0
    if np.any(safe):
        ys = y[safe]
        Cs = C[safe]
        Ss = S[safe]
        x = np.sqrt(ys / Cs)
        t = (x ** 3 * Ss + A * np.sqrt(ys)) / np.sqrt(mu)
        result[safe] = t - tof
    return result


def lambert_solve(r1_vec, r2_vec, tof, mu=GM_SUN):
    """Solve Lambert's problem for two position vectors and time of flight.

    Parameters
    ----------
    r1_vec : array-like
        First position vector [x, y, z] in meters.
    r2_vec : array-like
        Second position vector [x, y, z] in meters.
    tof : float
        Time of flight in seconds.
    mu : float, optional
        Standard gravitational parameter (default: GM_SUN).

    Returns
    -------
    tuple of numpy.ndarray
        Velocity vectors (v1, v2) at the two positions.

    Raises
    ------
    ValueError
        If no solution is found for the given inputs.
    """
    r1_vec = np.asarray(r1_vec, dtype=float)
    r2_vec = np.asarray(r2_vec, dtype=float)

    r1_mag = np.sqrt(r1_vec[0] ** 2 + r1_vec[1] ** 2 + r1_vec[2] ** 2)
    r2_mag = np.sqrt(r2_vec[0] ** 2 + r2_vec[1] ** 2 + r2_vec[2] ** 2)

    cos_dnu = np.clip(np.dot(r1_vec, r2_vec) / (r1_mag * r2_mag), -1.0, 1.0)
    cross = np.cross(r1_vec, r2_vec)
    dnu = np.arccos(cos_dnu) if cross[2] >= 0 else 2 * np.pi - np.arccos(cos_dnu)

    A = np.sin(dnu) * np.sqrt(r1_mag * r2_mag / (1 - cos_dnu))

    def tof_residual(z):
        C = stumpff_C(z)
        S = stumpff_S(z)
        denom = np.sqrt(C)
        if denom < 1e-30:
            return 1e20
        y = r1_mag + r2_mag + A * (z * S - 1) / denom
        if y < 0:
            return 1e20
        x = np.sqrt(y / C)
        t = (x ** 3 * S + A * np.sqrt(y)) / np.sqrt(mu)
        return t - tof

    z_bound = 4 * np.pi ** 2 - 0.1

    z = None
    try:
        z = brentq(tof_residual, -2.0, 4.0, xtol=1e-12, rtol=1e-12)
    except ValueError:
        n_scan = 200
        for z_lo, z_hi in [(0.01, z_bound), (-z_bound, -0.01)]:
            z_scan = np.linspace(z_lo, z_hi, n_scan)
            res_scan = _tof_residual_vec(z_scan, r1_mag, r2_mag, A, mu, tof)
            valid = np.abs(res_scan) < 1e19
            z_valid = z_scan[valid]
            res_valid = res_scan[valid]

            for i in range(len(res_valid) - 1):
                if res_valid[i] * res_valid[i + 1] < 0:
                    z = brentq(
                        tof_residual, z_valid[i], z_valid[i + 1], xtol=1e-12, rtol=1e-12
                    )
                    break

            if z is not None:
                break

    if z is None:
        raise ValueError("Lambert solver failed: no solution found")

    C = stumpff_C(z)
    S = stumpff_S(z)
    y = r1_mag + r2_mag + A * (z * S - 1) / np.sqrt(C)

    f = 1 - y / r1_mag
    g = A * np.sqrt(y / mu)
    g_dot = 1 - y / r2_mag

    v1 = (r2_vec - f * r1_vec) / g
    v2 = (g_dot * r2_vec - r1_vec) / g

    return v1, v2
