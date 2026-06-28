import numpy as np
from scipy.optimize import brentq

from .constants import GM_SUN


def stumpff_S(z):
    if abs(z) < 1e-6:
        return 1.0 / 6.0
    elif z > 0:
        sz = np.sqrt(z)
        return (sz - np.sin(sz)) / (sz**3)
    else:
        sz = np.sqrt(-z)
        return (np.sinh(sz) - sz) / ((-z) ** 1.5)


def stumpff_C(z):
    if abs(z) < 1e-6:
        return 0.5
    elif z > 0:
        return (1 - np.cos(np.sqrt(z))) / z
    else:
        return (np.cosh(np.sqrt(-z)) - 1) / (-z)


def lambert_solve(r1_vec, r2_vec, tof, mu=GM_SUN):
    r1_vec = np.asarray(r1_vec, dtype=float)
    r2_vec = np.asarray(r2_vec, dtype=float)

    r1_mag = np.linalg.norm(r1_vec)
    r2_mag = np.linalg.norm(r2_vec)

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
        t = (x**3 * S + A * np.sqrt(y)) / np.sqrt(mu)
        return t - tof

    z_bound = 4 * np.pi**2 - 0.1

    z = None
    try:
        z = brentq(tof_residual, -2.0, 4.0, xtol=1e-12, rtol=1e-12)
    except ValueError:
        n_scan = 200
        for z_lo, z_hi in [(0.01, z_bound), (-z_bound, -0.01)]:
            z_scan = np.linspace(z_lo, z_hi, n_scan)
            res_scan = np.array([tof_residual(zz) for zz in z_scan])
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
