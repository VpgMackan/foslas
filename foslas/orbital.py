import numpy as np
from scipy.optimize import brentq

from .constants import GM_SUN, KM_TO_M, AU_TO_M, SEC_TO_DAY
from .lambert import lambert_solve
from .integrator import integrate_trajectory


class OrbitalBody:
    def __init__(self, aphelion_km, perihelion_km):
        self.aphelion = aphelion_km * KM_TO_M
        self.perihelion = perihelion_km * KM_TO_M
        self.sma = (self.aphelion + self.perihelion) / 2

    @property
    def eccentricity(self):
        return (self.aphelion - self.perihelion) / (self.aphelion + self.perihelion)


def hohmann_delta_v(r1, r2):
    v1_circ = np.sqrt(GM_SUN / r1)
    v2_circ = np.sqrt(GM_SUN / r2)
    a_t = (r1 + r2) / 2
    v_dep = np.sqrt(GM_SUN * (2 / r1 - 1 / a_t)) - v1_circ
    v_arr = v2_circ - np.sqrt(GM_SUN * (2 / r2 - 1 / a_t))
    return v_dep, v_arr, v_dep + v_arr


def transfer_time(r1, r2, factor):
    a = ((r1 + r2) / 2) * factor
    e = 1 - r1 / a
    if e < 1e-10:
        return 0.0
    cos_E = np.clip((1 - r2 / a) / e, -1.0, 1.0)
    E = np.arccos(cos_E)
    M = E - e * np.sin(E)
    return M * np.sqrt(a ** 3 / GM_SUN) / SEC_TO_DAY


def _calc_dv_for_factor(r1, r2, factor):
    a = ((r1 + r2) / 2) * factor
    v1_circ = np.sqrt(GM_SUN / r1)
    v2_circ = np.sqrt(GM_SUN / r2)
    vt1 = np.sqrt(GM_SUN * (2 / r1 - 1 / a))
    v_theta = r1 * vt1 / r2
    v_arr_mag = np.sqrt(GM_SUN * (2 / r2 - 1 / a))
    v_radial_sq = v_arr_mag ** 2 - v_theta ** 2
    v_radial = np.sqrt(v_radial_sq) if v_radial_sq > 0 else 0.0
    dv_dep = abs(vt1 - v1_circ)
    dv_arr = np.hypot(v_radial, v2_circ - v_theta)
    return dv_dep + dv_arr


def find_factor_for_dv(r1, r2, target_dv, max_factor=50.0):
    dv_hoh, _, _ = hohmann_delta_v(r1, r2)
    if target_dv < dv_hoh:
        return 1.0, dv_hoh

    def residual(factor):
        return _calc_dv_for_factor(r1, r2, factor) - target_dv

    try:
        factor = brentq(residual, 1.0, max_factor, xtol=1e-10, rtol=1e-12)
    except ValueError:
        factor = max_factor

    return factor, _calc_dv_for_factor(r1, r2, factor)


def _hohmann_trajectory(r1, r2, points):
    a = (r1 + r2) / 2
    e = (r2 - r1) / (r2 + r1)
    thetas = np.linspace(0, np.pi, points)
    radii = (a * (1 - e ** 2)) / (1 + e * np.cos(thetas))
    x = radii * np.cos(thetas) / AU_TO_M
    y = radii * np.sin(thetas) / AU_TO_M
    return x, y


def _compute_r2_actual(r2, target_ecc, orbit_angle):
    return r2 * (1 - target_ecc ** 2) / (1 + target_ecc * np.cos(orbit_angle))


def _search_transfer(r1, r2, target_dv, points, target_ecc, target_rot, v1_circ):
    hohmann_tof = np.pi * np.sqrt(((r1 + r2) / 2) ** 3 / GM_SUN)

    tof_fractions = np.linspace(0.05, 0.95, 60)
    dnu_values = np.linspace(0.3, np.pi - 0.05, 40)

    r1_vec = np.array([r1, 0.0, 0.0])

    best = None
    for tof_frac in tof_fractions:
        tof = hohmann_tof * tof_frac
        for dnu in dnu_values:
            orbit_angle = dnu - target_rot
            r2_actual = _compute_r2_actual(r2, target_ecc, orbit_angle)
            v2_circ = np.sqrt(GM_SUN / r2_actual)
            r2_vec = np.array([r2_actual * np.cos(dnu), r2_actual * np.sin(dnu), 0.0])
            try:
                v1, v2 = lambert_solve(r1_vec, r2_vec, tof)
            except ValueError:
                continue
            dv_dep = np.sqrt(v1[0] ** 2 + (v1[1] - v1_circ) ** 2 + v1[2] ** 2)
            dv_arr = np.sqrt(v2[0] ** 2 + (v2[1] - v2_circ) ** 2 + v2[2] ** 2)
            total_dv = dv_dep + dv_arr
            if total_dv <= target_dv:
                if best is None or tof < best[0]:
                    best = (tof, dnu, v1, r1_vec, r2_actual)
        if best is not None:
            break

    return best, hohmann_tof


def compute_transfer_trajectory(r1, r2, target_dv, points=500, target_ecc=0.0, target_rot=0.0):
    dv_dep_h, dv_arr_h, dv_total_h = hohmann_delta_v(r1, r2)
    hohmann_tof = np.pi * np.sqrt(((r1 + r2) / 2) ** 3 / GM_SUN)

    if abs(target_dv - dv_total_h) < 1.0:
        x, y = _hohmann_trajectory(r1, r2, points)
        return (
            x,
            y,
            np.array([r1 / AU_TO_M, 0.0]),
            np.array([-r2 / AU_TO_M, 0.0]),
            np.pi,
        )

    v1_circ = np.sqrt(GM_SUN / r1)

    best, _ = _search_transfer(r1, r2, target_dv, points, target_ecc, target_rot, v1_circ)

    if best is None:
        tof = hohmann_tof * 0.5
        dnu = np.pi * 0.75
        orbit_angle = dnu - target_rot
        r2_actual = _compute_r2_actual(r2, target_ecc, orbit_angle)
        r1_vec = np.array([r1, 0.0, 0.0])
        r2_vec = np.array([r2_actual * np.cos(dnu), r2_actual * np.sin(dnu), 0.0])
        try:
            v1, v2 = lambert_solve(r1_vec, r2_vec, tof)
        except ValueError:
            x, y = _hohmann_trajectory(r1, r2, points)
            return (
                x,
                y,
                np.array([r1 / AU_TO_M, 0.0]),
                np.array([-r2 / AU_TO_M, 0.0]),
                np.pi,
            )
        best = (tof, dnu, v1, r1_vec, r2_actual)

    tof, dnu, v1, r1_vec, r2_actual = best
    positions, _ = integrate_trajectory(r1_vec, v1, tof, points)

    x = positions[:, 0] / AU_TO_M
    y = positions[:, 1] / AU_TO_M
    dep_burn = np.array([r1 / AU_TO_M, 0.0])
    arr_burn = np.array([r2_actual * np.cos(dnu) / AU_TO_M, r2_actual * np.sin(dnu) / AU_TO_M])

    return x, y, dep_burn, arr_burn, dnu
