import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.optimize import brentq
from scipy.integrate import solve_ivp

G = 6.67430e-11
M_SUN = 1.98847e30
GM_SUN = G * M_SUN
KM_TO_M = 1000
AU_TO_M = 1.496e11
AU_TO_KM = AU_TO_M / KM_TO_M
SEC_TO_DAY = 86400.0


# ==============================================================================
# Lambert Solver (Universal Variables)
# ==============================================================================


def lambert_solve(r1_vec, r2_vec, tof, mu=GM_SUN):
    r1_vec = np.asarray(r1_vec, dtype=float)
    r2_vec = np.asarray(r2_vec, dtype=float)

    r1_mag = np.linalg.norm(r1_vec)
    r2_mag = np.linalg.norm(r2_vec)

    cos_dnu = np.clip(np.dot(r1_vec, r2_vec) / (r1_mag * r2_mag), -1.0, 1.0)
    cross = np.cross(r1_vec, r2_vec)
    dnu = np.arccos(cos_dnu) if cross[2] >= 0 else 2 * np.pi - np.arccos(cos_dnu)

    A = np.sin(dnu) * np.sqrt(r1_mag * r2_mag / (1 - cos_dnu))

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


# ==============================================================================
# Trajectory Integration
# ==============================================================================


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


# ==============================================================================
# Orbital Mechanics
# ==============================================================================


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
    return M * np.sqrt(a**3 / GM_SUN) / SEC_TO_DAY


def find_factor_for_dv(r1, r2, target_dv, max_factor=50.0):
    dv_hoh, _, _ = hohmann_delta_v(r1, r2)
    if target_dv < dv_hoh:
        return 1.0, dv_hoh

    def residual(factor):
        a = ((r1 + r2) / 2) * factor
        e = 1 - r1 / a
        v1_circ = np.sqrt(GM_SUN / r1)
        v2_circ = np.sqrt(GM_SUN / r2)
        vt1 = np.sqrt(GM_SUN * (2 / r1 - 1 / a))
        h = r1 * vt1
        v_theta = h / r2
        v_arr_mag = np.sqrt(GM_SUN * (2 / r2 - 1 / a))
        v_radial = np.sqrt(max(0, v_arr_mag**2 - v_theta**2))
        dv_dep = np.linalg.norm(np.array([0, vt1 - v1_circ, 0]))
        dv_arr = np.linalg.norm(np.array([v_radial, v2_circ - v_theta, 0]))
        return dv_dep + dv_arr - target_dv

    try:
        factor = brentq(residual, 1.0, max_factor, xtol=1e-10, rtol=1e-12)
    except ValueError:
        factor = max_factor

    a = ((r1 + r2) / 2) * factor
    e = 1 - r1 / a
    v1_circ = np.sqrt(GM_SUN / r1)
    v2_circ = np.sqrt(GM_SUN / r2)
    vt1 = np.sqrt(GM_SUN * (2 / r1 - 1 / a))
    h = r1 * vt1
    v_theta = h / r2
    v_arr_mag = np.sqrt(GM_SUN * (2 / r2 - 1 / a))
    v_radial = np.sqrt(max(0, v_arr_mag**2 - v_theta**2))
    dv_dep = np.linalg.norm(np.array([0, vt1 - v1_circ, 0]))
    dv_arr = np.linalg.norm(np.array([v_radial, v2_circ - v_theta, 0]))

    return factor, dv_dep + dv_arr


def compute_transfer_trajectory(r1, r2, target_dv, points=500, target_ecc=0.0, target_rot=0.0):
    dv_dep_h, dv_arr_h, dv_total_h = hohmann_delta_v(r1, r2)
    hohmann_tof = np.pi * np.sqrt(((r1 + r2) / 2) ** 3 / GM_SUN)

    if abs(target_dv - dv_total_h) < 1.0:
        a = (r1 + r2) / 2
        e = (r2 - r1) / (r2 + r1)
        thetas = np.linspace(0, np.pi, points)
        radii = (a * (1 - e**2)) / (1 + e * np.cos(thetas))
        x = radii * np.cos(thetas) / AU_TO_M
        y = radii * np.sin(thetas) / AU_TO_M
        return (
            x,
            y,
            np.array([r1 / AU_TO_M, 0.0]),
            np.array([-r2 / AU_TO_M, 0.0]),
            np.pi,
        )

    v1_circ = np.sqrt(GM_SUN / r1)

    best = None
    for tof_frac in np.linspace(0.05, 0.95, 60):
        tof = hohmann_tof * tof_frac
        for dnu in np.linspace(0.3, np.pi - 0.05, 40):
            orbit_angle = dnu - target_rot
            r2_actual = r2 * (1 - target_ecc**2) / (1 + target_ecc * np.cos(orbit_angle))
            v2_circ = np.sqrt(GM_SUN / r2_actual)
            r1_vec = np.array([r1, 0.0, 0.0])
            r2_vec = np.array([r2_actual * np.cos(dnu), r2_actual * np.sin(dnu), 0.0])
            try:
                v1, v2 = lambert_solve(r1_vec, r2_vec, tof)
            except ValueError:
                continue
            dv_dep = np.linalg.norm(v1 - np.array([0.0, v1_circ, 0.0]))
            dv_arr = np.linalg.norm(v2 - np.array([0.0, v2_circ, 0.0]))
            total_dv = dv_dep + dv_arr
            if total_dv <= target_dv:
                if best is None or tof < best[0]:
                    best = (tof, dnu, v1, r1_vec, r2_actual)
        if best is not None:
            break

    if best is None:
        tof = hohmann_tof * 0.5
        dnu = np.pi * 0.75
        orbit_angle = dnu - target_rot
        r2_actual = r2 * (1 - target_ecc**2) / (1 + target_ecc * np.cos(orbit_angle))
        r1_vec = np.array([r1, 0.0, 0.0])
        r2_vec = np.array([r2_actual * np.cos(dnu), r2_actual * np.sin(dnu), 0.0])
        try:
            v1, v2 = lambert_solve(r1_vec, r2_vec, tof)
        except ValueError:
            a = (r1 + r2) / 2
            e = (r2 - r1) / (r2 + r1)
            thetas = np.linspace(0, np.pi, points)
            radii = (a * (1 - e**2)) / (1 + e * np.cos(thetas))
            x = radii * np.cos(thetas) / AU_TO_M
            y = radii * np.sin(thetas) / AU_TO_M
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


# ==============================================================================
# Visualization
# ==============================================================================


def plot_orbit(ax, body_data, rotation=0):
    a = (body_data["aphelion"] + body_data["perihelion"]) / 2
    e = (body_data["aphelion"] - body_data["perihelion"]) / (
        body_data["aphelion"] + body_data["perihelion"]
    )
    theta = np.linspace(0, 2 * np.pi, 1000)
    r = (a * (1 - e**2)) / (1 + e * np.cos(theta))
    ax.plot(
        r * np.cos(theta + rotation) / AU_TO_KM,
        r * np.sin(theta + rotation) / AU_TO_KM,
        linewidth=1.5,
        label=f"Orbit for {body_data['name']}",
    )


def plot_transfer(ax, x, y, dep, arr, label, color, linestyle="-"):
    ax.plot(x, y, linestyle=linestyle, color=color, linewidth=2, label=label)
    ax.plot(dep[0], dep[1], marker="^", color=color, markersize=10, zorder=5)
    ax.plot(arr[0], arr[1], marker="s", color=color, markersize=10, zorder=5)

    n = len(x)
    if n > 10:
        i = n // 4
        ax.add_patch(
            FancyArrowPatch(
                (x[i], y[i]),
                (x[i + 2], y[i + 2]),
                arrowstyle="->",
                color=color,
                mutation_scale=15,
                lw=1.5,
            )
        )


def visualize(r1, r2, target_dv, bodies_data, stats=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(0, 0, "yo", label="Sun", markersize=15)

    def orbit_ecc(body_data):
        ap = body_data.get("aphelion", 0)
        pe = body_data.get("perihelion", 0)
        denom = ap + pe
        if denom == 0:
            return 0.0
        return (ap - pe) / denom

    e_start = orbit_ecc(bodies_data[0]) if len(bodies_data) > 0 else 0.0
    e_end = orbit_ecc(bodies_data[1]) if len(bodies_data) > 1 else 0.0

    alpha_start = np.arccos(np.clip(-e_start, -1, 1))

    start_name = bodies_data[0]["name"] if len(bodies_data) > 0 else "Start"
    end_name = bodies_data[1]["name"] if len(bodies_data) > 1 else "End"

    _, _, hohmann_dv = hohmann_delta_v(r1, r2)
    x_h, y_h, dep_h, arr_h, nu_h = compute_transfer_trajectory(r1, r2, hohmann_dv)
    alpha_end_h = nu_h - np.arccos(np.clip(-e_end, -1, 1))

    for i, body in enumerate(bodies_data):
        rot = alpha_start if i == 0 else alpha_end_h
        plot_orbit(ax, body, rotation=rot)

    plot_transfer(
        ax, x_h, y_h, dep_h, arr_h, "Hohmann Transfer", "cyan", linestyle="--"
    )

    if target_dv > hohmann_dv + 1.0:
        x_f, y_f, dep_f, arr_f, nu_f = compute_transfer_trajectory(
            r1, r2, target_dv, target_ecc=e_end, target_rot=alpha_end_h
        )
        plot_transfer(ax, x_f, y_f, dep_f, arr_f, "Fast Transfer", "red")

    ax.plot([], [], "g^", markersize=10, label="Departure Burn")
    ax.plot([], [], "ms", markersize=10, label="Arrival Burn")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("Distance (AU)", fontsize=11)
    ax.set_ylabel("Distance (AU)", fontsize=11)
    ax.set_title(
        "Planetary Transfer Trajectory\n(ODE Integration with Lambert Solver)",
        fontsize=13,
    )

    if stats:
        stats_text = (
            "─── Hohmann Transfer ───\n"
            f"Δv required: {stats['hohmann_dv']:.2f} km/s\n"
            f"Est. time:    {stats['hohmann_time']:.1f} days\n\n"
            "─── Fast Transfer ───\n"
            f"Δv used:       {stats['fast_dv']:.2f} km/s\n"
            f"Energy factor: {stats['fast_factor']:.2f}\n"
            f"Est. time:     {stats['fast_time']:.1f} days"
        )
        props = dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.7)
        ax.text(
            0.02,
            0.02,
            stats_text,
            transform=ax.transAxes,
            fontsize=13,
            verticalalignment="bottom",
            fontfamily="monospace",
            color="white",
            bbox=props,
        )

    plt.tight_layout()


# ==============================================================================
# CLI
# ==============================================================================


def main():
    print("=== Orbital Transfer Calculator (by Delta-V Budget) ===\n")

    with open("./data.json") as f:
        dataset = json.load(f)["bodies"]

    start_id = input("Start body ID: ").strip().lower()
    end_id = input("Destination body ID: ").strip().lower()

    start_body = next((b for b in dataset if b["id"].lower() == start_id), None)
    end_body = next((b for b in dataset if b["id"].lower() == end_id), None)

    if not start_body or not end_body:
        print("Body not found!")
        return

    def get_ap_per(body):
        if body.get("apogee", 0) > 0 and body.get("perigee", 0) > 0:
            return body["apogee"], body["perigee"]
        sma = body.get("semimajorAxis", 0)
        return sma, sma

    a1, p1 = get_ap_per(start_body)
    a2, p2 = get_ap_per(end_body)

    try:
        start = OrbitalBody(a1, p1)
        end = OrbitalBody(a2, p2)
    except ValueError:
        print("\nError: Invalid orbit parameters for the selected bodies.")
        return

    print(
        f"\nTransfer: {start_body.get('name', start_id)} → {end_body.get('name', end_id)}"
    )

    try:
        available_dv_km = float(input("\nAvailable total Delta-V (km/s): ") or 30.0)
    except ValueError:
        available_dv_km = 30.0

    available_dv_m = available_dv_km * KM_TO_M

    dv_dep, dv_arr, total_hohmann = hohmann_delta_v(start.sma, end.sma)
    fast_factor, fast_dv = find_factor_for_dv(start.sma, end.sma, available_dv_m)

    print("\n" + "=" * 65)
    print("EFFICIENT TRANSFER (Hohmann)")
    print(f"Δv required:   {total_hohmann / 1000:.2f} km/s")
    print(f"Est. time:     {transfer_time(start.sma, end.sma, 1.0):.1f} days")

    if available_dv_m < total_hohmann - 1:
        print(f"\nWARNING: You do not have enough Delta-V for this journey.")
    else:
        print("\nFAST TRANSFER (Budget Maxed)")
        print(f"Δv used:       {fast_dv / 1000:.2f} km/s")
        print(f"Energy factor: {fast_factor:.2f}")
        print(
            f"Est. time:     {transfer_time(start.sma, end.sma, fast_factor):.1f} days"
        )
    print("=" * 65)


if __name__ == "__main__":
    main()
