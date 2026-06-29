"""Visualization module for orbital transfer trajectories.

Provides functions to plot planetary orbits and transfer trajectories
using matplotlib.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

import astropy.units as u
from astropy.coordinates import get_body
from astropy.time import Time

from .constants import AU_TO_KM, AU_TO_M, GM_SUN
from .orbital import hohmann_delta_v, compute_transfer_trajectory
from .lambert import lambert_solve
from .integrator import integrate_trajectory


def get_body_ecliptic(body_name, time_offset_days=0):
    time = Time.now()
    # time = Time.strptime("2012-Jun-30 23:59:60", "%Y-%b-%d %H:%M:%S")
    if time_offset_days != 0:
        time = time + time_offset_days * u.day
    ecl = get_body(body_name, time).heliocentricmeanecliptic
    return ecl.distance.to(u.au).value, ecl.lon.rad


def compute_orbit_rotation(body_data, planet_lon, planet_r_au):
    a = (body_data["aphelion"] + body_data["perihelion"]) / 2
    e = (body_data["aphelion"] - body_data["perihelion"]) / (
        body_data["aphelion"] + body_data["perihelion"]
    )
    if e < 1e-10:
        return planet_lon
    planet_r_km = planet_r_au * AU_TO_KM
    cos_val = np.clip((a * (1 - e**2) / planet_r_km - 1) / e, -1, 1)
    nu = np.arccos(cos_val)
    return planet_lon - nu


def plot_orbit(ax, body_data, rotation=0):
    """Plot a full elliptical orbit for a celestial body.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    body_data : dict
        Body data with 'aphelion', 'perihelion', and 'englishName' fields.
    rotation : float, optional
        Rotation angle in radians (default: 0).
    """
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
        label=f"Orbit for {body_data['englishName']}",
    )

    arrow_theta = np.pi / 3
    arrow_r = (a * (1 - e**2)) / (1 + e * np.cos(arrow_theta))
    px = arrow_r * np.cos(arrow_theta + rotation) / AU_TO_KM
    py = arrow_r * np.sin(arrow_theta + rotation) / AU_TO_KM
    dx = -np.sin(arrow_theta + rotation) * 0.05
    dy = np.cos(arrow_theta + rotation) * 0.05
    ax.add_patch(
        FancyArrowPatch(
            (px - dx, py - dy),
            (px + dx, py + dy),
            arrowstyle="->",
            color="black",
            mutation_scale=10,
            lw=1.0,
        )
    )


def plot_transfer(ax, x, y, dep, arr, label, color, linestyle="-"):
    """Plot a transfer trajectory with departure and arrival markers.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    x, y : numpy.ndarray
        Trajectory coordinates in AU.
    dep : numpy.ndarray
        Departure burn point [x, y] in AU.
    arr : numpy.ndarray
        Arrival burn point [x, y] in AU.
    label : str
        Label for the legend.
    color : str
        Color for the trajectory line.
    linestyle : str, optional
        Line style (default: "-").
    """
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


def _propagate_orbit_position(body_data, current_lon, current_rotation, dt_days):
    """Propagate a body along its orbit by a time offset in days."""
    sma_km = body_data.get("semimajorAxis", 0)
    aph = body_data.get("aphelion", 0)
    peri = body_data.get("perihelion", 0)
    if sma_km <= 0 or (aph + peri) <= 0:
        r = body_data.get("semimajorAxis", 0) / AU_TO_KM
        return r * np.cos(current_lon), r * np.sin(current_lon)

    a = sma_km * 1000.0
    e = (aph - peri) / (aph + peri)
    if e < 1e-10:
        angle = current_lon + 2 * np.pi * (dt_days / 365.25)
        r = a / AU_TO_M
        return r * np.cos(angle), r * np.sin(angle)

    nu0 = current_lon - current_rotation
    sin_half_nu0 = np.sin(nu0 / 2.0)
    cos_half_nu0 = np.cos(nu0 / 2.0)
    E0 = 2.0 * np.arctan2(
        np.sqrt(1 - e) * sin_half_nu0,
        np.sqrt(1 + e) * cos_half_nu0,
    )
    M0 = E0 - e * np.sin(E0)
    mean_motion = np.sqrt(GM_SUN / a**3)
    M1 = M0 + mean_motion * dt_days * 86400.0

    E = M1
    for _ in range(20):
        f = E - e * np.sin(E) - M1
        fp = 1 - e * np.cos(E)
        delta = -f / fp
        E += delta
        if abs(delta) < 1e-12:
            break

    r = a * (1 - e * np.cos(E)) / AU_TO_M
    nu = 2.0 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E / 2.0),
        np.sqrt(1 - e) * np.cos(E / 2.0),
    )
    angle = current_rotation + nu
    return r * np.cos(angle), r * np.sin(angle)


def visualize(r1, r2, target_dv, bodies_data, stats=None):
    """Create a complete visualization of orbital transfer trajectories.

    Plots the Sun, both planetary orbits, Hohmann transfer, and optionally
    a fast transfer trajectory.

    Parameters
    ----------
    r1 : float
        Radius of departure orbit in meters.
    r2 : float
        Radius of arrival orbit in meters.
    target_dv : float
        Available delta-V budget in m/s.
    bodies_data : list of dict
        Body data for departure and arrival bodies.
    stats : dict, optional
        Statistics to display in the plot (hohmann_dv, hohmann_time, etc.).
    """

    def _rotate(x, y, angle):
        c, s = np.cos(angle), np.sin(angle)
        return x * c - y * s, x * s + y * c

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(0, 0, "yo", label="Sun", markersize=15)

    e_end = (
        (bodies_data[1].get("aphelion", 0) - bodies_data[1].get("perihelion", 0))
        / (bodies_data[1].get("aphelion", 0) + bodies_data[1].get("perihelion", 0))
        if len(bodies_data) > 1
        else 0.0
    )

    dep_aph = bodies_data[0].get("aphelion", 0)
    dep_peri = bodies_data[0].get("perihelion", 0)
    dep_ecc = (
        (dep_aph - dep_peri) / (dep_aph + dep_peri) if (dep_aph + dep_peri) > 0 else 0.0
    )

    dep_r, dep_lon = get_body_ecliptic(bodies_data[0]["englishName"])
    arr_r, arr_lon = get_body_ecliptic(bodies_data[1]["englishName"])

    sx = dep_r * np.cos(dep_lon)
    sy = dep_r * np.sin(dep_lon)
    ex = arr_r * np.cos(arr_lon)
    ey = arr_r * np.sin(arr_lon)

    ax.plot(sx, sy, "bo", markersize=8, label=bodies_data[0].get("englishName"))
    ax.plot(ex, ey, "ro", markersize=8, label=bodies_data[1].get("englishName"))

    dep_rotation = compute_orbit_rotation(bodies_data[0], dep_lon, dep_r)
    arr_rotation = compute_orbit_rotation(bodies_data[1], arr_lon, arr_r)

    plot_orbit(ax, bodies_data[0], rotation=dep_rotation)
    plot_orbit(ax, bodies_data[1], rotation=arr_rotation)

    dep_sma_m = bodies_data[0].get("semimajorAxis", 0) * 1000.0
    arr_sma_m = bodies_data[1].get("semimajorAxis", 0) * 1000.0

    _, _, hohmann_dv = hohmann_delta_v(dep_sma_m, arr_sma_m)
    x_h, y_h, dep_h, arr_h, nu_h, hohmann_tof_s = compute_transfer_trajectory(
        dep_sma_m,
        arr_sma_m,
        hohmann_dv,
        target_ecc=e_end,
        target_rot=arr_rotation - dep_lon,
        dep_ecc=dep_ecc,
        dep_rot=dep_rotation - dep_lon,
    )

    x_h, y_h = _rotate(x_h, y_h, dep_lon)
    dep_h = np.array(_rotate(dep_h[0], dep_h[1], dep_lon))
    arr_h = np.array(_rotate(arr_h[0], arr_h[1], dep_lon))

    # plot_transfer(
    #     ax, x_h, y_h, dep_h, arr_h, "Hohmann Transfer", "cyan", linestyle="--"
    # )

    fast_tof_s = None
    if target_dv > hohmann_dv + 1.0:
        dep_rot = dep_rotation - dep_lon
        x_f, y_f, dep_f, arr_f, nu_f, fast_tof_s = compute_transfer_trajectory(
            dep_sma_m,
            arr_sma_m,
            target_dv,
            target_ecc=e_end,
            target_rot=arr_rotation - dep_lon,
            dep_ecc=dep_ecc,
            dep_rot=dep_rotation - dep_lon,
        )
        future_r, future_lon = get_body_ecliptic(
            bodies_data[1]["englishName"], time_offset_days=fast_tof_s / 86400.0
        )
        r1_vec = np.array([dep_r * np.cos(dep_lon), dep_r * np.sin(dep_lon), 0.0]) * AU_TO_M
        r2_vec = np.array(
            [future_r * np.cos(future_lon), future_r * np.sin(future_lon), 0.0]
        ) * AU_TO_M
        v1, v2 = lambert_solve(r1_vec, r2_vec, fast_tof_s)
        positions, _ = integrate_trajectory(r1_vec, v1, fast_tof_s, 500)
        x_f = positions[:, 0] / AU_TO_M
        y_f = positions[:, 1] / AU_TO_M
        dep_f = np.array([r1_vec[0] / AU_TO_M, r1_vec[1] / AU_TO_M])
        arr_f = np.array([r2_vec[0] / AU_TO_M, r2_vec[1] / AU_TO_M])
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
        actual_fast_dv = stats["fast_dv"]
        actual_fast_factor = stats["fast_factor"]
        actual_fast_time = (
            fast_tof_s / 86400.0 if fast_tof_s is not None else hohmann_tof_s / 86400.0
        )
        actual_hohmann_time = hohmann_tof_s / 86400.0

        if target_dv > hohmann_dv + 1.0:
            future_time = actual_fast_time
        else:
            future_time = actual_hohmann_time

        planet_x, planet_y = _propagate_orbit_position(
            bodies_data[1], arr_lon, arr_rotation, future_time
        )

        ax.plot(
            planet_x,
            planet_y,
            marker="o",
            linestyle="None",
            color="green",
            markersize=8,
            label=f"{bodies_data[1].get('englishName')} future",
        )

        stats_text = (
            # "--- Hohmann Transfer ---\n"
            # f"Dv required: {actual_hohmann_dv:.2f} km/s\n"
            # f"Est. time:    {actual_hohmann_time:.1f} days\n\n"
            "--- Fast Transfer ---\n"
            f"Dv used:       {actual_fast_dv:.2f} km/s\n"
            f"Energy factor: {actual_fast_factor:.2f}\n"
            f"Est. time:     {actual_fast_time:.1f} days"
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
