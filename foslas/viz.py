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

from .constants import AU_TO_KM, AU_TO_M
from .orbital import hohmann_delta_v, compute_transfer_trajectory


def get_body_ecliptic(body_name):
    time = Time.strptime("2012-Jun-30 23:59:60", "%Y-%b-%d %H:%M:%S")
    # time = Time.now()
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

    a_a = (bodies_data[1]["aphelion"] + bodies_data[1]["perihelion"]) / 2
    e_a = (bodies_data[1]["aphelion"] - bodies_data[1]["perihelion"]) / (
        bodies_data[1]["aphelion"] + bodies_data[1]["perihelion"]
    )

    r1_m = dep_r * AU_TO_M
    r2_arrival_angle = dep_lon + np.pi - arr_rotation
    r2_m = a_a * (1 - e_a**2) / (1 + e_a * np.cos(r2_arrival_angle)) * 1000

    _, _, hohmann_dv = hohmann_delta_v(r1_m, r2_m)
    x_h, y_h, dep_h, arr_h, nu_h = compute_transfer_trajectory(r1_m, r2_m, hohmann_dv)

    x_h, y_h = _rotate(x_h, y_h, dep_lon)
    dep_h = np.array(_rotate(dep_h[0], dep_h[1], dep_lon))
    arr_h = np.array(_rotate(arr_h[0], arr_h[1], dep_lon))

    plot_transfer(
        ax, x_h, y_h, dep_h, arr_h, "Hohmann Transfer", "cyan", linestyle="--"
    )

    if target_dv > hohmann_dv + 1.0:
        x_f, y_f, dep_f, arr_f, nu_f = compute_transfer_trajectory(
            r1_m,
            a_a * 1000,
            target_dv,
            target_ecc=e_end,
            target_rot=arr_rotation - dep_lon,
        )
        x_f, y_f = _rotate(x_f, y_f, dep_lon)
        dep_f = np.array(_rotate(dep_f[0], dep_f[1], dep_lon))
        arr_f = np.array(_rotate(arr_f[0], arr_f[1], dep_lon))
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
        from .orbital import transfer_time

        actual_hohmann_time = transfer_time(r1_m, r2_m, 1.0)
        actual_hohmann_dv = hohmann_dv / 1000

        actual_fast_dv = stats["fast_dv"]
        actual_fast_factor = stats["fast_factor"]
        actual_fast_time = transfer_time(r1_m, a_a * 1000, actual_fast_factor)

        stats_text = (
            "--- Hohmann Transfer ---\n"
            f"Dv required: {actual_hohmann_dv:.2f} km/s\n"
            f"Est. time:    {actual_hohmann_time:.1f} days\n\n"
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
