"""Visualization module for orbital transfer trajectories.

Provides functions to plot planetary orbits and transfer trajectories
using matplotlib.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

from .constants import AU_TO_KM
from .orbital import hohmann_delta_v, compute_transfer_trajectory


def plot_orbit(ax, body_data, rotation=0):
    """Plot a full elliptical orbit for a celestial body.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    body_data : dict
        Body data with 'aphelion', 'perihelion', and 'name' fields.
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
        label=f"Orbit for {body_data['name']}",
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
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(0, 0, "yo", label="Sun", markersize=15)

    e_start = (bodies_data[0].get("aphelion", 0) - bodies_data[0].get("perihelion", 0)) / (
        bodies_data[0].get("aphelion", 0) + bodies_data[0].get("perihelion", 0)
    ) if len(bodies_data) > 0 else 0.0
    e_end = (bodies_data[1].get("aphelion", 0) - bodies_data[1].get("perihelion", 0)) / (
        bodies_data[1].get("aphelion", 0) + bodies_data[1].get("perihelion", 0)
    ) if len(bodies_data) > 1 else 0.0

    alpha_start = np.arccos(np.clip(-e_start, -1, 1))

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
            "--- Hohmann Transfer ---\n"
            f"Dv required: {stats['hohmann_dv']:.2f} km/s\n"
            f"Est. time:    {stats['hohmann_time']:.1f} days\n\n"
            "--- Fast Transfer ---\n"
            f"Dv used:       {stats['fast_dv']:.2f} km/s\n"
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
