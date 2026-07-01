"""Visualization module for orbital transfer trajectories.

Provides the main visualization function for plotting transfer trajectories.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from ..hohmann import hohmann_delta_v
from .. import compute_transfer_trajectory
from ..base import compute_eccentricity
from .core import (
    get_body_ecliptic,
    compute_orbit_rotation,
    plot_orbit,
    plot_transfer,
)

__all__ = [
    "visualize",
    "plot_orbit",
    "plot_transfer",
    "get_body_ecliptic",
    "compute_orbit_rotation",
]


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
        compute_eccentricity(
            bodies_data[1].get("aphelion", 0),
            bodies_data[1].get("perihelion", 0),
        )
        if len(bodies_data) > 1
        else 0.0
    )

    dep_ecc = compute_eccentricity(
        bodies_data[0].get("aphelion", 0),
        bodies_data[0].get("perihelion", 0),
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

    fast_tof_s = None
    if target_dv > hohmann_dv + 1.0:
        try:
            x_f, y_f, dep_f, arr_f, nu_f, fast_tof_s = compute_transfer_trajectory(
                dep_sma_m,
                arr_sma_m,
                target_dv,
                points=500,
                target_ecc=e_end,
                target_rot=arr_rotation - dep_lon,
                dep_ecc=dep_ecc,
                dep_rot=dep_rotation - dep_lon,
            )
        except Exception:
            fast_tof_s = None

        if fast_tof_s is not None:
            try:
                plot_transfer(ax, x_f, y_f, dep_f, arr_f, "Fast Transfer", "red")
            except (ValueError, ZeroDivisionError):
                fast_tof_s = None

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
        display_fast_time = stats["fast_time"]
        display_hohmann_time = stats["hohmann_time"]

        if fast_tof_s is not None:
            future_time = fast_tof_s / 86400.0
        else:
            future_time = hohmann_tof_s / 86400.0

        future_r, future_lon = get_body_ecliptic(
            bodies_data[1]["englishName"], time_offset_days=future_time
        )
        planet_x = future_r * np.cos(future_lon)
        planet_y = future_r * np.sin(future_lon)

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
            "--- Fast Transfer ---\n"
            f"Dv used:       {actual_fast_dv:.2f} km/s\n"
            f"Energy factor: {actual_fast_factor:.2f}\n"
            f"Est. time:     {display_fast_time:.1f} days"
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
