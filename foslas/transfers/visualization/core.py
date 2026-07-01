"""Visualization module for orbital transfer trajectories.

Provides functions to plot planetary orbits and transfer trajectories
using matplotlib.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pykep as pk
from datetime import datetime, timedelta

from ...constants import AU_TO_KM, AU_TO_M, GM_SUN, JD_EPOCH_OFFSET
from ..base import compute_eccentricity


def _datetime_to_jd(dt):
    """Convert datetime to Julian date."""
    return dt.timestamp() / 86400.0 + JD_EPOCH_OFFSET


def get_body_ecliptic(body_name, time_offset_days=0):
    from ...bodies import compute_asteroid_ephemeris, ASTEROID_CATALOG

    body_name_lower = body_name.lower().replace(" ", "_")
    if body_name_lower in ASTEROID_CATALOG:
        jd = _datetime_to_jd(datetime.now() + timedelta(days=time_offset_days))
        return compute_asteroid_ephemeris(body_name_lower, jd)

    now = datetime.now() + timedelta(days=time_offset_days)
    jd = now.timestamp() / 86400.0 + 2440587.5
    days_since_j2000 = jd - 2451545.0
    epoch = pk.epoch(days_since_j2000)

    planet = pk.planet(pk.udpla.jpl_lp(body_name))
    r, _ = planet.eph(epoch)

    x, y, z = r
    distance_au = np.sqrt(x**2 + y**2 + z**2) / pk.AU
    longitude_rad = np.arctan2(y, x)

    return distance_au, longitude_rad


def compute_orbit_rotation(body_data, planet_lon, planet_r_au):
    a = (body_data["aphelion"] + body_data["perihelion"]) / 2
    e = compute_eccentricity(body_data["aphelion"], body_data["perihelion"])
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
    e = compute_eccentricity(body_data["aphelion"], body_data["perihelion"])
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
    if x is None or y is None or dep is None or arr is None:
        return
    if len(x) == 0 or len(y) == 0:
        return
    if not (np.any(np.isfinite(x)) and np.any(np.isfinite(y))):
        return

    ax.plot(x, y, linestyle=linestyle, color=color, linewidth=2, label=label)
    ax.plot(dep[0], dep[1], marker="^", color=color, markersize=10, zorder=5)
    ax.plot(arr[0], arr[1], marker="s", color=color, markersize=10, zorder=5)

    n = len(x)
    if n > 10:
        i = n // 4
        if i + 2 < n:
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
