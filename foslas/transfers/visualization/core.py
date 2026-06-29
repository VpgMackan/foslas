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

import pykep as pk
from datetime import datetime, timedelta

from ...constants import AU_TO_KM, AU_TO_M, GM_SUN


def get_body_ecliptic(body_name, time_offset_days=0):
    now = datetime.now().utcnow() + timedelta(days=time_offset_days)
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


def propagate_orbit_position(body_data, current_lon, current_rotation, dt_days):
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
