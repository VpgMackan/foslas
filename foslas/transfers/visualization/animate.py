"""Animated GIF visualization of orbital transfer trajectories.

Renders a frame-by-frame animation showing planetary motion along their
orbits, a spacecraft traveling along the transfer trajectory, and the
transfer trail appearing behind the ship.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ...constants import AU_TO_KM, AU_TO_M, GM_SUN, SEC_TO_DAY
from ..hohmann import hohmann_delta_v
from .. import compute_transfer_trajectory
from .core import (
    get_body_ecliptic,
    compute_orbit_rotation,
)


def _rotate(x, y, angle):
    c, s = np.cos(angle), np.sin(angle)
    return x * c - y * s, x * s + y * c


def _propagate_lon(body_data, current_lon, current_rotation, dt_days):
    """Return the updated longitude of a body after dt_days."""
    sma_km = body_data.get("semimajorAxis", 0)
    aph = body_data.get("aphelion", 0)
    peri = body_data.get("perihelion", 0)
    if sma_km <= 0 or (aph + peri) <= 0:
        return current_lon

    a = sma_km * 1000.0
    e = (aph - peri) / (aph + peri)
    if e < 1e-10:
        return current_lon + 2 * np.pi * (dt_days / 365.25)

    nu0 = current_lon - current_rotation
    sin_half_nu0 = np.sin(nu0 / 2.0)
    cos_half_nu0 = np.cos(nu0 / 2.0)
    E0 = 2.0 * np.arctan2(
        np.sqrt(1 - e) * sin_half_nu0,
        np.sqrt(1 + e) * cos_half_nu0,
    )
    M0 = E0 - e * np.sin(E0)
    mean_motion = np.sqrt(GM_SUN / a**3)
    M1 = M0 + mean_motion * dt_days * SEC_TO_DAY

    E = M1
    for _ in range(20):
        f = E - e * np.sin(E) - M1
        fp = 1 - e * np.cos(E)
        delta = -f / fp
        E += delta
        if abs(delta) < 1e-12:
            break

    nu = 2.0 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E / 2.0),
        np.sqrt(1 - e) * np.cos(E / 2.0),
    )
    return current_rotation + nu


def animate_transfer(
    r1,
    r2,
    target_dv,
    bodies_data,
    output_path,
    fps=30,
    duration_seconds=None,
    dpi=150,
    stats=None,
    pad_frames=15,
):
    """Create an animated GIF of an orbital transfer.

    Parameters
    ----------
    r1 : float
        Semi-major axis of departure orbit in meters.
    r2 : float
        Semi-major axis of arrival orbit in meters.
    target_dv : float
        Available delta-V budget in m/s.
    bodies_data : list of dict
        Body data for departure and arrival bodies.
    output_path : str
        Output file path (should end in .gif).
    fps : int, optional
        Frames per second (default: 30).
    duration_seconds : float, optional
        Total animation duration in seconds. If None, scaled from transfer time.
    dpi : int, optional
        Resolution in dots per inch (default: 150).
    stats : dict, optional
        Statistics to display in the plot.
    pad_frames : int, optional
        Number of extra frames worth of hold time at start and end (default: 15).
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    dep_aph = bodies_data[0].get("aphelion", 0)
    dep_peri = bodies_data[0].get("perihelion", 0)
    dep_ecc = (
        (dep_aph - dep_peri) / (dep_aph + dep_peri) if (dep_aph + dep_peri) > 0 else 0.0
    )

    e_end = (
        (bodies_data[1].get("aphelion", 0) - bodies_data[1].get("perihelion", 0))
        / (bodies_data[1].get("aphelion", 0) + bodies_data[1].get("perihelion", 0))
        if len(bodies_data) > 1
        else 0.0
    )

    dep_r, dep_lon = get_body_ecliptic(bodies_data[0]["englishName"])
    arr_r, arr_lon = get_body_ecliptic(bodies_data[1]["englishName"])

    dep_rotation = compute_orbit_rotation(bodies_data[0], dep_lon, dep_r)
    arr_rotation = compute_orbit_rotation(bodies_data[1], arr_lon, arr_r)

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

    use_fast = target_dv > hohmann_dv + 1.0

    if use_fast:
        from ...lambert import lambert_solve
        from ...integrator import integrate_trajectory

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
        traj_x = positions[:, 0] / AU_TO_M
        traj_y = positions[:, 1] / AU_TO_M
        tof_s = fast_tof_s
        transfer_color = "red"
    else:
        traj_x = x_h
        traj_y = y_h
        tof_s = hohmann_tof_s
        transfer_color = "blue"

    tof_days = tof_s / SEC_TO_DAY
    n_traj = len(traj_x)

    if duration_seconds is None:
        duration_seconds = max(5.0, min(20.0, tof_days / 30.0))

    transfer_frames = int(fps * duration_seconds)
    ms_per_frame = 1000 / fps
    hold_ms = int(ms_per_frame * pad_frames)

    def _get_orbit_xy(body_data, rotation):
        a = (body_data["aphelion"] + body_data["perihelion"]) / 2
        e = (body_data["aphelion"] - body_data["perihelion"]) / (
            body_data["aphelion"] + body_data["perihelion"]
        )
        theta = np.linspace(0, 2 * np.pi, 500)
        r = (a * (1 - e**2)) / (1 + e * np.cos(theta))
        return r * np.cos(theta + rotation) / AU_TO_KM, r * np.sin(theta + rotation) / AU_TO_KM

    dep_orbit_x, dep_orbit_y = _get_orbit_xy(bodies_data[0], dep_rotation)
    arr_orbit_x, arr_orbit_y = _get_orbit_xy(bodies_data[1], arr_rotation)

    max_r = max(
        bodies_data[0].get("aphelion", 0),
        bodies_data[1].get("aphelion", 0),
    ) / AU_TO_KM
    margin = max_r * 0.15

    def _render_frame(t_days, frac, show_trail=True):
        idx = min(int(frac * (n_traj - 1)), n_traj - 1)

        fig_frame, ax_frame = plt.subplots(figsize=(10, 10))
        ax_frame.set_aspect("equal", adjustable="box")
        ax_frame.set_xlim(-max_r - margin, max_r + margin)
        ax_frame.set_ylim(-max_r - margin, max_r + margin)
        ax_frame.set_xlabel("Distance (AU)", fontsize=11)
        ax_frame.set_ylabel("Distance (AU)", fontsize=11)
        ax_frame.set_title(
            "Planetary Transfer Trajectory\n(Animation)",
            fontsize=13,
        )
        ax_frame.grid(True, alpha=0.3)

        ax_frame.plot(0, 0, "yo", markersize=15, label="Sun", zorder=10)
        ax_frame.plot(dep_orbit_x, dep_orbit_y, linewidth=1.2, color="steelblue", alpha=0.5)
        ax_frame.plot(arr_orbit_x, arr_orbit_y, linewidth=1.2, color="indianred", alpha=0.5)

        ax_frame.plot(
            traj_x, traj_y, linewidth=0.8, color=transfer_color, alpha=0.2, linestyle="--"
        )

        if show_trail and idx > 0:
            ax_frame.plot(
                traj_x[:idx + 1], traj_y[:idx + 1],
                linewidth=2, color=transfer_color, alpha=0.8,
            )

        ax_frame.plot(
            [traj_x[idx]], [traj_y[idx]],
            marker=(8, 2, 0), color="cyan", markersize=12, zorder=15,
        )

        dep_angle = _propagate_lon(bodies_data[0], dep_lon, dep_rotation, t_days)
        arr_angle = _propagate_lon(bodies_data[1], arr_lon, arr_rotation, t_days)
        dep_x = dep_r * np.cos(dep_angle)
        dep_y = dep_r * np.sin(dep_angle)
        arr_x = arr_r * np.cos(arr_angle)
        arr_y = arr_r * np.sin(arr_angle)

        ax_frame.plot([dep_x], [dep_y], "bo", markersize=8, label=bodies_data[0]["englishName"], zorder=12)
        ax_frame.plot([arr_x], [arr_y], "ro", markersize=8, label=bodies_data[1]["englishName"], zorder=12)

        ax_frame.plot([], [], "g^", markersize=10, label="Departure Burn")
        ax_frame.plot([], [], "ms", markersize=10, label="Arrival Burn")
        ax_frame.plot([], [], marker=(8, 2, 0), color="cyan", markersize=12, label="Spacecraft")
        ax_frame.legend(loc="upper right", fontsize=9)

        ax_frame.text(
            0.02, 0.95,
            f"Day {t_days:.0f} / {tof_days:.0f}",
            transform=ax_frame.transAxes, fontsize=11,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
            color="white",
        )

        fig_frame.tight_layout()
        return fig_frame

    frames_pil = []

    for i in range(transfer_frames):
        frac = i / max(transfer_frames - 1, 1)
        t_days = frac * tof_days
        show = (i > 0)
        fig_frame = _render_frame(t_days, frac, show_trail=show)
        fig_frame.savefig("/tmp/_foslas_tmp.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig_frame)
        frames_pil.append(Image.open("/tmp/_foslas_tmp.png").convert("RGB"))

    durations = [ms_per_frame] * len(frames_pil)
    durations[0] = hold_ms
    durations[-1] = hold_ms

    frames_pil[0].save(
        output_path,
        save_all=True,
        append_images=frames_pil[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )

    import os
    if os.path.exists("/tmp/_foslas_tmp.png"):
        os.remove("/tmp/_foslas_tmp.png")

    plt.close(fig)
    print(f"Animation saved to {output_path}")
