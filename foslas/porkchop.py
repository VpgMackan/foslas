"""Porkchop plot generation for interplanetary transfer analysis.

Sweeps launch dates × times of flight and computes total Δv for each
combination using Lambert's problem, producing contour plots that reveal
optimal launch windows.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pykep as pk

from .constants import GM_SUN, AU_TO_M, AU_TO_KM, J2000_JD, JD_EPOCH_OFFSET
from .bodies import ASTEROID_CATALOG
from .integrator import integrate_trajectory


def _make_pykep_planet(name):
    name_lower = name.strip().lower()
    for aid, data in ASTEROID_CATALOG.items():
        if name_lower == aid or name_lower == data.get("englishName", "").lower():
            a_au = data["a_au"]
            ecc = data["ecc"]
            inc_rad = np.radians(data["inc_deg"])
            omega_rad = np.radians(data["omega_deg"])
            w_rad = np.radians(data["w_deg"])
            M0_rad = np.radians(data["M0_deg"])
            epoch = pk.epoch(data["epoch_jd"] - J2000_JD)
            return pk.planet(pk.udpla.keplerian(
                epoch, [a_au * AU_TO_M, ecc, inc_rad, omega_rad, w_rad, M0_rad], GM_SUN
            ))
    return pk.planet(pk.udpla.jpl_lp(name))


def _date_to_epoch(dt):
    jd = dt.timestamp() / 86400.0 + JD_EPOCH_OFFSET
    return pk.epoch(jd - J2000_JD)


@dataclass
class PorkchopResult:
    launch_days: np.ndarray
    tof_days: np.ndarray
    grid: np.ndarray
    date_labels: list
    dep_body: str
    arr_body: str


@dataclass
class LambertTrajectory:
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    y: np.ndarray = field(default_factory=lambda: np.array([]))
    dep_burn: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    arr_burn: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0]))
    dv_dep: float = 0.0
    dv_arr: float = 0.0
    dv_total: float = 0.0
    tof_days: float = 0.0
    launch_date: datetime = None
    arrival_date: datetime = None
    dep_r: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    arr_r: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    dep_v: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    arr_v: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))


def compute_porkchop(
    dep_body,
    arr_body,
    start_date=None,
    num_dates=146,
    date_step=5,
    tof_min=50,
    tof_max=400,
    num_tofs=71,
):
    if start_date is None:
        start_date = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    dep = _make_pykep_planet(dep_body)
    arr = _make_pykep_planet(arr_body)

    launch_days = np.array([date_step * i for i in range(num_dates)])
    tof_days = np.linspace(tof_min, tof_max, num_tofs)
    grid = np.full((num_dates, num_tofs), np.nan)

    for i, d in enumerate(launch_days):
        t_launch = _date_to_epoch(start_date + timedelta(days=float(d)))
        r1, v1 = dep.eph(t_launch)

        for j, tof_d in enumerate(tof_days):
            tof_sec = tof_d * 86400.0
            t_arrive = pk.epoch(t_launch.mjd2000 + tof_d)
            r2, v2 = arr.eph(t_arrive)

            try:
                lp = pk.lambert_problem(list(r1), list(r2), tof_sec, GM_SUN)
                dv_dep = np.linalg.norm(np.array(lp.v0[0]) - np.array(v1))
                dv_arr = np.linalg.norm(np.array(lp.v1[0]) - np.array(v2))
                grid[i, j] = (dv_dep + dv_arr) / 1000.0
            except Exception:
                pass

    date_labels = [start_date + timedelta(days=float(d)) for d in launch_days]

    return PorkchopResult(
        launch_days=launch_days,
        tof_days=tof_days,
        grid=grid,
        date_labels=date_labels,
        dep_body=dep_body,
        arr_body=arr_body,
    )


def plot_porkchop(result, dv_budget=None, ax=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    grid = result.grid
    tof_days = result.tof_days
    num_dates = len(result.launch_days)
    date_labels = result.date_labels

    tick_positions = np.arange(0, num_dates, 24)
    tick_labels = [date_labels[int(k)].strftime("%Y-%m-%d") for k in tick_positions]

    valid = np.isfinite(grid) & (grid < 50)
    if not np.any(valid):
        ax.text(
            0.5, 0.5, "No feasible transfers found",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=16, color="gray",
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"{result.dep_body.title()} → {result.arr_body.title()} Porkchop Plot"
        )
        if fig is not None:
            plt.tight_layout()
        return fig

    levels = np.linspace(0, np.nanmax(grid[valid]), 30)
    cs = ax.contourf(
        np.arange(num_dates), tof_days, grid.T, levels=levels, cmap="RdYlGn_r"
    )
    plt.colorbar(cs, ax=ax, label="Δv (km/s)")

    min_idx = np.unravel_index(np.nanargmin(grid), grid.shape)
    ax.plot(
        min_idx[0],
        tof_days[min_idx[1]],
        "k*",
        markersize=15,
        label="Minimum Δv",
    )

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_xlabel("Launch date")
    ax.set_ylabel("Time of flight (days)")
    ax.set_title(
        f"{result.dep_body.title()} → {result.arr_body.title()} Porkchop Plot"
    )
    ax.legend()

    if fig is not None:
        plt.tight_layout()

    return fig


def plot_porkchop_budget(result, dv_budget, ax_region=None, ax_fastest=None):
    fig = None
    if ax_region is None or ax_fastest is None:
        fig, (ax_region, ax_fastest) = plt.subplots(1, 2, figsize=(16, 7))

    grid = result.grid
    tof_days = result.tof_days
    num_dates = len(result.launch_days)
    date_labels = result.date_labels
    tof_max = tof_days[-1]

    tick_positions = np.arange(0, num_dates, 24)
    tick_labels = [date_labels[int(k)].strftime("%Y-%m-%d") for k in tick_positions]

    valid = np.isfinite(grid) & (grid < 50)
    if not np.any(valid):
        for ax in (ax_region, ax_fastest):
            ax.text(
                0.5, 0.5, "No feasible transfers found",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=14, color="gray",
            )
            ax.set_xticks([])
            ax.set_yticks([])
        ax_region.set_title(f"Feasible region (Δv ≤ {dv_budget} km/s)")
        ax_fastest.set_title(
            f"Fastest arrival per launch date (Δv ≤ {dv_budget} km/s)"
        )
        if fig is not None:
            plt.tight_layout()
        return fig, np.full(num_dates, np.nan), np.full(num_dates, np.nan)

    levels = np.linspace(0, np.nanmax(grid[valid]), 30)

    ax_region.contourf(
        np.arange(num_dates), tof_days, grid.T, levels=levels, cmap="RdYlGn_r"
    )
    feasible = grid <= dv_budget
    feasible_T = feasible.T.astype(float)
    feasible_T[~feasible.T] = np.nan
    ax_region.contourf(
        np.arange(num_dates),
        tof_days,
        feasible_T,
        levels=[0.5, 1.5],
        colors=["none"],
        hatches=["///"],
        edgecolor="black",
    )
    ax_region.contour(
        np.arange(num_dates),
        tof_days,
        feasible.T.astype(float),
        levels=[0.5],
        colors="black",
        linewidths=1.5,
    )
    ax_region.set_xticks(tick_positions)
    ax_region.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax_region.set_xlabel("Launch date")
    ax_region.set_ylabel("Time of flight (days)")
    ax_region.set_title(f"Feasible region (Δv ≤ {dv_budget} km/s)")

    fastest_tof = np.full(num_dates, np.nan)
    fastest_dv = np.full(num_dates, np.nan)

    for i in range(num_dates):
        row = grid[i, :]
        feasible_mask = row <= dv_budget
        if np.any(feasible_mask):
            fastest_idx = np.where(feasible_mask)[0][0]
            fastest_tof[i] = tof_days[fastest_idx]
            fastest_dv[i] = row[fastest_idx]

    has_data = ~np.isnan(fastest_tof)
    ax_fastest.plot(
        np.arange(num_dates)[has_data],
        fastest_tof[has_data],
        "b-",
        linewidth=1.5,
        label="Min TOF within budget",
    )
    ax_fastest.fill_between(
        np.arange(num_dates)[has_data],
        0,
        fastest_tof[has_data],
        alpha=0.15,
        color="blue",
    )
    ax_fastest_r = ax_fastest.twinx()
    ax_fastest_r.plot(
        np.arange(num_dates)[has_data],
        fastest_dv[has_data],
        "r--",
        linewidth=1,
        alpha=0.7,
        label="Δv used",
    )
    ax_fastest_r.set_ylabel("Δv used (km/s)", color="red")
    ax_fastest_r.tick_params(axis="y", labelcolor="red")

    ax_fastest.set_xticks(tick_positions)
    ax_fastest.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax_fastest.set_xlabel("Launch date")
    ax_fastest.set_ylabel("Fastest time of flight (days)", color="blue")
    ax_fastest.tick_params(axis="y", labelcolor="blue")
    ax_fastest.set_title(
        f"Fastest arrival per launch date (Δv ≤ {dv_budget} km/s)"
    )
    ax_fastest.set_ylim(0, tof_max)

    lines1, labels1 = ax_fastest.get_legend_handles_labels()
    lines2, labels2 = ax_fastest_r.get_legend_handles_labels()
    ax_fastest.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    if fig is not None:
        plt.tight_layout()

    return fig, fastest_tof, fastest_dv


def summarize(result, dv_budget=None):
    grid = result.grid
    date_labels = result.date_labels
    tof_days = result.tof_days

    min_idx = np.unravel_index(np.nanargmin(grid), grid.shape)
    lines = [
        f"Porkchop: {result.dep_body.title()} → {result.arr_body.title()}",
        f"Minimum Δv: {grid[min_idx[0], min_idx[1]]:.2f} km/s",
        f"  Launch: {date_labels[min_idx[0]].strftime('%Y-%m-%d')}",
        f"  TOF: {tof_days[min_idx[1]]:.0f} days",
    ]

    if dv_budget is not None:
        fastest_tof = np.full(len(result.launch_days), np.nan)
        fastest_dv = np.full(len(result.launch_days), np.nan)
        for i in range(len(result.launch_days)):
            row = grid[i, :]
            feasible_mask = row <= dv_budget
            if np.any(feasible_mask):
                fastest_idx = np.where(feasible_mask)[0][0]
                fastest_tof[i] = tof_days[fastest_idx]
                fastest_dv[i] = row[fastest_idx]

        has_data = ~np.isnan(fastest_tof)
        if np.any(has_data):
            best = np.nanargmin(fastest_tof)
            lines.extend([
                f"\nFastest at Δv ≤ {dv_budget} km/s:",
                f"  TOF: {fastest_tof[best]:.0f} days",
                f"  Launch: {date_labels[best].strftime('%Y-%m-%d')}",
                f"  Δv used: {fastest_dv[best]:.2f} km/s",
            ])
        else:
            lines.append(f"\nNo feasible transfers at Δv ≤ {dv_budget} km/s")

    return "\n".join(lines)


def compute_lambert_trajectory(dep_body, arr_body, start_date, tof_days, points=500):
    dep = _make_pykep_planet(dep_body)
    arr = _make_pykep_planet(arr_body)

    t_launch = _date_to_epoch(start_date)
    r1, v1 = dep.eph(t_launch)

    t_arrive = pk.epoch(t_launch.mjd2000 + tof_days)
    r2, v2 = arr.eph(t_arrive)

    r1_m = np.array(r1)
    r2_m = np.array(r2)
    v1_ms = np.array(v1)
    v2_ms = np.array(v2)

    tof_sec = tof_days * 86400.0
    lp = pk.lambert_problem(list(r1_m), list(r2_m), tof_sec, GM_SUN)

    v_lambert_dep = np.array(lp.v0[0])
    v_lambert_arr = np.array(lp.v1[0])

    dv_dep = np.linalg.norm(v_lambert_dep - v1_ms) / 1000.0
    dv_arr = np.linalg.norm(v_lambert_arr - v2_ms) / 1000.0

    positions, _ = integrate_trajectory(r1_m, v_lambert_dep, tof_sec, points=points)
    positions_au = positions / AU_TO_M

    x_raw = positions_au[:, 0]
    y_raw = positions_au[:, 1]

    dep_burn_au = np.array([r1[0] / AU_TO_M, r1[1] / AU_TO_M])
    arr_burn_au = np.array([r2[0] / AU_TO_M, r2[1] / AU_TO_M])

    arrival_date = start_date + timedelta(days=tof_days)

    return LambertTrajectory(
        x=x_raw,
        y=y_raw,
        dep_burn=dep_burn_au,
        arr_burn=arr_burn_au,
        dv_dep=dv_dep,
        dv_arr=dv_arr,
        dv_total=dv_dep + dv_arr,
        tof_days=tof_days,
        launch_date=start_date,
        arrival_date=arrival_date,
        dep_r=r1_m,
        arr_r=r2_m,
        dep_v=v_lambert_dep,
        arr_v=v_lambert_arr,
    )


def get_feasible_windows(result, dv_budget=None, max_windows=500):
    grid = result.grid
    date_labels = result.date_labels
    tof_days = result.tof_days

    windows = []
    for i in range(len(result.launch_days)):
        for j in range(len(tof_days)):
            dv = grid[i, j]
            if np.isfinite(dv):
                if dv_budget is None or dv <= dv_budget:
                    windows.append((
                        date_labels[i],
                        tof_days[j],
                        dv,
                    ))

    windows.sort(key=lambda w: w[2])
    return windows[:max_windows]


def plot_lambert_trajectory(traject, dep_body_data, arr_body_data):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(0, 0, "yo", label="Sun", markersize=15)

    dep_r_au = np.sqrt(traject.dep_r[0]**2 + traject.dep_r[1]**2) / AU_TO_M
    dep_lon = np.arctan2(traject.dep_r[1], traject.dep_r[0])
    arr_r_au = np.sqrt(traject.arr_r[0]**2 + traject.arr_r[1]**2) / AU_TO_M
    arr_lon = np.arctan2(traject.arr_r[1], traject.arr_r[0])

    dep_sx = dep_r_au * np.cos(dep_lon)
    dep_sy = dep_r_au * np.sin(dep_lon)
    arr_sx = arr_r_au * np.cos(arr_lon)
    arr_sy = arr_r_au * np.sin(arr_lon)

    ax.plot(dep_sx, dep_sy, "bo", markersize=8,
            label=dep_body_data.get("englishName", "Departure"))
    ax.plot(arr_sx, arr_sy, "ro", markersize=8,
            label=arr_body_data.get("englishName", "Arrival"))

    from .transfers.visualization.core import (
        compute_orbit_rotation,
        plot_orbit,
    )

    dep_rotation = compute_orbit_rotation(dep_body_data, dep_lon, dep_r_au)
    arr_rotation = compute_orbit_rotation(arr_body_data, arr_lon, arr_r_au)

    plot_orbit(ax, dep_body_data, rotation=dep_rotation)
    plot_orbit(ax, arr_body_data, rotation=arr_rotation)

    from .transfers.visualization.core import plot_transfer
    plot_transfer(ax, traject.x, traject.y, traject.dep_burn, traject.arr_burn,
                  "Lambert Transfer", "green")

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("Distance (AU)", fontsize=11)
    ax.set_ylabel("Distance (AU)", fontsize=11)
    ax.set_title(
        f"Lambert Transfer: {dep_body_data.get('englishName', '?')} → "
        f"{arr_body_data.get('englishName', '?')}\n"
        f"Launch: {traject.launch_date.strftime('%Y-%m-%d')}  "
        f"TOF: {traject.tof_days:.0f} days",
        fontsize=13,
    )

    stats_text = (
        f"Δv departure:  {traject.dv_dep:.2f} km/s\n"
        f"Δv arrival:    {traject.dv_arr:.2f} km/s\n"
        f"Δv total:      {traject.dv_total:.2f} km/s\n"
        f"TOF:           {traject.tof_days:.0f} days\n"
        f"Arrival:       {traject.arrival_date.strftime('%Y-%m-%d')}"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.7)
    ax.text(
        0.02, 0.02, stats_text,
        transform=ax.transAxes, fontsize=11,
        verticalalignment="bottom", fontfamily="monospace",
        color="white", bbox=props,
    )

    plt.tight_layout()
    return fig
