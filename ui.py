"""Gradio web UI for the foslas orbital transfer calculator."""

import tempfile
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from foslas.bodies import load_planet_bodies, load_asteroid_body, ASTEROID_CATALOG
from foslas.constants import KM_TO_M
from foslas.transfers.base import OrbitalBody, transfer_time
from foslas.transfers.hohmann import hohmann_delta_v
from foslas.transfers.fast import find_factor_for_dv

BODIES = [
    "Mercury",
    "Venus",
    "Earth",
    "Mars",
    "Jupiter",
    "Saturn",
    "Uranus",
    "Neptune",
]
ASTEROID_NAMES = [data["englishName"] for data in ASTEROID_CATALOG.values()]
ALL_BODIES = BODIES + sorted(ASTEROID_NAMES)
BODY_IDS = [name.lower().replace(" ", "_") for name in BODIES]


def find_asteroid(body_id):
    body_id_lower = body_id.strip().lower()
    for aid, data in ASTEROID_CATALOG.items():
        if body_id_lower == aid:
            return load_asteroid_body(aid)
        if body_id_lower == data.get("englishName", "").lower():
            return load_asteroid_body(aid)
    return None


def find_body(bodies, name):
    name_lower = name.strip().lower()
    return next(
        (
            b
            for b in bodies
            if name_lower in {b["id"].lower(), b["englishName"].lower()}
        ),
        None,
    )


def _resolve_body_data(body_id):
    bodies = load_planet_bodies()
    bd = find_asteroid(body_id)
    if bd is None:
        bd = find_body(bodies, body_id)
    return bd


def _orbit_params(body, day_offset=0):
    from foslas.transfers.visualization import get_body_ecliptic, compute_orbit_rotation

    aph = body.get("aphelion", 0)
    peri = body.get("perihelion", 0)
    ecc = (aph - peri) / (aph + peri) if (aph + peri) > 0 else 0.0
    r_au, lon = get_body_ecliptic(body["englishName"], time_offset_days=day_offset)
    rotation = compute_orbit_rotation(body, lon, r_au)
    return ecc, rotation - lon


def compute_transfer(start_name, end_name, dv_km_s, day_offset):
    bodies = load_planet_bodies(day_offset)
    start = find_asteroid(start_name)
    if start is None:
        start = find_body(bodies, start_name)
    end = find_asteroid(end_name)
    if end is None:
        end = find_body(bodies, end_name)

    if not start:
        raise ValueError(f"Body '{start_name}' not found.")
    if not end:
        raise ValueError(f"Body '{end_name}' not found.")

    start_ob = OrbitalBody(start["aphelion"], start["perihelion"])
    end_ob = OrbitalBody(end["aphelion"], end["perihelion"])

    dv_dep, dv_arr, total_hohmann = hohmann_delta_v(start_ob.sma, end_ob.sma)
    hohmann_tof = transfer_time(start_ob.sma, end_ob.sma, 1.0)

    dep_ecc, dep_rot = _orbit_params(start, day_offset)
    arr_ecc, arr_rot = _orbit_params(end, day_offset)

    available_dv_m = dv_km_s * KM_TO_M

    fast_factor, fast_dv = find_factor_for_dv(
        start_ob.sma,
        end_ob.sma,
        available_dv_m,
        dep_ecc=dep_ecc,
        dep_rot=dep_rot,
        arr_ecc=arr_ecc,
        arr_rot=arr_rot,
    )
    fast_tof = transfer_time(start_ob.sma, end_ob.sma, fast_factor)

    from foslas.transfers.visualization import visualize

    stats = {
        "hohmann_dv": total_hohmann / 1000,
        "hohmann_time": hohmann_tof,
        "fast_dv": fast_dv / 1000,
        "fast_factor": fast_factor,
        "fast_time": fast_tof,
    }

    visualize(start_ob.sma, end_ob.sma, available_dv_m, [start, end], stats=stats)

    plot_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close("all")

    dep_day = day_offset
    dep_date_str = f"+{dep_day}d" if dep_day != 0 else "now"

    stats_text = (
        f"Transfer: {start['englishName']} -> {end['englishName']}  "
        f"(departure: {dep_date_str})\n"
        f"{'=' * 55}\n"
        f"HOHMANN TRANSFER (minimum energy)\n"
        f"{'-' * 55}\n"
        f"  Delta-V required:   {total_hohmann / 1000:.2f} km/s\n"
        f"    Departure burn:   {dv_dep / 1000:.2f} km/s\n"
        f"    Arrival burn:     {dv_arr / 1000:.2f} km/s\n"
        f"  Transfer time:      {hohmann_tof:.1f} days ({hohmann_tof / 365.25:.2f} years)\n"
        f"\n"
        f"FAST TRANSFER (maximise delta-V budget)\n"
        f"{'-' * 55}\n"
        f"  Delta-V used:       {fast_dv / 1000:.2f} km/s\n"
        f"  Energy factor:      {fast_factor:.2f}x Hohmann\n"
        f"  Transfer time:      {fast_tof:.1f} days ({fast_tof / 365.25:.2f} years)\n"
        f"  Time saved:         {hohmann_tof - fast_tof:.1f} days\n"
        f"{'=' * 55}"
    )

    return stats_text, plot_path


def run_transfer(start, end, dv, day):
    try:
        stats_text, plot_path = compute_transfer(start, end, dv, day)
        return stats_text, plot_path
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return f"Error occurred: {e}", None


def run_porkchop_generate(
    start, end, start_date, dv_budget, tof_min, tof_max, date_step
):
    from foslas.porkchop import (
        compute_porkchop,
        plot_porkchop,
        plot_porkchop_budget,
        summarize,
    )

    try:
        sd = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
        dv = float(dv_budget) if dv_budget else None

        result = compute_porkchop(
            start,
            end,
            start_date=sd,
            date_step=int(date_step),
            tof_min=int(tof_min),
            tof_max=int(tof_max),
        )

        text = summarize(result, dv_budget=dv)

        if dv is not None:
            fig, _, _ = plot_porkchop_budget(result, dv)
        else:
            fig = plot_porkchop(result)

        plot_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close("all")

        return text, plot_path, result
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return f"Error occurred: {e}", None, None


def _pick_window(result, criterion, dv_budget=None):
    grid = result.grid
    date_labels = result.date_labels
    tof_days = result.tof_days

    best_dv = np.inf
    best_i, best_j = 0, 0
    found = False

    for i in range(len(result.launch_days)):
        for j in range(len(tof_days)):
            dv = grid[i, j]
            if not np.isfinite(dv):
                continue
            if dv_budget is not None and dv > dv_budget:
                continue

            if criterion == "Nearest launch date":
                if not found or date_labels[i] < date_labels[best_i]:
                    best_i, best_j = i, j
                    found = True
            elif criterion == "Shortest transfer time":
                if not found or tof_days[j] < tof_days[best_j]:
                    best_i, best_j = i, j
                    found = True
            elif criterion == "Lowest Δv":
                if not found or dv < best_dv:
                    best_i, best_j = i, j
                    best_dv = dv
                    found = True

    if not found:
        return None, None
    return date_labels[best_i], tof_days[best_j]


def run_porkchop_transfer(start, end, criterion, porkchop_state, start_date, dv_budget):
    from foslas.porkchop import (
        compute_lambert_trajectory,
        plot_lambert_trajectory,
    )

    if porkchop_state is None:
        return "Generate a porkchop plot first.", None

    try:
        dv = float(dv_budget) if dv_budget else None
        launch_date, tof_days = _pick_window(porkchop_state, criterion, dv_budget=dv)
        if launch_date is None:
            return "No feasible transfer found for this criterion.", None

        traj = compute_lambert_trajectory(start, end, launch_date, tof_days)

        dep_data = _resolve_body_data(start)
        arr_data = _resolve_body_data(end)

        if dep_data is None or arr_data is None:
            return "Could not resolve body data for plotting.", None

        fig = plot_lambert_trajectory(traj, dep_data, arr_data)

        plot_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close("all")

        stats_text = (
            f"Lambert Transfer: {start.title()} → {end.title()}\n"
            f"{'=' * 45}\n"
            f"  Launch date:   {traj.launch_date.strftime('%Y-%m-%d')}\n"
            f"  Arrival date:  {traj.arrival_date.strftime('%Y-%m-%d')}\n"
            f"  TOF:           {traj.tof_days:.0f} days\n"
            f"{'-' * 45}\n"
            f"  Δv departure:  {traj.dv_dep:.2f} km/s\n"
            f"  Δv arrival:    {traj.dv_arr:.2f} km/s\n"
            f"  Δv total:      {traj.dv_total:.2f} km/s\n"
            f"{'=' * 45}"
        )

        return stats_text, plot_path
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return f"Error occurred: {e}", None


import gradio as gr

with gr.Blocks(title="foslas - Orbital Transfer Calculator") as demo:
    gr.Markdown("# foslas — Orbital Transfer Calculator")

    with gr.Row():
        with gr.Column(scale=1):
            start_input = gr.Dropdown(
                choices=ALL_BODIES, value="Earth", label="Departure Body"
            )
            end_input = gr.Dropdown(choices=ALL_BODIES, value="Mars", label="Arrival Body")
            start_date_input = gr.Textbox(
                label="Start Date (YYYY-MM-DD)",
                value="",
                placeholder="YYYY-MM-DD or empty for today",
            )
            dv_budget_input = gr.Textbox(
                label="Δv Budget (km/s, optional)",
                value="",
                placeholder="e.g. 10 — enables fastest-transfer view",
            )
            tof_min_input = gr.Slider(
                minimum=10,
                maximum=200,
                value=50,
                step=10,
                label="Min TOF (days)",
            )
            tof_max_input = gr.Slider(
                minimum=100,
                maximum=600,
                value=400,
                step=10,
                label="Max TOF (days)",
            )
            date_step_input = gr.Slider(
                minimum=1,
                maximum=20,
                value=5,
                step=1,
                label="Launch date step (days)",
            )
            generate_btn = gr.Button("Generate Porkchop", variant="primary")

        with gr.Column(scale=1):
            porkchop_stats = gr.Textbox(label="Porkchop Summary", lines=8)
            porkchop_plot = gr.Image(label="Porkchop Plot", type="filepath")

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            criterion_radio = gr.Radio(
                choices=["Nearest launch date", "Shortest transfer time", "Lowest Δv"],
                value="Lowest Δv",
                label="Optimize Transfer For",
            )
            plot_transfer_btn = gr.Button("Plot Transfer", variant="primary")

        with gr.Column(scale=1):
            transfer_stats = gr.Textbox(label="Transfer Statistics", lines=8)
            transfer_plot = gr.Image(label="Transfer Trajectory", type="filepath")

    porkchop_state = gr.State(value=None)

    generate_btn.click(
        fn=run_porkchop_generate,
        inputs=[
            start_input,
            end_input,
            start_date_input,
            dv_budget_input,
            tof_min_input,
            tof_max_input,
            date_step_input,
        ],
        outputs=[porkchop_stats, porkchop_plot, porkchop_state],
    )

    plot_transfer_btn.click(
        fn=run_porkchop_transfer,
        inputs=[
            start_input,
            end_input,
            criterion_radio,
            porkchop_state,
            start_date_input,
            dv_budget_input,
        ],
        outputs=[transfer_stats, transfer_plot],
    )

if __name__ == "__main__":
    demo.launch()
