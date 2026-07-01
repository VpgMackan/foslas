"""Gradio web UI for the foslas orbital transfer calculator."""

import tempfile
from datetime import datetime

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from foslas.constants import KM_TO_M
from foslas.bodies import load_planet_bodies
from foslas.transfers.base import OrbitalBody
from foslas.transfers.hohmann import hohmann_delta_v
from foslas.transfers.fast import find_factor_for_dv
from foslas.utils import (
    resolve_body_data,
    orbit_params,
    ASTEROID_CATALOG,
)

STAT_BOX_STYLE = (
    "background:#1f2937;border:1px solid #374151;border-radius:8px;"
    "padding:12px 16px;text-align:center;min-width:120px;"
)
STAT_LABEL_STYLE = "font-size:0.8em;color:#9ca3af;margin:0 0 4px 0;"
STAT_VALUE_STYLE = "font-size:1.4em;font-weight:600;color:#f9fafb;margin:0;"
STAT_SUB_STYLE = "font-size:0.75em;color:#6b7280;margin:2px 0 0 0;"


def _stat_box(label, value, sub=None):
    parts = [
        f'<div style="{STAT_BOX_STYLE}">',
        f'<p style="{STAT_LABEL_STYLE}">{label}</p>',
        f'<p style="{STAT_VALUE_STYLE}">{value}</p>',
    ]
    if sub:
        parts.append(f'<p style="{STAT_SUB_STYLE}">{sub}</p>')
    parts.append("</div>")
    return "\n".join(parts)


def _stats_grid(*boxes):
    inner = "".join(boxes)
    return (
        f'<div style="display:flex;flex-wrap:wrap;gap:12px;margin-bottom:12px;">'
        f"{inner}</div>"
    )


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


def compute_transfer(start_name, end_name, dv_km_s, day_offset):
    from foslas.transfers.visualization import visualize

    bodies = load_planet_bodies(day_offset)
    start = resolve_body_data(start_name, bodies)
    end = resolve_body_data(end_name, bodies)

    if not start:
        raise ValueError(f"Body '{start_name}' not found.")
    if not end:
        raise ValueError(f"Body '{end_name}' not found.")

    start_ob = OrbitalBody(start.aphelion_km, start.perihelion_km)
    end_ob = OrbitalBody(end.aphelion_km, end.perihelion_km)

    dv_dep, dv_arr, total_hohmann = hohmann_delta_v(start_ob.sma, end_ob.sma)
    hohmann_tof = start_ob.transfer_time_to(end_ob, factor=1.0)

    dep_ecc, dep_rot = orbit_params(start, day_offset)
    arr_ecc, arr_rot = orbit_params(end, day_offset)

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
    fast_tof = start_ob.transfer_time_to(end_ob, factor=fast_factor)

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

    hohmann_row = _stats_grid(
        _stat_box("Δv Total", f"{total_hohmann / 1000:.2f} km/s"),
        _stat_box("Departure Burn", f"{dv_dep / 1000:.2f} km/s"),
        _stat_box("Arrival Burn", f"{dv_arr / 1000:.2f} km/s"),
        _stat_box(
            "Transfer Time",
            f"{hohmann_tof:.1f} days",
            f"{hohmann_tof / 365.25:.2f} years",
        ),
    )
    fast_row = _stats_grid(
        _stat_box("Δv Used", f"{fast_dv / 1000:.2f} km/s"),
        _stat_box("Energy Factor", f"{fast_factor:.2f}x"),
        _stat_box(
            "Transfer Time", f"{fast_tof:.1f} days", f"{fast_tof / 365.25:.2f} years"
        ),
        _stat_box("Time Saved", f"{hohmann_tof - fast_tof:.1f} days"),
    )
    stats_text = (
        f'<p style="font-weight:600;color:#e5e7eb;margin:0 0 4px 0;">'
        f'Transfer: {start.english_name} → {end.english_name}'
        f" <span style='font-weight:400;color:#9ca3af;'>(departure: {dep_date_str})</span></p>"
        f'<p style="font-size:0.85em;color:#9ca3af;margin:0 0 8px 0;">Hohmann Transfer (minimum energy)</p>'
        f"{hohmann_row}"
        f'<p style="font-size:0.85em;color:#9ca3af;margin:0 0 8px 0;">Fast Transfer (maximise Δv budget)</p>'
        f"{fast_row}"
    )

    return stats_text, plot_path


def run_transfer(start, end, dv, day):
    try:
        stats_text, plot_path = compute_transfer(start, end, dv, day)
        return stats_text, plot_path
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return f'<p style="color:#ef4444;">Error occurred: {e}</p>', None


def run_porkchop_generate(
    start, end, start_date, dv_budget, tof_min, tof_max, date_step, criterion
):
    from foslas.porkchop import (
        compute_porkchop,
        plot_porkchop,
        plot_porkchop_budget,
        compute_lambert_trajectory,
        plot_lambert_trajectory,
    )

    bodies = load_planet_bodies()

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

        grid = result.grid
        date_labels = result.date_labels
        tof_days = result.tof_days
        min_idx = np.unravel_index(np.nanargmin(grid), grid.shape)

        min_row = _stats_grid(
            _stat_box("Minimum Δv", f"{grid[min_idx[0], min_idx[1]]:.2f} km/s"),
            _stat_box("Launch Date", date_labels[min_idx[0]].strftime("%Y-%m-%d")),
            _stat_box("TOF", f"{tof_days[min_idx[1]]:.0f} days"),
        )
        porkchop_html = (
            f'<p style="font-weight:600;color:#e5e7eb;margin:0 0 8px 0;">'
            f"Porkchop: {result.dep_body.title()} → {result.arr_body.title()}</p>"
            f'<p style="font-size:0.85em;color:#9ca3af;margin:0 0 8px 0;">Minimum Δv Solution</p>'
            f"{min_row}"
        )

        if dv is not None:
            feasible = grid <= dv
            tof_grid = np.broadcast_to(tof_days, grid.shape)
            masked_tof = np.where(feasible, tof_grid, np.inf)
            fastest_tof = np.nanmin(masked_tof, axis=1)
            row_indices = np.nanargmin(masked_tof, axis=1)
            has_data = np.isfinite(fastest_tof)

            if np.any(has_data):
                best = np.nanargmin(fastest_tof)
                fast_row = _stats_grid(
                    _stat_box("Δv Budget", f"≤ {dv:.0f} km/s"),
                    _stat_box("Fastest TOF", f"{fastest_tof[best]:.0f} days"),
                    _stat_box("Launch Date", date_labels[best].strftime("%Y-%m-%d")),
                    _stat_box("Δv Used", f"{grid[best, row_indices[best]]:.2f} km/s"),
                )
                porkchop_html += (
                    f'<p style="font-size:0.85em;color:#9ca3af;margin:12px 0 8px 0;">'
                    f"Fastest at Δv ≤ {dv:.0f} km/s</p>"
                    f"{fast_row}"
                )
            else:
                porkchop_html += (
                    f'<p style="color:#f59e0b;margin:12px 0 0 0;">'
                    f"No feasible transfers at Δv ≤ {dv:.0f} km/s</p>"
                )

        if dv is not None:
            fig, _, _ = plot_porkchop_budget(result, dv)
        else:
            fig = plot_porkchop(result)

        porkchop_plot_path = tempfile.NamedTemporaryFile(
            suffix=".png", delete=False
        ).name
        fig.savefig(porkchop_plot_path, dpi=150, bbox_inches="tight")
        plt.close("all")

        launch_date, tof_days_selected = _pick_window(result, criterion, dv_budget=dv)
        if launch_date is None:
            transfer_stats = '<p style="color:#f59e0b;">No feasible transfer found for this criterion.</p>'
            transfer_plot = None
        else:
            traj = compute_lambert_trajectory(
                start, end, launch_date, tof_days_selected
            )

            dep_data = resolve_body_data(start, bodies)
            arr_data = resolve_body_data(end, bodies)

            if dep_data is None or arr_data is None:
                transfer_stats = '<p style="color:#ef4444;">Could not resolve body data for plotting.</p>'
                transfer_plot = None
            else:
                fig = plot_lambert_trajectory(traj, dep_data, arr_data)

                transfer_plot = tempfile.NamedTemporaryFile(
                    suffix=".png", delete=False
                ).name
                fig.savefig(transfer_plot, dpi=150, bbox_inches="tight")
                plt.close("all")

                date_row = _stats_grid(
                    _stat_box("Launch Date", traj.launch_date.strftime("%Y-%m-%d")),
                    _stat_box("Arrival Date", traj.arrival_date.strftime("%Y-%m-%d")),
                    _stat_box("Transfer Time", f"{traj.tof_days:.0f} days"),
                )
                dv_row = _stats_grid(
                    _stat_box("Δv Departure", f"{traj.dv_dep:.2f} km/s"),
                    _stat_box("Δv Arrival", f"{traj.dv_arr:.2f} km/s"),
                    _stat_box("Δv Total", f"{traj.dv_total:.2f} km/s"),
                )
                transfer_stats = (
                    f'<p style="font-weight:600;color:#e5e7eb;margin:0 0 8px 0;">'
                    f"Pincer Transfer: {start.title()} → {end.title()}</p>"
                    f"{date_row}"
                    f"{dv_row}"
                )

        return porkchop_html, porkchop_plot_path, transfer_stats, transfer_plot
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return (
            f'<p style="color:#ef4444;">Error occurred: {e}</p>',
            None,
            f'<p style="color:#ef4444;">Error occurred: {e}</p>',
            None,
        )


def _pick_window(result, criterion, dv_budget=None):
    grid = result.grid
    date_labels = result.date_labels
    tof_days = result.tof_days

    feasible = np.isfinite(grid)
    if dv_budget is not None:
        feasible &= grid <= dv_budget

    if not np.any(feasible):
        return None, None

    masked_grid = np.where(feasible, grid, np.inf)

    if criterion == "Nearest launch date":
        launch_idx = np.arange(len(result.launch_days))
        launch_2d = np.broadcast_to(launch_idx[:, None], grid.shape)
        score = np.where(feasible, launch_2d.astype(float), np.inf)
        best = np.unravel_index(np.argmin(score), grid.shape)
    elif criterion == "Shortest transfer time":
        tof_2d = np.broadcast_to(tof_days, grid.shape)
        score = np.where(feasible, tof_2d, np.inf)
        best = np.unravel_index(np.argmin(score), grid.shape)
    else:
        best = np.unravel_index(np.argmin(masked_grid), grid.shape)

    return date_labels[best[0]], tof_days[best[1]]


import gradio as gr

with gr.Blocks(title="foslas - Orbital Transfer Calculator") as demo:
    gr.Markdown("# foslas — Orbital Transfer Calculator")

    with gr.Row():
        with gr.Column(scale=1):
            start_input = gr.Dropdown(
                choices=ALL_BODIES, value="Earth", label="Departure Body"
            )
            end_input = gr.Dropdown(
                choices=ALL_BODIES, value="Mars", label="Arrival Body"
            )
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
            criterion_radio = gr.Radio(
                choices=["Nearest launch date", "Shortest transfer time", "Lowest Δv"],
                value="Lowest Δv",
                label="Optimize Transfer For",
            )
            generate_btn = gr.Button(
                "Generate Porkchop & Plot Transfer", variant="primary"
            )

        with gr.Column(scale=1):
            porkchop_stats = gr.HTML(label="Porkchop Summary")
            porkchop_plot = gr.Image(label="Porkchop Plot", type="filepath")

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            transfer_stats = gr.HTML(label="Transfer Statistics")
            transfer_plot = gr.Image(label="Transfer Trajectory", type="filepath")

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
            criterion_radio,
        ],
        outputs=[porkchop_stats, porkchop_plot, transfer_stats, transfer_plot],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
