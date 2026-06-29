"""Gradio web UI for the foslas orbital transfer calculator."""

import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from foslas.bodies import load_planet_bodies, load_asteroid_body, ASTEROID_CATALOG
from foslas.constants import KM_TO_M
from foslas.transfers.base import OrbitalBody, transfer_time
from foslas.transfers.hohmann import hohmann_delta_v
from foslas.transfers.fast import find_factor_for_dv

BODIES = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
BODY_IDS = [name.lower() for name in BODIES]


def find_asteroid(body_id):
    """Find an asteroid by ID or englishName in the asteroid catalog."""
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
        # This will print the actual technical error to your console
        print(f"CRITICAL ERROR: {e}")
        return f"Error occurred: {e}"


import gradio as gr

with gr.Blocks(title="foslas - Orbital Transfer Calculator") as demo:
    gr.Markdown("# foslas — Orbital Transfer Calculator")

    with gr.Row():
        with gr.Column(scale=1):
            start_input = gr.Dropdown(
                choices=BODIES, value="Earth", label="Departure Body"
            )
            end_input = gr.Dropdown(choices=BODIES, value="Mars", label="Arrival Body")
            dv_input = gr.Slider(
                minimum=1.0,
                maximum=200.0,
                value=30.0,
                step=0.5,
                label="Delta-V Budget (km/s)",
            )
            day_input = gr.Slider(
                minimum=0, maximum=1825, value=0, step=1, label="Day Offset from Now"
            )
            submit_btn = gr.Button("Calculate", variant="primary")

        with gr.Column(scale=1):
            stats_output = gr.Textbox(label="Transfer Statistics", lines=14)
            plot_output = gr.Image(label="Trajectory Plot", type="filepath")

    submit_btn.click(
        fn=run_transfer,
        inputs=[start_input, end_input, dv_input, day_input],
        outputs=[stats_output, plot_output],
    )

if __name__ == "__main__":
    demo.launch(share=True)
