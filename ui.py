"""Gradio web UI for the foslas orbital transfer calculator."""

import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from foslas.constants import KM_TO_M
from foslas.transfers.base import OrbitalBody, transfer_time
from foslas.transfers.hohmann import hohmann_delta_v
from foslas.transfers.fast import find_factor_for_dv

BODIES = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
BODY_IDS = [name.lower() for name in BODIES]

_BODY_SPECS = [
    ("mercury", "mercury", "Mercury"),
    ("venus", "venus", "Venus"),
    ("earth", "earth", "Earth"),
    ("mars", "mars", "Mars"),
    ("jupiter", "jupiter", "Jupiter"),
    ("saturn", "saturn", "Saturn"),
    ("uranus", "uranus", "Uranus"),
    ("neptune", "neptune", "Neptune"),
]


def load_bodies(day_offset=0):
    try:
        import astropy.units as u
        from astropy.constants import G, M_sun
        from astropy.coordinates import get_body_barycentric_posvel
        from astropy.time import Time
    except ModuleNotFoundError:
        raise RuntimeError("astropy is required. Install with: pip install astropy")

    def _elements_from_state(r_m, v_m_s, mu):
        import numpy as np

        r = np.linalg.norm(r_m)
        v = np.linalg.norm(v_m_s)
        h = np.cross(r_m, v_m_s)
        e_vec = np.cross(v_m_s, h) / mu - (r_m / r)
        ecc = float(np.linalg.norm(e_vec))
        eps = v * v / 2.0 - mu / r
        if eps >= 0:
            return None
        sma = -mu / (2.0 * eps)
        return float(sma), ecc

    epoch = Time.now() + day_offset
    sun_pos, sun_vel = get_body_barycentric_posvel("sun", epoch)
    sun_pos = sun_pos.xyz.to(u.m).value
    sun_vel = sun_vel.xyz.to(u.m / u.s).value
    mu_sun = (G * M_sun).to_value(u.m**3 / u.s**2)

    bodies = []
    for body_id, astropy_name, english_name in _BODY_SPECS:
        try:
            body_pos, body_vel = get_body_barycentric_posvel(astropy_name, epoch)
            r_vec = body_pos.xyz.to(u.m).value - sun_pos
            v_vec = body_vel.xyz.to(u.m / u.s).value - sun_vel
            elements = _elements_from_state(r_vec, v_vec, mu_sun)
            if elements is None:
                continue
            sma_m, ecc = elements
            sma_km = sma_m / 1000.0
            aphelion = sma_km * (1.0 + ecc)
            perihelion = sma_km * (1.0 - ecc)
            bodies.append(
                {
                    "id": body_id,
                    "englishName": english_name,
                    "semimajorAxis": sma_km,
                    "eccentricity": ecc,
                    "aphelion": aphelion,
                    "perihelion": perihelion,
                }
            )
        except Exception:
            continue

    if not bodies:
        raise RuntimeError("Failed to load bodies from Astropy ephemerides.")
    return bodies


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
    r_au, lon = get_body_ecliptic(body["englishName"], time_offset_days=day_offset)
    rotation = compute_orbit_rotation(body, lon, r_au)
    return rotation - lon


def compute_transfer(start_name, end_name, dv_km_s, day_offset):
    bodies = load_bodies(day_offset)
    start = find_body(bodies, start_name)
    end = find_body(bodies, end_name)

    if not start:
        raise ValueError(f"Body '{start_name}' not found.")
    if not end:
        raise ValueError(f"Body '{end_name}' not found.")

    start_ob = OrbitalBody(start["aphelion"], start["perihelion"])
    end_ob = OrbitalBody(end["aphelion"], end["perihelion"])

    dv_dep, dv_arr, total_hohmann = hohmann_delta_v(start_ob.sma, end_ob.sma)
    hohmann_tof = transfer_time(start_ob.sma, end_ob.sma, 1.0)

    dep_rot = _orbit_params(start, day_offset)
    arr_rot = _orbit_params(end, day_offset)

    available_dv_m = dv_km_s * KM_TO_M

    fast_factor, fast_dv = find_factor_for_dv(
        start_ob.sma,
        end_ob.sma,
        available_dv_m,
        dep_ecc=0.0,
        dep_rot=dep_rot,
        arr_ecc=0.0,
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
    demo.launch()
