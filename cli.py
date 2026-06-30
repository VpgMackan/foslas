"""Command-line interface for the foslas orbital transfer calculator.

Provides subcommands for computing transfer statistics, generating
trajectory plots, and listing available celestial bodies.
"""

import argparse
import sys

import numpy as np

from foslas.constants import KM_TO_M
from foslas.transfers.base import OrbitalBody
from foslas.transfers.hohmann import hohmann_delta_v
from foslas.transfers.fast import find_factor_for_dv
from foslas.transfers.base import transfer_time
from foslas.utils import resolve_bodies, find_body, find_asteroid, orbit_params, compute_eccentricity


def get_ap_per(body):
    """Extract apogee and perigee from body data.

    Parameters
    ----------
    body : dict
        Body data dictionary.

    Returns
    -------
    tuple of float
        (apogee, perigee) in kilometers.
    """
    if body.get("apogee", 0) > 0 and body.get("perigee", 0) > 0:
        return body["apogee"], body["perigee"]
    sma = body.get("semimajorAxis", 0)
    return sma, sma


def make_orbital_body(body):
    """Create an OrbitalBody from body data.

    Parameters
    ----------
    body : dict
        Body data dictionary with apogee/perigee or semimajorAxis.

    Returns
    -------
    OrbitalBody
        The orbital body object.
    """
    a, p = get_ap_per(body)
    return OrbitalBody(a, p)


def cmd_stats(args):
    """Handle the 'stats' subcommand.

    Computes and prints Hohmann transfer delta-V, transfer time, and
    optionally a fast transfer profile using the delta-V budget.
    """
    start, end, start_ob, end_ob = resolve_bodies(args.start, args.end)

    dv_dep, dv_arr, total_hohmann = hohmann_delta_v(start_ob.sma, end_ob.sma)
    hohmann_tof = transfer_time(start_ob.sma, end_ob.sma, 1.0)

    dep_ecc, dep_rot = orbit_params(start)
    arr_ecc, arr_rot = orbit_params(end)

    dep_ecc_val = compute_eccentricity(
        start.get("aphelion", 0), start.get("perihelion", 0)
    )
    arr_ecc_val = compute_eccentricity(
        end.get("aphelion", 0), end.get("perihelion", 0)
    )
    ecc_warning = dep_ecc_val > 0.05 or arr_ecc_val > 0.05

    available_dv_m = args.dv * KM_TO_M if args.dv else None

    print(f"\nTransfer: {start['englishName']} -> {end['englishName']}")
    print("=" * 55)
    print("  HOHMANN TRANSFER (minimum energy)")
    print("-" * 55)
    print(f"  Delta-V required:  {total_hohmann / 1000:.2f} km/s")
    print(f"    Departure burn:  {dv_dep / 1000:.2f} km/s")
    print(f"    Arrival burn:    {dv_arr / 1000:.2f} km/s")
    print(
        f"  Transfer time:     {hohmann_tof:.1f} days ({hohmann_tof / 365.25:.2f} years)"
    )
    if ecc_warning:
        print()
        print("  NOTE: Hohmann delta-V above is a circularized reference.")
        print(
            f"  {start['englishName']} eccentricity: {dep_ecc_val:.4f}, "
            f"{end['englishName']} eccentricity: {arr_ecc_val:.4f}"
        )
        print("  Actual delta-V depends on the bodies' positions in their orbits.")

    if available_dv_m is not None:
        if available_dv_m < total_hohmann - 1:
            print(f"\n  WARNING: insufficient delta-V for this transfer.")
            print(f"  Need {total_hohmann / 1000:.2f} km/s, have {args.dv:.2f} km/s")
        else:
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
            print()
            print("  FAST TRANSFER (maximise delta-V budget)")
            print("-" * 55)
            print(f"  Delta-V used:      {fast_dv / 1000:.2f} km/s")
            print(f"  Energy factor:     {fast_factor:.2f}x Hohmann")
            print(
                f"  Transfer time:     {fast_tof:.1f} days ({fast_tof / 365.25:.2f} years)"
            )
            print(f"  Time saved:        {hohmann_tof - fast_tof:.1f} days")

    print("=" * 55)


def cmd_plot(args):
    """Handle the 'plot' subcommand.

    Generates and saves a PNG plot of the transfer trajectory.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from foslas.transfers.visualization import visualize

    start, end, start_ob, end_ob = resolve_bodies(args.start, args.end)

    dep_ecc, dep_rot = orbit_params(start)
    arr_ecc, arr_rot = orbit_params(end)

    available_dv_m = args.dv * KM_TO_M if args.dv else 30.0 * KM_TO_M
    _, _, total_hohmann = hohmann_delta_v(start_ob.sma, end_ob.sma)
    fast_factor, fast_dv = find_factor_for_dv(
        start_ob.sma,
        end_ob.sma,
        available_dv_m,
        dep_ecc=dep_ecc,
        dep_rot=dep_rot,
        arr_ecc=arr_ecc,
        arr_rot=arr_rot,
    )

    stats = {
        "hohmann_dv": total_hohmann / 1000,
        "hohmann_time": transfer_time(start_ob.sma, end_ob.sma, 1.0),
        "fast_dv": fast_dv / 1000,
        "fast_factor": fast_factor,
        "fast_time": transfer_time(start_ob.sma, end_ob.sma, fast_factor),
    }

    visualize(start_ob.sma, end_ob.sma, available_dv_m, [start, end], stats=stats)

    output = args.output
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output}")


def cmd_porkchop(args):
    """Handle the 'porkchop' subcommand.

    Generates a porkchop plot sweeping launch dates × times of flight for a given
    departure/arrival pair, with optional Δv budget analysis.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from datetime import datetime
    from foslas.porkchop import (
        compute_porkchop,
        plot_porkchop,
        plot_porkchop_budget,
        summarize,
    )

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d") if args.start_date else None

    print(f"Computing porkchop: {args.start} → {args.end} ...")
    result = compute_porkchop(
        args.start,
        args.end,
        start_date=start_date,
        num_dates=args.num_dates,
        date_step=args.date_step,
        tof_min=args.tof_min,
        tof_max=args.tof_max,
        num_tofs=args.num_tofs,
    )

    print(summarize(result, dv_budget=args.dv))

    if args.dv is not None:
        fig, _, _ = plot_porkchop_budget(result, args.dv)
    else:
        fig = plot_porkchop(result)

    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close("all")
    print(f"\nPlot saved to {args.output}")


def cmd_list(args):
    """Handle the 'list' subcommand.

    Lists available celestial bodies, optionally filtered by search query.
    """
    from foslas.bodies import load_planet_bodies

    bodies = load_planet_bodies()
    planets = [
        b
        for b in bodies
        if b.get("semimajorAxis", 0) > 1e6 and b.get("aphelion", 0) > 0
    ]
    planets.sort(key=lambda b: b.get("semimajorAxis", 0))

    if args.search:
        query = args.search.lower()
        planets = [b for b in bodies if query in b["englishName"].lower()]

    print(f"\n{'ID':<25} {'englishName':<30} {'Semi-major axis (km)':>22}")
    print("-" * 80)
    for b in planets:
        sma = b.get("semimajorAxis", 0)
        print(f"{b['id']:<25} {b.get('englishName', 'N/A'):<30} {sma:>22,.0f}")
    print(f"\n{len(planets)} body(s) found.")


def main():
    """Entry point for the CLI.

    Sets up argument parsing and dispatches to the appropriate subcommand.
    """
    parser = argparse.ArgumentParser(
        prog="foslas",
        description="Orbital Transfer Trajectory Calculator",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_stats = sub.add_parser("stats", help="Calculate transfer delta-V and time")
    p_stats.add_argument("start", help="Departure body ID (e.g. terre, mars)")
    p_stats.add_argument("end", help="Arrival body ID (e.g. jupiter, saturne)")
    p_stats.add_argument(
        "-d", "--dv", type=float, default=None, help="Available delta-V budget in km/s"
    )
    p_stats.set_defaults(func=cmd_stats)

    p_plot = sub.add_parser("plot", help="Generate transfer trajectory plot")
    p_plot.add_argument("start", help="Departure body ID")
    p_plot.add_argument("end", help="Arrival body ID")
    p_plot.add_argument(
        "-d",
        "--dv",
        type=float,
        default=None,
        help="Available delta-V budget in km/s (default: 30)",
    )
    p_plot.add_argument(
        "-o",
        "--output",
        default="transfer.png",
        help="Output file (default: transfer.png)",
    )
    p_plot.set_defaults(func=cmd_plot)

    p_porkchop = sub.add_parser("porkchop", help="Generate porkchop launch window plot")
    p_porkchop.add_argument("start", help="Departure body ID (e.g. earth, mars)")
    p_porkchop.add_argument("end", help="Arrival body ID (e.g. mars, jupiter)")
    p_porkchop.add_argument(
        "-s", "--start-date", default=None,
        help="First launch date, YYYY-MM-DD (default: today)",
    )
    p_porkchop.add_argument(
        "-d", "--dv", type=float, default=None,
        help="Δv budget in km/s; if set, show fastest-transfer analysis",
    )
    p_porkchop.add_argument(
        "-o", "--output", default="porkchop.png",
        help="Output file (default: porkchop.png)",
    )
    p_porkchop.add_argument("--num-dates", type=int, default=146, help="Number of launch dates to sweep")
    p_porkchop.add_argument("--date-step", type=int, default=5, help="Days between launch dates")
    p_porkchop.add_argument("--tof-min", type=int, default=50, help="Minimum TOF in days")
    p_porkchop.add_argument("--tof-max", type=int, default=400, help="Maximum TOF in days")
    p_porkchop.add_argument("--num-tofs", type=int, default=71, help="Number of TOF samples")
    p_porkchop.set_defaults(func=cmd_porkchop)

    p_list = sub.add_parser("list", help="List available celestial bodies")
    p_list.add_argument(
        "-s", "--search", default=None, help="Filter bodies by name or ID"
    )
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()