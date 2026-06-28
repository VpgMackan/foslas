"""Command-line interface for the foslas orbital transfer calculator.

Provides subcommands for computing transfer statistics, generating
trajectory plots, and listing available celestial bodies.
"""

import argparse
import json
import sys

from foslas.constants import KM_TO_M
from foslas.orbital import (
    OrbitalBody,
    hohmann_delta_v,
    find_factor_for_dv,
    transfer_time,
)

DATA_PATH = "data/data.json"


def load_bodies():
    """Load celestial body data from the JSON file.

    Returns
    -------
    list of dict
        List of body data dictionaries.
    """
    with open(DATA_PATH) as f:
        return json.load(f)["bodies"]


def find_body(bodies, body_id):
    """Find a body by its ID (case-insensitive).

    Parameters
    ----------
    bodies : list of dict
        List of body data dictionaries.
    body_id : str
        The body ID to search for.

    Returns
    -------
    dict or None
        The body data dictionary, or None if not found.
    """
    body_id = body_id.strip().lower()
    return next((b for b in bodies if b["englishName"].lower() == body_id), None)


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


def resolve_bodies(start_id, end_id):
    """Resolve body IDs to OrbitalBody objects.

    Parameters
    ----------
    start_id : str
        Departure body ID.
    end_id : str
        Arrival body ID.

    Returns
    -------
    tuple
        (start_data, end_data, start_ob, end_ob) where data are dicts
        and ob are OrbitalBody objects.

    Raises
    ------
    SystemExit
        If either body is not found or has invalid parameters.
    """
    bodies = load_bodies()
    start = find_body(bodies, start_id)
    end = find_body(bodies, end_id)

    if not start:
        print(f"Error: body '{start_id}' not found.")
        sys.exit(1)
    if not end:
        print(f"Error: body '{end_id}' not found.")
        sys.exit(1)

    try:
        start_ob = make_orbital_body(start)
        end_ob = make_orbital_body(end)
    except ValueError:
        print("Error: invalid orbit parameters for the selected bodies.")
        sys.exit(1)

    return start, end, start_ob, end_ob


def cmd_stats(args):
    """Handle the 'stats' subcommand.

    Computes and prints Hohmann transfer delta-V, transfer time, and
    optionally a fast transfer profile using the delta-V budget.
    """
    start, end, start_ob, end_ob = resolve_bodies(args.start, args.end)

    dv_dep, dv_arr, total_hohmann = hohmann_delta_v(start_ob.sma, end_ob.sma)
    hohmann_tof = transfer_time(start_ob.sma, end_ob.sma, 1.0)

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

    if available_dv_m is not None:
        if available_dv_m < total_hohmann - 1:
            print(f"\n  WARNING: insufficient delta-V for this transfer.")
            print(f"  Need {total_hohmann / 1000:.2f} km/s, have {args.dv:.2f} km/s")
        else:
            fast_factor, fast_dv = find_factor_for_dv(
                start_ob.sma, end_ob.sma, available_dv_m
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

    from foslas.viz import visualize

    start, end, start_ob, end_ob = resolve_bodies(args.start, args.end)

    available_dv_m = args.dv * KM_TO_M if args.dv else 30.0 * KM_TO_M
    _, _, total_hohmann = hohmann_delta_v(start_ob.sma, end_ob.sma)
    fast_factor, fast_dv = find_factor_for_dv(start_ob.sma, end_ob.sma, available_dv_m)

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


def cmd_list(args):
    """Handle the 'list' subcommand.

    Lists available celestial bodies, optionally filtered by search query.
    """
    bodies = load_bodies()
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

    p_list = sub.add_parser("list", help="List available celestial bodies")
    p_list.add_argument(
        "-s", "--search", default=None, help="Filter bodies by name or ID"
    )
    p_list.set_defaults(func=cmd_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
