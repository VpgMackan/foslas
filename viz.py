import json
import matplotlib

matplotlib.use("Agg")

from calc import OrbitalBody, visualize, hohmann_delta_v, find_factor_for_dv, transfer_time, KM_TO_M


def main():
    with open("./data.json") as f:
        dataset = json.load(f)["bodies"]

    start_id = input("Start body ID: ").strip().lower()
    end_id = input("Destination body ID: ").strip().lower()

    start_body = next((b for b in dataset if b["id"].lower() == start_id), None)
    end_body = next((b for b in dataset if b["id"].lower() == end_id), None)

    if not start_body or not end_body:
        print("Body not found!")
        return

    def get_ap_per(body):
        if body.get("apogee", 0) > 0 and body.get("perigee", 0) > 0:
            return body["apogee"], body["perigee"]
        sma = body.get("semimajorAxis", 0)
        return sma, sma

    a1, p1 = get_ap_per(start_body)
    a2, p2 = get_ap_per(end_body)

    try:
        start = OrbitalBody(a1, p1)
        end = OrbitalBody(a2, p2)
    except ValueError:
        print("\nError: Invalid orbit parameters for the selected bodies.")
        return

    print(
        f"\nTransfer: {start_body.get('name', start_id)} → {end_body.get('name', end_id)}"
    )

    try:
        available_dv_km = float(input("\nAvailable total Delta-V (km/s): ") or 30.0)
    except ValueError:
        available_dv_km = 30.0

    available_dv_m = available_dv_km * KM_TO_M

    _, _, total_hohmann = hohmann_delta_v(start.sma, end.sma)
    fast_factor, fast_dv = find_factor_for_dv(start.sma, end.sma, available_dv_m)

    stats = {
        "hohmann_dv": total_hohmann / 1000,
        "hohmann_time": transfer_time(start.sma, end.sma, 1.0),
        "fast_dv": fast_dv / 1000,
        "fast_factor": fast_factor,
        "fast_time": transfer_time(start.sma, end.sma, fast_factor),
    }

    visualize(start.sma, end.sma, available_dv_m, [start_body, end_body], stats=stats)

    import matplotlib.pyplot as plt

    plt.savefig("Figure_1.png", dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
