import json
import numpy as np

# Constants
G = 6.67430e-11
M_SUN = 1.98847e30
GM_SUN = G * M_SUN
KM_TO_M = 1000


class OrbitalStructure:
    def __init__(self, apogee_km: float, perigee_km: float):
        if perigee_km <= 0 or apogee_km < perigee_km:
            raise ValueError(f"Invalid orbit")

        self.apogee = apogee_km * KM_TO_M
        self.perigee = perigee_km * KM_TO_M
        self.semi_major_axis = (self.apogee + self.perigee) / 2
        self.current_radius = self.semi_major_axis


class RocketStructure:
    @staticmethod
    def calc_transfer_delta_v(r1: float, r2: float, extra_factor: float) -> float:
        # Departure circular velocity
        v1 = np.sqrt(GM_SUN / r1)
        # Arrival circular velocity
        v2 = np.sqrt(GM_SUN / r2)

        # Transfer orbit semi-major axis
        a_transfer = ((r1 + r2) / 2) * extra_factor

        # Departure velocity (periapsis of transfer orbit)
        vt1 = np.sqrt(GM_SUN * (2 / r1 - 1 / a_transfer))
        # Cost of departure burn (purely tangential)
        dv1 = abs(vt1 - v1)

        if extra_factor == 1.0:
            # Hohmann transfer: arriving perfectly parallel at aphelion
            vt2 = np.sqrt(GM_SUN * (2 / r2 - 1 / a_transfer))
            dv2 = abs(v2 - vt2)
        else:
            # Fast transfer: arriving at an angle, requiring vector subtraction
            # 1. Angular momentum is conserved (h = r * v)
            h = r1 * vt1

            # 2. Arrival tangential velocity
            v_theta = h / r2

            # 3. Arrival total velocity magnitude
            v_arr_mag = np.sqrt(GM_SUN * (2 / r2 - 1 / a_transfer))

            # 4. Arrival radial velocity (Pythagorean theorem)
            v_radial = np.sqrt(max(0, v_arr_mag**2 - v_theta**2))

            # 5. Delta-V required to match purely circular target orbit
            dv2 = np.sqrt(v_radial**2 + (v_theta - v2) ** 2)

        return dv1 + dv2

    @staticmethod
    def find_factor_for_dv(r1, r2, target_dv, max_factor=50.0):
        # Increased max_factor to allow for true high-energy hyper-fast transfers
        low, high = 1.0, max_factor
        best_factor = 1.0
        best_dv = RocketStructure.calc_transfer_delta_v(r1, r2, 1.0)

        # If they don't even have enough DV for a standard transfer
        if target_dv < best_dv:
            return 1.0, best_dv

        for _ in range(80):
            mid = (low + high) / 2
            dv = RocketStructure.calc_transfer_delta_v(r1, r2, mid)

            if abs(dv - target_dv) < 100:  # Within 100 m/s is good enough
                return mid, dv

            if dv < target_dv:
                low = mid
                best_factor = mid
                best_dv = dv
            else:
                high = mid

        return best_factor, best_dv


def main():
    print("=== Orbital Transfer Calculator (by Delta-V Budget) ===\n")

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
        start = OrbitalStructure(a1, p1)
        end = OrbitalStructure(a2, p2)
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

    available_dv_m = available_dv_km * 1000

    eff_factor, eff_dv = RocketStructure.find_factor_for_dv(
        start.current_radius, end.current_radius, min(available_dv_m * 0.9, 50_000)
    )
    fast_factor, fast_dv = RocketStructure.find_factor_for_dv(
        start.current_radius, end.current_radius, available_dv_m
    )

    def get_time(factor):
        # Calculate time using Kepler's Equation for true flight time to intersection
        a = ((start.current_radius + end.current_radius) / 2) * factor
        e = 1 - (start.current_radius / a)

        if e == 0:
            return 0

        # Calculate Eccentric Anomaly at the destination radius
        cos_E = (1 - (end.current_radius / a)) / e
        cos_E = max(-1.0, min(1.0, cos_E))  # Clamp to avoid float precision errors

        E = np.arccos(cos_E)

        # Mean Anomaly
        M = E - e * np.sin(E)

        # Final time in seconds, converted to days
        t_sec = M * np.sqrt(a**3 / GM_SUN)
        return t_sec / 86400

    print("\n" + "=" * 65)
    print("EFFICIENT TRANSFER (Hohmann)")
    print(
        f"Δv required:   {RocketStructure.calc_transfer_delta_v(start.current_radius, end.current_radius, 1.0)/1000:.2f} km/s"
    )
    print(f"Est. time:     {get_time(1.0):.1f} days")

    if available_dv_m < RocketStructure.calc_transfer_delta_v(
        start.current_radius, end.current_radius, 1.0
    ):
        print(f"\nWARNING: You do not have enough Delta-V for this journey.")
    else:
        print("\nFAST TRANSFER (Budget Maxed)")
        print(f"Δv used:       {fast_dv/1000:.2f} km/s")
        print(f"Energy factor: {fast_factor:.2f}")
        print(f"Est. time:     {get_time(fast_factor):.1f} days")
    print("=" * 65)


if __name__ == "__main__":
    main()
