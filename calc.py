import json
import numpy as np
from scipy.optimize import brentq
from scipy.integrate import solve_ivp

# Constants
G = 6.67430e-11
M_SUN = 1.98847e30
GM_SUN = G * M_SUN
KM_TO_M = 1000
AU_TO_M = 1.496e11
SEC_TO_DAY = 86400.0

# ==============================================================================
# Physics Engine - All calculations in SI units (meters, kg, seconds)
# ==============================================================================

class OrbitalStructure:
    def __init__(self, apogee_km: float, perigee_km: float):
        if perigee_km <= 0 or apogee_km < perigee_km:
            raise ValueError(f"Invalid orbit")

        self.apogee = apogee_km * KM_TO_M
        self.perigee = perigee_km * KM_TO_M
        self.semi_major_axis = (self.apogee + self.perigee) / 2
        self.current_radius = self.semi_major_axis

    @property
    def eccentricity(self) -> float:
        return (self.apogee - self.perigee) / (self.apogee + self.perigee)


class TrajectoryIntegrator:
    @staticmethod
    def two_body_equations(t: float, state: np.ndarray) -> np.ndarray:
        """ODE for two-body problem."""
        r = np.linalg.norm(state[:3])
        return np.array([
            state[3], state[4], state[5],
            -GM_SUN * state[0] / r**3,
            -GM_SUN * state[1] / r**3,
            -GM_SUN * state[2] / r**3
        ])

    @staticmethod
    def integrate_trajectory(r1_vec: np.ndarray, v1_vec: np.ndarray,
                             tof: float, points: int = 100) -> tuple:
        """Integrate the trajectory from initial state to final time."""
        initial_state = np.concatenate([r1_vec, v1_vec])
        t_eval = np.linspace(0, tof, points)

        solution = solve_ivp(
            TrajectoryIntegrator.two_body_equations,
            [0, tof],
            initial_state,
            t_eval=t_eval,
            method='DOP853',
            rtol=1e-12,
            atol=1e-14
        )

        return solution.y[:3].T, solution.t


class LambertSolver:
    """
    Lambert solver using universal variables.
    Computes departure and arrival velocities for a given time of flight.
    """

    @staticmethod
    def solve(r1: np.ndarray, r2: np.ndarray, tof: float, mu: float = GM_SUN) -> tuple:
        """
        Solve Lambert's problem for two position vectors and time of flight.

        Args:
            r1: Initial position vector
            r2: Final position vector
            tof: Time of flight
            mu: Standard gravitational parameter

        Returns:
            (v1, v2): Departure and arrival velocity vectors
        """
        r1 = np.asarray(r1, dtype=float)
        r2 = np.asarray(r2, dtype=float)

        r1_mag = np.linalg.norm(r1)
        r2_mag = np.linalg.norm(r2)

        # Chord length
        c = np.linalg.norm(r2 - r1)

        # Calculate the angle between r1 and r2
        cos_gamma = np.dot(r1, r2) / (r1_mag * r2_mag)
        cos_gamma = np.clip(cos_gamma, -1.0, 1.0)
        gamma = np.arccos(cos_gamma)

        # Universal variable formulation
        def find_x(x):
            # Stumpff functions
            if abs(x) < 1e-12:
                c2, c3 = 0.5, 1.0/6.0
            elif x > 0:
                s = np.sqrt(x)
                c2 = (1 - np.cos(s)) / x
                c3 = (s - np.sin(s)) / (x * s)
            else:
                s = np.sqrt(-x)
                c2 = (np.cosh(s) - 1) / (-x)
                c3 = (np.sinh(s) - s) / (-x * s)

            # Semi-latus rectum
            p = c * (r1_mag + r2_mag) / 2 + c3 * c * (r1_mag - r2_mag) / 2

            # Semi-major axis
            a = p / (1 - c2 * (r1_mag + r2_mag) / 2)

            # Time of flight
            n = np.sqrt(mu / abs(a)**3) if a != 0 else 1.0
            y = c2 + c3 * c / 6
            t = y / n

            return t - tof

        # Solve for x using Brent's method
        x_low, x_high = 0.01, 100.0

        try:
            x = brentq(find_x, x_low, x_high, xtol=1e-12, rtol=1e-12)
        except ValueError:
            x = 1.0

        # Calculate velocities
        c2 = LambertSolver._stumpff_c2(x)
        c3 = LambertSolver._stumpff_c3(x)

        f = 1 - c2 * (r1_mag + r2_mag) / 2 + c3 * c / 6
        g = c2 * c
        f_dot = 1 - c3 * (r1_mag + r2_mag) / (6 * x) if abs(x) > 1e-12 else 1
        g_dot = 1 - c3 * c / 2

        v1 = (r2 - f * r1) / g_dot
        v2 = (f_dot * r2 - r1) / g

        return v1, v2

    @staticmethod
    def _stumpff_c2(z):
        if abs(z) < 1e-12:
            return 0.5
        elif z > 0:
            s = np.sqrt(z)
            return (1 - np.cos(s)) / z
        else:
            s = np.sqrt(-z)
            return (np.cosh(s) - 1) / (-z)

    @staticmethod
    def _stumpff_c3(z):
        if abs(z) < 1e-12:
            return 1.0/6.0
        elif z > 0:
            s = np.sqrt(z)
            return (s - np.sin(s)) / (z * s)
        else:
            s = np.sqrt(-z)
            return (np.sinh(s) - s) / (-z * s)


class RocketStructure:
    @staticmethod
    def calc_hohmann_delta_v(r1: float, r2: float) -> tuple:
        """Calculate Hohmann transfer delta-v."""
        v1_circ = np.sqrt(GM_SUN / r1)
        v2_circ = np.sqrt(GM_SUN / r2)

        a_transfer = (r1 + r2) / 2
        v_periapsis = np.sqrt(GM_SUN * (2 / r1 - 1 / a_transfer))
        v_apoapsis = np.sqrt(GM_SUN * (2 / r2 - 1 / a_transfer))

        dv_dep = v_periapsis - v1_circ
        dv_arr = v2_circ - v_apoapsis
        total_dv = dv_dep + dv_arr

        return dv_dep, dv_arr, total_dv, {'semi_major_axis': a_transfer}

    @staticmethod
    def calc_transfer_delta_v(r1: float, r2: float, factor: float,
                             tof_override: float = None) -> tuple:
        """Calculate transfer delta-v using Energy Factor approach with vector summation."""
        if abs(factor - 1.0) < 1e-6:
            dv_dep, dv_arr, total_dv, orbit_params = RocketStructure.calc_hohmann_delta_v(r1, r2)
            return total_dv, np.array([0, dv_dep, 0]), np.array([0, -dv_arr, 0]), orbit_params

        a_transfer = ((r1 + r2) / 2) * factor
        e = 1 - r1 / a_transfer

        v1_circ = np.sqrt(GM_SUN / r1)
        v2_circ = np.sqrt(GM_SUN / r2)

        vt1 = np.sqrt(GM_SUN * (2 / r1 - 1 / a_transfer))

        h = r1 * vt1
        v_theta = h / r2
        v_arr_mag = np.sqrt(GM_SUN * (2 / r2 - 1 / a_transfer))
        v_radial = np.sqrt(max(0, v_arr_mag**2 - v_theta**2))

        dv_dep_vec = np.array([0, vt1 - v1_circ, 0])
        dv_arr_vec = np.array([v_radial, v2_circ - v_theta, 0])

        dv_dep = np.linalg.norm(dv_dep_vec)
        dv_arr = np.linalg.norm(dv_arr_vec)
        total_dv = dv_dep + dv_arr

        if tof_override is None:
            cos_E = e
            E = np.arccos(np.clip(cos_E, -1, 1))
            M = E - e * np.sin(E)
            tof = M * np.sqrt(a_transfer**3 / GM_SUN)
        else:
            tof = tof_override

        return total_dv, dv_dep_vec, dv_arr_vec, {'times_of_flight': tof, 'factor': factor}

    @staticmethod
    def find_factor_for_dv(r1: float, r2: float, target_dv: float, max_factor: float = 50.0) -> tuple:
        """Find the energy factor that achieves the target delta-v using high-precision root-finding."""
        dv_hohmann, _, _, _ = RocketStructure.calc_hohmann_delta_v(r1, r2)

        if target_dv < dv_hohmann:
            return 1.0, dv_hohmann

        def delta_v_residual(factor):
            dv, _, _, _ = RocketStructure.calc_transfer_delta_v(r1, r2, factor)
            return dv - target_dv

        try:
            factor = brentq(delta_v_residual, 1.0, max_factor, xtol=1e-10, rtol=1e-12)
        except ValueError:
            factor = max_factor

        dv, _, _, _ = RocketStructure.calc_transfer_delta_v(r1, r2, factor)
        return factor, dv

    @staticmethod
    def get_transfer_path(r1: float, r2: float, factor: float, points: int = 100) -> tuple:
        """Generate the transfer trajectory using ODE integration."""
        if abs(factor - 1.0) < 1e-6:
            a = (r1 + r2) / 2
            e = (r2 - r1) / (r2 + r1)

            thetas = np.linspace(0, np.pi, points)
            radii = (a * (1 - e**2)) / (1 + e * np.cos(thetas))

            x = radii * np.cos(thetas)
            y = radii * np.sin(thetas)

            return x, y, np.array([r1 / AU_TO_M, 0.0]), np.array([-r2 / AU_TO_M, 0.0])

        # For fast transfers, compute the trajectory using the Energy Factor approach
        a_transfer = ((r1 + r2) / 2) * factor
        e = 1 - r1 / a_transfer

        # Time of flight
        tof = 0.5 * np.pi * np.sqrt(a_transfer**3 / GM_SUN)

        # Departure velocity
        vt1 = np.sqrt(GM_SUN * (2 / r1 - 1 / a_transfer))
        r1_vec = np.array([r1, 0.0, 0.0])
        v1_vec = np.array([0, vt1, 0.0])

        # Integrate trajectory
        positions, _ = TrajectoryIntegrator.integrate_trajectory(r1_vec, v1_vec, tof, points)

        x = positions[:, 0] / AU_TO_M
        y = positions[:, 1] / AU_TO_M

        dep_burn = np.array([r1 / AU_TO_M, 0.0])
        arr_burn = np.array([-r2 / AU_TO_M, 0.0])

        return x, y, dep_burn, arr_burn


# ==============================================================================
# User Interface
# ==============================================================================

def calculate_transfer_time(r1: float, r2: float, factor: float) -> float:
    """Calculate transfer time in days."""
    a = ((r1 + r2) / 2) * factor
    e = 1 - (r1 / a)

    if e < 1e-10:
        return 0.0

    cos_E = (1 - (r2 / a)) / e
    cos_E = np.clip(cos_E, -1.0, 1.0)
    E = np.arccos(cos_E)

    M = E - e * np.sin(E)
    t_sec = M * np.sqrt(a**3 / GM_SUN)
    return t_sec / SEC_TO_DAY


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

    print(f"\nTransfer: {start_body.get('name', start_id)} → {end_body.get('name', end_id)}")

    try:
        available_dv_km = float(input("\nAvailable total Delta-V (km/s): ") or 30.0)
    except ValueError:
        available_dv_km = 30.0

    available_dv_m = available_dv_km * 1000

    eff_dv, eff_dep, eff_arr, _ = RocketStructure.calc_hohmann_delta_v(
        start.current_radius, end.current_radius
    )

    fast_factor, fast_dv = RocketStructure.find_factor_for_dv(
        start.current_radius, end.current_radius, available_dv_m
    )

    print("\n" + "=" * 65)
    print("EFFICIENT TRANSFER (Hohmann)")
    print(f"Δv required:   {eff_dv/1000:.2f} km/s")
    print(f"Est. time:     {calculate_transfer_time(start.current_radius, end.current_radius, 1.0):.1f} days")

    if available_dv_m < eff_dv - 1:
        print(f"\nWARNING: You do not have enough Delta-V for this journey.")
    else:
        print("\nFAST TRANSFER (Budget Maxed)")
        print(f"Δv used:       {fast_dv/1000:.2f} km/s")
        print(f"Energy factor: {fast_factor:.2f}")
        print(f"Est. time:     {calculate_transfer_time(start.current_radius, end.current_radius, fast_factor):.1f} days")
    print("=" * 65)


if __name__ == "__main__":
    main()