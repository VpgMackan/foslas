import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from calc import RocketStructure, OrbitalStructure, AU_TO_M


def plot_planetary_orbits(ax, bodies_data: list):
    """Plot planetary orbits based on orbital elements."""
    for body in bodies_data:
        a = (body["aphelion"] + body["perihelion"]) / 2
        e = (body["aphelion"] - body["perihelion"]) / (body["aphelion"] + body["perihelion"])
        
        theta = np.linspace(0, 2 * np.pi, 1000)
        r = (a * (1 - e**2)) / (1 + e * np.cos(theta))
        
        x = (r * np.cos(theta)) / AU_TO_M
        y = (r * np.sin(theta)) / AU_TO_M
        ax.plot(x, y, label=f"Orbit for {body['name']}", linewidth=1.5)
        
        # Mark current position (at aphelion for simplicity)
        ax.plot(body["aphelion"] / AU_TO_M, 0, 'o', markersize=4)


def plot_transfer_trajectory(ax, x_path: np.ndarray, y_path: np.ndarray,
                            dep_burn: np.ndarray, arr_burn: np.ndarray,
                            start_name: str, end_name: str):
    """Plot the transfer trajectory with burn point annotations."""
    # Plot integrated trajectory
    ax.plot(x_path, y_path, "--r", linewidth=2, label="Transfer Trajectory")
    
    # Mark departure burn point
    ax.plot(dep_burn[0], dep_burn[1], "g^", markersize=12, label=f"Departure Burn ({start_name})")
    
    # Mark arrival burn point
    ax.plot(arr_burn[0], arr_burn[1], "bs", markersize=12, label=f"Arrival Burn ({end_name})")
    
    # Add arrows to show direction of travel
    n_points = len(x_path)
    if n_points > 10:
        arrow_idx = n_points // 4
        arrow = FancyArrowPatch(
            (x_path[arrow_idx], y_path[arrow_idx]),
            (x_path[arrow_idx + 2], y_path[arrow_idx + 2]),
            arrowstyle='->', color='red', mutation_scale=15, lw=1.5
        )
        ax.add_patch(arrow)


def main():
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot Sun at origin
    ax.plot(0, 0, "yo", label="Sun", markersize=15)
    
    # Bodies to plot
    PLOTS = [
        {"name": "Earth", "perihelion": 147099736.0159, "aphelion": 152096309.98409998},
        {"name": "Mars", "perihelion": 206626884.79999998, "aphelion": 249251515.2},
    ]
    
    plot_planetary_orbits(ax, PLOTS)
    
    START = "Earth"
    END = "Mars"
    
    start_body = next((b for b in PLOTS if b["name"].lower() == START.lower()), None)
    end_body = next((b for b in PLOTS if b["name"].lower() == END.lower()), None)
    
    if start_body and end_body:
        start = OrbitalStructure(start_body["aphelion"], start_body["perihelion"])
        end = OrbitalStructure(end_body["aphelion"], end_body["perihelion"])
        
        # Get transfer trajectory using Lambert solver
        factor, _ = RocketStructure.find_factor_for_dv(
            start.current_radius, end.current_radius, 30 * 1000
        )
        
        x_path, y_path, dep_burn, arr_burn = RocketStructure.get_transfer_path(
            start.current_radius, end.current_radius, factor, points=100
        )
        
        plot_transfer_trajectory(ax, x_path, y_path, dep_burn, arr_burn, START, END)
    
    # Formatting
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlabel("Distance (AU)", fontsize=11)
    ax.set_ylabel("Distance (AU)", fontsize=11)
    ax.set_title("Planetary Transfer Trajectory\n(ODE Integration with Energy Factor)", fontsize=13)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()