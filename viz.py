import numpy as np
import matplotlib.pyplot as plt

from calc import RocketStructure, OrbitalStructure

plt.figure(figsize=(8, 8))
plt.plot(0, 0, "yo", label="Sun", markersize=10)

PLOTS = [
    {"name": "Earth", "perihelion": 147099736.0159, "aphelion": 152096309.98409998},
    {"name": "Mars", "perihelion": 206626884.79999998, "aphelion": 249251515.2},
]
KM_TO_AU = 1.496e8

for idx, i in enumerate(PLOTS):
    # Calculate orbital parameters
    a = (i["aphelion"] + i["perihelion"]) / 2
    e = (i["aphelion"] - i["perihelion"]) / (i["aphelion"] + i["perihelion"])

    # Generate theta values
    theta = np.linspace(0, 2 * np.pi, 1000)

    # Calculate radius r for each theta
    r = (a * (1 - e**2)) / (1 + e * np.cos(theta))

    # Convert to Cartesian coordinates
    x = (r * np.cos(theta)) / KM_TO_AU
    y = (r * np.sin(theta)) / KM_TO_AU
    plt.plot(x, y, label=f"Orbit for {i['name']}")

START = "earth"
END = "mars"

start_body = next((b for b in PLOTS if b["name"].lower() == START), None)
end_body = next((b for b in PLOTS if b["name"].lower() == END), None)

start = OrbitalStructure(start_body["aphelion"], start_body["perihelion"])
end = OrbitalStructure(end_body["aphelion"], end_body["perihelion"])
factor, _ = RocketStructure.find_factor_for_dv(
    start.current_radius, end.current_radius, 30 * 1000
)

x_path, y_path = RocketStructure.get_transfer_path(
    start.current_radius, end.current_radius, factor
)
plt.plot(
    x_path / (KM_TO_AU * 1000),
    y_path / (KM_TO_AU * 1000),
    "--r",
    label="Transfer Trajectory",
)

# Formatting
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.title(f"Planetary Orbit (e={e:.4f})")
plt.xlabel("Distance (AU)")
plt.ylabel("Distance (AU)")

# Save the plot
plt.show()
