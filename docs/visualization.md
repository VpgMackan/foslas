# Visualization

## Overview

foslas provides matplotlib-based visualization of orbital transfers. The visualization shows the Sun, planetary orbits, and transfer trajectories in a 2D projection of the ecliptic plane.

## Plot Components

### 1. The Sun

Plotted as a yellow circle at the origin:

```python
ax.plot(0, 0, "yo", label="Sun", markersize=15)
```

### 2. Planetary Orbits

Full elliptical orbits are plotted using the polar equation:

```
r(θ) = a × (1 - e²) / (1 + e × cos(θ))
```

```python
def plot_orbit(ax, body_data, rotation=0):
    a = (body_data["aphelion"] + body_data["perihelion"]) / 2
    e = (body_data["aphelion"] - body_data["perihelion"]) / (
        body_data["aphelion"] + body_data["perihelion"]
    )
    theta = np.linspace(0, 2 * np.pi, 1000)
    r = (a * (1 - e**2)) / (1 + e * np.cos(theta))
    ax.plot(
        r * np.cos(theta + rotation) / AU_TO_KM,
        r * np.sin(theta + rotation) / AU_TO_KM,
        linewidth=1.5,
        label=f"Orbit for {body_data['englishName']}",
    )
```

**Rotation:** Orbits are rotated so the departure point aligns with the x-axis for visual clarity.

### 3. Transfer Trajectories

Transfer arcs are plotted with directional arrows and burn markers:

```python
def plot_transfer(ax, x, y, dep, arr, label, color, linestyle="-"):
    # Main trajectory line
    ax.plot(x, y, linestyle=linestyle, color=color, linewidth=2, label=label)
    
    # Departure burn (triangle marker)
    ax.plot(dep[0], dep[1], marker="^", color=color, markersize=10, zorder=5)
    
    # Arrival burn (square marker)
    ax.plot(arr[0], arr[1], marker="s", color=color, markersize=10, zorder=5)
    
    # Directional arrow at 25% along path
    n = len(x)
    if n > 10:
        i = n // 4
        ax.add_patch(
            FancyArrowPatch(
                (x[i], y[i]),
                (x[i + 2], y[i + 2]),
                arrowstyle="->",
                color=color,
                mutation_scale=15,
                lw=1.5,
            )
        )
```

### 4. Legend and Labels

```python
ax.plot([], [], "g^", markersize=10, label="Departure Burn")
ax.plot([], [], "ms", markersize=10, label="Arrival Burn")

ax.set_aspect("equal", adjustable="box")
ax.grid(True, alpha=0.3)
ax.legend(loc="upper right", fontsize=9)
ax.set_xlabel("Distance (AU)", fontsize=11)
ax.set_ylabel("Distance (AU)", fontsize=11)
ax.set_title(
    "Planetary Transfer Trajectory\n(ODE Integration with Lambert Solver)",
    fontsize=13,
)
```

### 5. Statistics Box

An optional text box shows transfer statistics:

```python
if stats:
    stats_text = (
        "--- Hohmann Transfer ---\n"
        f"Dv required: {stats['hohmann_dv']:.2f} km/s\n"
        f"Est. time:    {stats['hohmann_time']:.1f} days\n\n"
        "--- Fast Transfer ---\n"
        f"Dv used:       {stats['fast_dv']:.2f} km/s\n"
        f"Energy factor: {stats['fast_factor']:.2f}\n"
        f"Est. time:     {stats['fast_time']:.1f} days"
    )
    props = dict(boxstyle="round,pad=0.5", facecolor="black", alpha=0.7)
    ax.text(
        0.02,
        0.02,
        stats_text,
        transform=ax.transAxes,
        fontsize=13,
        verticalalignment="bottom",
        fontfamily="monospace",
        color="white",
        bbox=props,
    )
```

## Orbit Rotation

To make the visualization intuitive, orbits are rotated so the departure point lies on the positive x-axis:

### Departure Orbit

The departure orbit is rotated by the angle needed to align the departure point with the x-axis:

```python
alpha_start = np.arccos(np.clip(-e_start, -1, 1))
```

### Arrival Orbit

The arrival orbit is rotated so that the arrival point (where the transfer arc ends) is correctly positioned:

```python
_, _, hohmann_dv = hohmann_delta_v(r1, r2)
x_h, y_h, dep_h, arr_h, nu_h = compute_transfer_trajectory(r1, r2, hohmann_dv)
alpha_end_h = nu_h - np.arccos(np.clip(-e_end, -1, 1))
```

## Visualization Pipeline

The `visualize()` function orchestrates the full plot:

```python
def visualize(r1, r2, target_dv, bodies_data, stats=None):
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 1. Plot Sun
    ax.plot(0, 0, "yo", label="Sun", markersize=15)
    
    # 2. Compute orbit rotations
    alpha_start = ...  # departure orbit rotation
    alpha_end_h = ...  # arrival orbit rotation for Hohmann
    
    # 3. Plot both orbits
    for i, body in enumerate(bodies_data):
        rot = alpha_start if i == 0 else alpha_end_h
        plot_orbit(ax, body, rotation=rot)
    
    # 4. Compute and plot Hohmann transfer
    x_h, y_h, dep_h, arr_h, nu_h = compute_transfer_trajectory(r1, r2, hohmann_dv)
    plot_transfer(ax, x_h, y_h, dep_h, arr_h, "Hohmann Transfer", "cyan", linestyle="--")
    
    # 5. Compute and plot fast transfer (if budget allows)
    if target_dv > hohmann_dv + 1.0:
        x_f, y_f, dep_f, arr_f, nu_f = compute_transfer_trajectory(
            r1, r2, target_dv, target_ecc=e_end, target_rot=alpha_end_h
        )
        plot_transfer(ax, x_f, y_f, dep_f, arr_f, "Fast Transfer", "red")
    
    # 6. Add legend and labels
    ax.plot([], [], "g^", markersize=10, label="Departure Burn")
    ax.plot([], [], "ms", markersize=10, label="Arrival Burn")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    
    # 7. Add stats box
    if stats:
        ax.text(0.02, 0.02, stats_text, ...)
    
    plt.tight_layout()
```

## Color Scheme

| Element | Color | Style |
|---------|-------|-------|
| Sun | Yellow | Circle marker |
| Departure orbit | Default (blue) | Solid line |
| Arrival orbit | Default (blue) | Solid line |
| Hohmann transfer | Cyan | Dashed line |
| Fast transfer | Red | Solid line |
| Departure burn | Green | Triangle marker |
| Arrival burn | Magenta | Square marker |

## Output

The plot is saved as a PNG file with high resolution:

```python
plt.savefig(output, dpi=150, bbox_inches="tight")
```

- **DPI:** 150 (print quality)
- **Bounding box:** Tight to avoid clipping

## Dependencies

- **matplotlib** — Core plotting library
- **FancyArrowPatch** — Directional arrows on trajectories
- **Agg backend** — Non-interactive rendering for file output
