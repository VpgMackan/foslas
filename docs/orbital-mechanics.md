# Orbital Mechanics

## Orbital Elements

An elliptical orbit around the Sun is fully described by six classical orbital elements. In foslas, we work with a simplified set focused on the orbit's shape and size:

### Semi-Major Axis (a)

The semi-major axis defines the size of the ellipse. It is the average of the aphelion and perihelion distances:

```
a = (r_aphelion + r_perihelion) / 2
```

### Eccentricity (e)

The eccentricity defines the shape of the ellipse:

```
e = (r_aphelion - r_perihelion) / (r_aphelion + r_perihelion)
```

- `e = 0` → circular orbit
- `0 < e < 1` → elliptical orbit
- `e = 1` → parabolic trajectory
- `e > 1` → hyperbolic trajectory

### Aphelion and Perihelion

- **Aphelion**: the farthest point from the Sun
- **Perihelion**: the closest point to the Sun

These are related to the semi-major axis and eccentricity:

```
r_aphelion = a × (1 + e)
r_perihelion = a × (1 - e)
```

## OrbitalBody Class

The `OrbitalBody` class in `foslas/orbital.py` encapsulates these properties:

```python
class OrbitalBody:
    def __init__(self, aphelion_km, perihelion_km):
        self.aphelion = aphelion_km * KM_TO_M  # Convert to meters
        self.perihelion = perihelion_km * KM_TO_M
        self.sma = (self.aphelion + self.perihelion) / 2

    @property
    def eccentricity(self):
        return (self.aphelion - self.perihelion) / (self.aphelion + self.perihelion)
```

## Circular Orbit Velocity

The velocity of a body in a circular orbit at radius `r` from the Sun is:

```
v_circ = √(GM_SUN / r)
```

where `GM_SUN` is the Sun's standard gravitational parameter (1.327 × 10²⁰ m³/s²).

## Vis-Viva Equation

The vis-viva equation gives the velocity of an object in an elliptical orbit at any point:

```
v = √(GM_SUN × (2/r - 1/a))
```

where:
- `r` is the current distance from the Sun
- `a` is the semi-major axis of the orbit

This equation is fundamental to all delta-V calculations in foslas.

## Hohmann Transfer

The Hohmann transfer is the most fuel-efficient two-impulse transfer between two circular coplanar orbits. It uses an elliptical transfer orbit that is tangent to both the departure and arrival orbits.

### Transfer Orbit

For a Hohmann transfer from orbit radius `r1` to `r2`:

1. **Transfer semi-major axis:**
   ```
   a_t = (r1 + r2) / 2
   ```

2. **Departure velocity on transfer orbit** (at `r1`):
   ```
   v_t1 = √(GM_SUN × (2/r1 - 1/a_t))
   ```

3. **Arrival velocity on transfer orbit** (at `r2`):
   ```
   v_t2 = √(GM_SUN × (2/r2 - 1/a_t))
   ```

### Delta-V Calculations

The delta-V required for each burn is the difference between the transfer orbit velocity and the circular orbit velocity:

**Departure burn:**
```
Δv_departure = v_t1 - v_circ1
             = √(GM_SUN × (2/r1 - 1/a_t)) - √(GM_SUN / r1)
```

**Arrival burn:**
```
Δv_arrival = v_circ2 - v_t2
           = √(GM_SUN / r2) - √(GM_SUN × (2/r2 - 1/a_t))
```

**Total delta-V:**
```
Δv_total = Δv_departure + Δv_arrival
```

### Implementation

```python
def hohmann_delta_v(r1, r2):
    v1_circ = np.sqrt(GM_SUN / r1)
    v2_circ = np.sqrt(GM_SUN / r2)
    a_t = (r1 + r2) / 2
    v_dep = np.sqrt(GM_SUN * (2 / r1 - 1 / a_t)) - v1_circ
    v_arr = v2_circ - np.sqrt(GM_SUN * (2 / r2 - 1 / a_t))
    return v_dep, v_arr, v_dep + v_arr
```

### Transfer Time

The transfer time for a Hohmann transfer is half the period of the transfer orbit:

```
T_hohmann = π × √(a_t³ / GM_SUN)
```

For a scaled transfer orbit with energy factor `f`:

```
a = a_t × f
e = 1 - r1 / a
```

Using Kepler's equation via eccentric anomaly:

```
cos(E) = (1 - r2/a) / e
M = E - e × sin(E)
t = M × √(a³ / GM_SUN)
```

where `E` is the eccentric anomaly and `M` is the mean anomaly.

## Energy Factor Scaling

foslas supports faster transfers by scaling the transfer orbit's semi-major axis by an energy factor `f > 1.0`. This creates a larger, faster orbit that uses more delta-V.

### Computing Delta-V for a Scaled Transfer

For a transfer orbit scaled by factor `f`:

```
a = a_t × f
```

The departure burn remains similar, but the arrival geometry changes because the transfer orbit no longer arrives tangentially. The arrival delta-V must account for the radial velocity component:

```
v_t1 = √(GM_SUN × (2/r1 - 1/a))
v_tangential = r1 × v_t1 / r2
v_arrival_mag = √(GM_SUN × (2/r2 - 1/a))
v_radial = √(v_arrival_mag² - v_tangential²)

Δv_departure = |v_t1 - v_circ1|
Δv_arrival = √(v_radial² + (v_circ2 - v_tangential)²)
```

### Finding the Optimal Factor

Given a delta-V budget, we find the energy factor that exactly uses it:

```python
def find_factor_for_dv(r1, r2, target_dv, max_factor=50.0):
    dv_hoh, _, _ = hohmann_delta_v(r1, r2)
    if target_dv < dv_hoh:
        return 1.0, dv_hoh  # Minimum energy

    def residual(factor):
        return _calc_dv_for_factor(r1, r2, factor) - target_dv

    factor = brentq(residual, 1.0, max_factor, xtol=1e-10, rtol=1e-12)
    return factor, _calc_dv_for_factor(r1, r2, factor)
```

This uses Brent's root-finding method to solve for the factor where the computed delta-V equals the budget.

## Hohmann Trajectory Geometry

The Hohmann transfer arc is a semi-ellipse from `r1` to `r2`. In polar coordinates:

```
r(θ) = a × (1 - e²) / (1 + e × cos(θ))
```

where:
- `a = (r1 + r2) / 2`
- `e = (r2 - r1) / (r2 + r1)`
- `θ` ranges from 0 to π (half orbit)

This is used to generate the transfer arc for visualization:

```python
def _hohmann_trajectory(r1, r2, points):
    a = (r1 + r2) / 2
    e = (r2 - r1) / (r2 + r1)
    thetas = np.linspace(0, np.pi, points)
    radii = (a * (1 - e ** 2)) / (1 + e * np.cos(thetas))
    x = radii * np.cos(thetas) / AU_TO_M
    y = radii * np.sin(thetas) / AU_TO_M
    return x, y
```
