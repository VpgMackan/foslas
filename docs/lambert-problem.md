# Lambert's Problem

## Overview

Lambert's problem is a fundamental problem in orbital mechanics: given two position vectors and a time of flight, find the velocity vectors at both positions. This allows us to compute transfer trajectories between arbitrary points in space, not just tangential Hohmann transfers.

## Problem Statement

Given:
- Position vector **r₁** at departure
- Position vector **r₂** at arrival
- Time of flight `tof`

Find:
- Velocity vector **v₁** at departure
- Velocity vector **v₂** at arrival

## Universal Variable Formulation

foslas solves Lambert's problem using the universal variable formulation with Stumpff functions. This approach works for all conic sections (elliptical, parabolic, hyperbolic) without separate equations for each case.

### Geometry

First, we compute the geometric parameters:

1. **Magnitudes:**
   ```
   r₁ = |r₁|
   r₂ = |r₂|
   ```

2. **True anomaly change (Δν):**
   ```
   cos(Δν) = (r₁ · r₂) / (r₁ × r₂)
   ```
   
   The cross product determines the direction (prograde vs retrograde):
   ```
   if (r₁ × r₂)_z ≥ 0:
       Δν = arccos(cos(Δν))
   else:
       Δν = 2π - arccos(cos(Δν))
   ```

3. **Geometry constant A:**
   ```
   A = sin(Δν) × √(r₁ × r₂ / (1 - cos(Δν)))
   ```

### Stumpff Functions

Stumpff functions `C(z)` and `S(z)` are special functions that unify the equations for all conic sections. They depend on a universal variable `z`:

**For z > 0 (elliptical):**
```
C(z) = (1 - cos(√z)) / z
S(z) = (√z - sin(√z)) / z^(3/2)
```

**For z ≈ 0 (parabolic):**
```
C(z) ≈ 1/2
S(z) ≈ 1/6
```

**For z < 0 (hyperbolic):**
```
C(z) = (cosh(√(-z)) - 1) / (-z)
S(z) = (sinh(√(-z)) - √(-z)) / (-z)^(3/2)
```

### Time of Flight Equation

The time of flight is related to `z` through:

```
y = r₁ + r₂ + A × (z × S(z) - 1) / √C(z)

x = √(y / C(z))

t = (x³ × S(z) + A × √y) / √μ
```

where `μ = GM_SUN` is the gravitational parameter.

### Solving for z

We need to find `z` such that `t(z) = tof`. This is a root-finding problem:

```
f(z) = t(z) - tof = 0
```

**Initial search interval:**
- Try `z ∈ [-2, 4]` with Brent's method (`brentq`)

**Fallback scan:**
If the initial search fails, scan 200 points across:
- `z ∈ [0.01, 4π²]` (elliptical solutions)
- `z ∈ [-4π², -0.01]` (hyperbolic solutions)

Look for sign changes in the residual, then refine with `brentq`.

### Computing Velocities

Once `z` is found, compute the Lagrange coefficients:

```
f = 1 - y / r₁
g = A × √(y / μ)
ġ = 1 - y / r₂
```

The velocity vectors are:

```
v₁ = (r₂ - f × r₁) / g
v₂ = (ġ × r₂ - r₁) / g
```

## Implementation

### Stumpff Functions

```python
def stumpff_C(z):
    if abs(z) < 1e-6:
        return 0.5
    elif z > 0:
        return (1 - np.cos(np.sqrt(z))) / z
    else:
        return (np.cosh(np.sqrt(-z)) - 1) / (-z)

def stumpff_S(z):
    if abs(z) < 1e-6:
        return 1.0 / 6.0
    elif z > 0:
        sz = np.sqrt(z)
        return (sz - np.sin(sz)) / (sz ** 3)
    else:
        sz = np.sqrt(-z)
        return (np.sinh(sz) - sz) / ((-z) ** 1.5)
```

### Vectorized Versions

For batch scanning, vectorized versions operate on numpy arrays:

```python
def _stumpff_C_vec(z):
    result = np.empty_like(z)
    small = np.abs(z) < 1e-6
    pos = (z > 0) & ~small
    neg = (z < 0) & ~small
    
    result[small] = 0.5
    result[pos] = (1 - np.cos(np.sqrt(z[pos]))) / z[pos]
    result[neg] = (np.cosh(np.sqrt(-z[neg])) - 1) / (-z[neg])
    return result
```

### Main Solver

```python
def lambert_solve(r1_vec, r2_vec, tof, mu=GM_SUN):
    # Compute geometry
    r1_mag = np.sqrt(np.sum(r1_vec**2))
    r2_mag = np.sqrt(np.sum(r2_vec**2))
    
    cos_dnu = np.clip(np.dot(r1_vec, r2_vec) / (r1_mag * r2_mag), -1.0, 1.0)
    cross = np.cross(r1_vec, r2_vec)
    dnu = np.arccos(cos_dnu) if cross[2] >= 0 else 2 * np.pi - np.arccos(cos_dnu)
    
    A = np.sin(dnu) * np.sqrt(r1_mag * r2_mag / (1 - cos_dnu))
    
    # Time of flight residual
    def tof_residual(z):
        C = stumpff_C(z)
        S = stumpff_S(z)
        denom = np.sqrt(C)
        if denom < 1e-30:
            return 1e20
        y = r1_mag + r2_mag + A * (z * S - 1) / denom
        if y < 0:
            return 1e20
        x = np.sqrt(y / C)
        t = (x ** 3 * S + A * np.sqrt(y)) / np.sqrt(mu)
        return t - tof
    
    # Find z
    z = brentq(tof_residual, -2.0, 4.0, xtol=1e-12, rtol=1e-12)
    
    # Compute velocities
    C = stumpff_C(z)
    S = stumpff_S(z)
    y = r1_mag + r2_mag + A * (z * S - 1) / np.sqrt(C)
    
    f = 1 - y / r1_mag
    g = A * np.sqrt(y / mu)
    g_dot = 1 - y / r2_mag
    
    v1 = (r2_vec - f * r1_vec) / g
    v2 = (g_dot * r2_vec - r1_vec) / g
    
    return v1, v2
```

## Transfer Search Algorithm

foslas uses Lambert's solver in a brute-force search to find the fastest transfer within a delta-V budget:

### Search Space

- **Time-of-flight fractions:** 60 values from 5% to 95% of Hohmann time
- **Arrival angles (Δν):** 40 values from 0.3 to π-0.05 radians

### Algorithm

```python
def _search_transfer(r1, r2, target_dv, ...):
    hohmann_tof = np.pi * np.sqrt(((r1 + r2) / 2) ** 3 / GM_SUN)
    
    for tof_frac in np.linspace(0.05, 0.95, 60):
        tof = hohmann_tof * tof_frac
        for dnu in np.linspace(0.3, np.pi - 0.05, 40):
            # Compute arrival position
            orbit_angle = dnu - target_rot
            r2_actual = _compute_r2_actual(r2, target_ecc, orbit_angle)
            r2_vec = [r2_actual * cos(dnu), r2_actual * sin(dnu), 0]
            
            # Solve Lambert's problem
            v1, v2 = lambert_solve(r1_vec, r2_vec, tof)
            
            # Compute delta-V
            dv_dep = |v1 - v_circ1|
            dv_arr = |v2 - v_circ2|
            total_dv = dv_dep + dv_arr
            
            if total_dv <= target_dv:
                if best is None or tof < best[0]:
                    best = (tof, dnu, v1, r1_vec, r2_actual)
        
        if best is not None:
            break  # Found valid transfer at this TOF fraction
    
    return best
```

The search returns the **fastest** valid transfer (lowest time of flight) that fits within the delta-V budget.

## Error Handling

The Lambert solver handles several edge cases:

1. **Denominator near zero:** If `√C(z) < 1e-30`, return large residual
2. **Negative y:** If `y < 0`, return large residual (invalid geometry)
3. **No solution:** If no sign change found in scan, raise `ValueError`
4. **Fallback:** If Lambert fails, fall back to Hohmann trajectory

## References

- Bate, R. R., Mueller, D. D., & White, J. E. (1971). *Fundamentals of Astrodynamics*. Dover.
- Curtis, H. D. (2013). *Orbital Mechanics for Engineering Students*. Butterworth-Heinemann.
- Vallado, D. A. (2013). *Fundamentals of Astrodynamics and Applications*. Microcosm Press.
