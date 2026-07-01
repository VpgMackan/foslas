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

## Implementation

foslas delegates Lambert's problem solving to [pykep](https://github.com/esa/pykep), a library for astrodynamics and orbit calculations. The `lambert_solve` function in `foslas/lambert.py` is a thin wrapper that calls `pk.lambert_problem`:

```python
def lambert_solve(r1_vec, r2_vec, tof, mu=GM_SUN):
    lp = pk.lambert_problem(r1_vec, r2_vec, tof, mu)
    v1 = np.array(lp.v0[0])
    v2 = np.array(lp.v1[0])
    return v1, v2
```

pykep's implementation uses a robust numerical solver that handles all conic sections (elliptical, parabolic, hyperbolic) without requiring separate equations for each case.

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

1. **No solution found:** If no valid transfer is found within the search space, the solver falls back to a Hohmann trajectory.

## References

- Bate, R. R., Mueller, D. D., & White, J. E. (1971). *Fundamentals of Astrodynamics*. Dover.
- Curtis, H. D. (2013). *Orbital Mechanics for Engineering Students*. Butterworth-Heinemann.
- Vallado, D. A. (2013). *Fundamentals of Astrodynamics and Applications*. Microcosm Press.