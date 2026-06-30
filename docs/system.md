# System Architecture

## Overview

**foslas** (Fast Orbital Solar System Lambert Solver) is a Python library for computing interplanetary transfer trajectories between celestial bodies in the solar system. It combines classical orbital mechanics (Hohmann transfers) with modern numerical methods (Lambert solvers, ODE integration) to compute both minimum-energy and faster, higher-energy transfer orbits.

## Project Structure

```
foslas/
├── foslas/                 # Core library package
│   ├── __init__.py         # Public API re-exports
│   ├── constants.py        # Physical constants & unit conversions
│   ├── integrator.py       # Numerical ODE integrator (2-body problem)
│   ├── lambert.py          # Lambert's problem solver
│   └── transfers/          # Transfer models + visualization modules
├── data/
│   └── data.json           # Legacy cached body data (optional)
└── docs/                   # Documentation
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Data Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Astropy ephemerides (runtime)                                  │
│       │                                                         │
│       ▼                                                         │
│  load_bodies() computes a, e, aphelion/perihelion               │
│  from heliocentric state vectors                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      Computation Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  User Input (start, end, dv budget)                               │
│       │                                                         │
│       ▼                                                         │
│  load_bodies() ──→ find_body() ──→ OrbitalBody objects         │
│       │                                                         │
│       ├──► hohmann_delta_v()  →  minimum-energy delta-V + time │
│       │                                                         │
│       └──► find_factor_for_dv()  →  energy factor for fast xfer│
│                │                                                │
│                └──► _search_transfer()  →  Lambert search       │
│                        │                                        │
│                        └──► lambert_solve()  →  velocities      │
│                                │                                │
│                                └──► integrate_trajectory()      │
│                                        │                        │
│                                        └──► visualize()         │
│                                                │                │
│                                                └──► PNG output  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Unit Conversions

All internal calculations use SI units (meters, seconds). The visualization handles conversions:

| Constant | Value | Description |
|----------|-------|-------------|
| `G` | 6.67430 × 10⁻¹¹ m³ kg⁻¹ s⁻² | Gravitational constant |
| `M_SUN` | 1.98847 × 10³⁰ kg | Sun's mass |
| `GM_SUN` | G × M_SUN | Standard gravitational parameter |
| `KM_TO_M` | 1000 | Kilometers to meters |
| `AU_TO_M` | 1.496 × 10¹¹ m | Astronomical Unit to meters |
| `SEC_TO_DAY` | 86400.0 | Seconds to days |

## Dependencies

- **NumPy** — Array operations, linear algebra, trigonometric functions
- **SciPy** — Root-finding (`brentq`), ODE integration (`solve_ivp`), statistics
- **Matplotlib** — Trajectory visualization
- **Astropy** — Solar system ephemerides and unit conversions

## Body Data Format

Each body loaded at runtime has the following structure:

```json
{
  "id": "earth",
  "name": "Earth",
  "semimajorAxis": 149598023,
  "perihelion": 147095000,
  "aphelion": 152100000,
  "eccentricity": 0.0167,
  "apogee": 152100000,
  "perigee": 147095000
}
```

- `apogee` and `perigee` are computed from `semimajorAxis × (1 ± eccentricity)`
- Legacy aliases like `terre` and `saturne` are still supported
