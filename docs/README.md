# foslas Documentation

**foslas** (Fast Orbital Solar System Lambert Solver) is a Python library and CLI tool for computing interplanetary transfer trajectories between celestial bodies in the solar system.

## Quick Start

```bash
# Install dependencies
pip install numpy scipy matplotlib requests python-dotenv

# Fetch celestial body data
python scripts/fetch_data.py

# Compute transfer statistics
foslas stats terre mars -d 15

# Generate trajectory plot
foslas plot terre mars -d 15 -o mars_transfer.png

# List available bodies
foslas list -s jupiter
```

## Documentation

### System Architecture
- [System Overview](system.md) — Project structure, data flow, CLI commands

### Orbital Mechanics
- [Orbital Mechanics](orbital-mechanics.md) — Orbital elements, Hohmann transfers, energy factor scaling

### Core Algorithms
- [Lambert's Problem](lambert-problem.md) — Lambert solver, Stumpff functions, transfer search
- [Numerical Integration](numerical-integration.md) — Two-body ODE, DOP853 method, accuracy

### Visualization
- [Visualization](visualization.md) — Matplotlib plotting, orbit rotation, color scheme

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        foslas                                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │  constants   │    │   lambert   │    │  integrator │        │
│  │  (SI units)  │    │  (Stumpff)  │    │  (DOP853)   │        │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘        │
│         │                  │                  │                 │
│         └──────────────────┼──────────────────┘                 │
│                            │                                    │
│                     ┌──────▼──────┐                            │
│                     │   orbital   │                            │
│                     │  (Hohmann,  │                            │
│                     │   Lambert)  │                            │
│                     └──────┬──────┘                            │
│                            │                                    │
│                     ┌──────▼──────┐                            │
│                     │     viz     │                            │
│                     │ (matplotlib)│                            │
│                     └─────────────┘                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Mathematical Foundations

### Hohmann Transfer

The minimum-energy two-impulse transfer between circular orbits:

```
Δv_total = Δv_departure + Δv_arrival

Δv_departure = √(GM × (2/r1 - 1/a_t)) - √(GM/r1)
Δv_arrival = √(GM/r2) - √(GM × (2/r2 - 1/a_t))

a_t = (r1 + r2) / 2
```

### Lambert's Problem

Given two positions and time of flight, find the velocities:

```
v₁ = (r₂ - f × r₁) / g
v₂ = (ġ × r₂ - r₁) / g

where f, g, ġ are Lagrange coefficients computed from Stumpff functions
```

### Vis-Viva Equation

Velocity at any point in an elliptical orbit:

```
v = √(GM × (2/r - 1/a))
```

## Data Sources

- Celestial body data: [le-systeme-solaire.net](https://api.le-systeme-solaire.net)
- Energy data: [Our World in Data](https://github.com/owid/energy-data)

## References

- Bate, R. R., Mueller, D. D., & White, J. E. (1971). *Fundamentals of Astrodynamics*. Dover.
- Curtis, H. D. (2013). *Orbital Mechanics for Engineering Students*. Butterworth-Heinemann.
- Vallado, D. A. (2013). *Fundamentals of Astrodynamics and Applications*. Microcosm Press.
