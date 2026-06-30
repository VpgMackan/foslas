"""Shared utility functions for body resolution.

Provides common functions used by both CLI and UI modules.
"""

from .bodies import load_planet_bodies, load_asteroid_body, ASTEROID_CATALOG


def find_body(bodies, body_id):
    """Find a body by its ID (case-insensitive).

    Parameters
    ----------
    bodies : list of dict
        List of body data dictionaries.
    body_id : str
        The body ID to search for.

    Returns
    -------
    dict or None
        The body data dictionary, or None if not found.
    """
    body_id = body_id.strip().lower()
    return next(
        (
            b
            for b in bodies
            if body_id
            in {
                b.get("id", "").lower(),
                b.get("englishName", "").lower(),
                *[a.lower() for a in b.get("aliases", [])],
            }
        ),
        None,
    )


def find_asteroid(body_id):
    """Find an asteroid by ID or englishName in the asteroid catalog.

    Parameters
    ----------
    body_id : str
        Body ID or name to search for.

    Returns
    -------
    dict or None
        The asteroid body data, or None if not found.
    """
    body_id_lower = body_id.strip().lower()
    for aid, data in ASTEROID_CATALOG.items():
        if body_id_lower == aid:
            return load_asteroid_body(aid)
        if body_id_lower == data.get("englishName", "").lower():
            return load_asteroid_body(aid)
    return None


def resolve_bodies(start_id, end_id):
    """Resolve body IDs to OrbitalBody objects.

    Parameters
    ----------
    start_id : str
        Departure body ID.
    end_id : str
        Arrival body ID.

    Returns
    -------
    tuple
        (start_data, end_data, start_ob, end_ob) where data are dicts
        and ob are OrbitalBody objects.
    """
    from .transfers.base import OrbitalBody

    bodies = load_planet_bodies()

    start = find_asteroid(start_id)
    if start is None:
        start = find_body(bodies, start_id)

    end = find_asteroid(end_id)
    if end is None:
        end = find_body(bodies, end_id)

    if not start:
        raise ValueError(f"Body '{start_id}' not found.")
    if not end:
        raise ValueError(f"Body '{end_id}' not found.")

    try:
        start_ob = OrbitalBody(start["aphelion"], start["perihelion"])
        end_ob = OrbitalBody(end["aphelion"], end["perihelion"])
    except ValueError:
        raise ValueError("Invalid orbit parameters for the selected bodies.")

    return start, end, start_ob, end_ob


__all__ = [
    "find_body",
    "find_asteroid",
    "resolve_bodies",
]