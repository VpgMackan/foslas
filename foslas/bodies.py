import numpy as np

from .constants import GM_SUN, AU_TO_M
from .math_utils import solve_kepler, true_anomaly_from_eccentric_anomaly

ASTEROID_CATALOG = {
    "16_psyche": {
        "id": "16_psyche_a852_fa",
        "number": 20000016,
        "englishName": "16 Psyche",
        "a_au": 2.925720466462538,
        "ecc": 0.1349324738201893,
        "inc_deg": 3.098749116151128,
        "omega_deg": 149.9753859305033,
        "w_deg": 230.0326782748359,
        "M0_deg": 79.76939505329617,
        "epoch_jd": 2461200.5,
    },
    "1_ceres": {
        "id": "1_ceres_a801_aa",
        "number": 20000001,
        "englishName": "1 Ceres",
        "a_au": 2.765552595034094,
        "ecc": 0.07969229514816586,
        "inc_deg": 10.58802780183462,
        "omega_deg": 80.24862682043221,
        "w_deg": 73.29421453021587,
        "M0_deg": 274.4193463761342,
        "epoch_jd": 2461200.5,
    },
    "4_vesta": {
        "id": "4_vesta_a807_fa",
        "number": 20000004,
        "englishName": "4 Vesta",
        "a_au": 2.361365965127599,
        "ecc": 0.09020374382834395,
        "inc_deg": 7.143925545058711,
        "omega_deg": 103.701293265032,
        "w_deg": 151.4686478221564,
        "M0_deg": 81.19015607686903,
        "epoch_jd": 2461200.5,
    },
    "2_pallas": {
        "id": "2_pallas_a802_fa",
        "number": 20000002,
        "englishName": "2 Pallas",
        "a_au": 2.769559010737709,
        "ecc": 0.2307000995648547,
        "inc_deg": 34.93279321851542,
        "omega_deg": 172.8866193357694,
        "w_deg": 310.9699161652136,
        "M0_deg": 254.2496521742734,
        "epoch_jd": 2461200.5,
    },
    "15308_ulfdanielsson": {
        "id": "15308_ulfdanielsson_1993_fr4",
        "number": 20015308,
        "englishName": "15308 Ulfdanielsson",
        "a_au": 2.351286131674873,
        "ecc": 0.2066265477026832,
        "inc_deg": 3.222774791479786,
        "omega_deg": 152.4960163532508,
        "w_deg": 115.6768554877572,
        "M0_deg": 13.24656029940937,
        "epoch_jd": 2461200.5,
    },
    "15312_wandt": {
        "id": "15312_wandt",
        "number": 20015312,
        "englishName": "15312 Wandt",
        "a_au": 2.740847284800487,
        "ecc": 0.1434194761508568,
        "inc_deg": 5.836486522366694,
        "omega_deg": 339.2339598289466,
        "w_deg": 22.38174660721835,
        "M0_deg": 299.2301069881405,
        "epoch_jd": 2461200.5,
    },
}


PLANET_SPECS = [
    ("mercury", "mercury", "Mercury", ["mercure"]),
    ("venus", "venus", "Venus", []),
    ("earth", "earth", "Earth", ["terre"]),
    ("mars", "mars", "Mars", []),
    ("jupiter", "jupiter", "Jupiter", []),
    ("saturn", "saturn", "Saturn", ["saturne"]),
    ("uranus", "uranus", "Uranus", []),
    ("neptune", "neptune", "Neptune", []),
]


def keplerian_to_state(a_au, ecc, inc_deg, omega_deg, w_deg, M0_deg, epoch_jd, t_jd):
    """Convert Keplerian elements to state vectors.

    Parameters
    ----------
    a_au : float
        Semi-major axis in AU.
    ecc : float
        Eccentricity.
    inc_deg : float
        Inclination in degrees.
    omega_deg : float
        Longitude of ascending node in degrees.
    w_deg : float
        Argument of perihelion in degrees.
    M0_deg : float
        Mean anomaly at epoch in degrees.
    epoch_jd : float
        Epoch Julian date.
    t_jd : float
        Target Julian date.

    Returns
    -------
    tuple
        (r_au, lon_rad) - distance in AU and longitude in radians.
    """
    a = a_au * AU_TO_M
    inc = np.radians(inc_deg)
    omega = np.radians(omega_deg)
    w = np.radians(w_deg)

    dt_days = t_jd - epoch_jd
    mean_motion = np.sqrt(GM_SUN / a**3) * 86400.0
    M = np.radians(M0_deg) + mean_motion * dt_days

    E = solve_kepler(M, ecc)

    r = a * (1 - ecc * np.cos(E))
    nu = true_anomaly_from_eccentric_anomaly(E, ecc)

    x_orb = r * np.cos(nu)
    y_orb = r * np.sin(nu)
    z_orb = 0.0

    cos_w, sin_w = np.cos(w), np.sin(w)
    cos_inc, sin_inc = np.cos(inc), np.sin(inc)
    cos_omega, sin_omega = np.cos(omega), np.sin(omega)

    x = (cos_omega * cos_w - sin_omega * cos_inc * sin_w) * x_orb + (
        -cos_omega * sin_w - sin_omega * cos_inc * cos_w
    ) * y_orb
    y = (sin_omega * cos_w + cos_omega * cos_inc * sin_w) * x_orb + (
        -sin_omega * sin_w + cos_omega * cos_inc * cos_w
    ) * y_orb
    z = sin_inc * sin_w * x_orb + sin_inc * cos_w * y_orb

    lon = np.arctan2(y, x)
    lat = np.arcsin(z / np.sqrt(x**2 + y**2 + z**2))
    r_au = np.sqrt(x**2 + y**2 + z**2) / AU_TO_M

    return r_au, lon


def compute_asteroid_ephemeris(asteroid_id, t_jd):
    """Compute asteroid position at a given Julian date.

    Parameters
    ----------
    asteroid_id : str
        Asteroid identifier (lowercase, e.g., 'ceres').
    t_jd : float
        Julian date.

    Returns
    -------
    tuple
        (r_au, lon_rad) - distance in AU and longitude in radians.
    """
    if asteroid_id not in ASTEROID_CATALOG:
        raise ValueError(f"Asteroid '{asteroid_id}' not found in catalog.")

    ast = ASTEROID_CATALOG[asteroid_id]
    return keplerian_to_state(
        ast["a_au"],
        ast["ecc"],
        ast["inc_deg"],
        ast["omega_deg"],
        ast["w_deg"],
        ast["M0_deg"],
        ast["epoch_jd"],
        t_jd,
    )


def load_asteroid_body(asteroid_id):
    """Load asteroid body data for transfer calculations.

    Parameters
    ----------
    asteroid_id : str
        Asteroid identifier (lowercase).

    Returns
    -------
    dict
        Body data dictionary with aphelion, perihelion, etc.
    """
    if asteroid_id not in ASTEROID_CATALOG:
        return None

    ast = ASTEROID_CATALOG[asteroid_id]
    a_au = ast["a_au"]
    ecc = ast["ecc"]
    aphelion_au = a_au * (1 + ecc)
    perihelion_au = a_au * (1 - ecc)

    from .constants import AU_TO_KM

    return {
        "id": asteroid_id,
        "englishName": ast["englishName"],
        "semimajorAxis": a_au * AU_TO_KM,
        "eccentricity": ecc,
        "aphelion": aphelion_au * AU_TO_KM,
        "perihelion": perihelion_au * AU_TO_KM,
        "apogee": aphelion_au * AU_TO_KM,
        "perigee": perihelion_au * AU_TO_KM,
    }


def load_planet_bodies(day_offset=0):
    """Load celestial body data from Astropy ephemerides.

    Parameters
    ----------
    day_offset : int, optional
        Days offset from current time (default: 0).

    Returns
    -------
    list of dict
        List of body data dictionaries.
    """
    try:
        import astropy.units as u
        from astropy.constants import G as G_ASTRO, M_sun
        from astropy.coordinates import get_body_barycentric_posvel
        from astropy.time import Time, TimeDelta
    except ModuleNotFoundError:
        raise ImportError("astropy is required. Install with: pip install astropy")

    def _elements_from_state(r_m, v_m_s, mu):
        r = np.linalg.norm(r_m)
        v = np.linalg.norm(v_m_s)
        h = np.cross(r_m, v_m_s)
        e_vec = np.cross(v_m_s, h) / mu - (r_m / r)
        ecc = float(np.linalg.norm(e_vec))
        eps = v * v / 2.0 - mu / r
        if eps >= 0:
            return None
        sma = -mu / (2.0 * eps)
        return float(sma), ecc

    epoch = Time.now() + TimeDelta(day_offset, format='jd')
    sun_pos, sun_vel = get_body_barycentric_posvel("sun", epoch)
    sun_pos = sun_pos.xyz.to(u.m).value
    sun_vel = sun_vel.xyz.to(u.m / u.s).value
    mu_sun = (G_ASTRO * M_sun).to_value(u.m**3 / u.s**2)

    bodies = []
    for body_id, astropy_name, english_name, aliases in PLANET_SPECS:
        try:
            body_pos, body_vel = get_body_barycentric_posvel(astropy_name, epoch)
            r_vec = body_pos.xyz.to(u.m).value - sun_pos
            v_vec = body_vel.xyz.to(u.m / u.s).value - sun_vel
            elements = _elements_from_state(r_vec, v_vec, mu_sun)
            if elements is None:
                continue
            sma_m, ecc = elements
            sma_km = sma_m / 1000.0
            aphelion = sma_km * (1.0 + ecc)
            perihelion = sma_km * (1.0 - ecc)
            bodies.append(
                {
                    "id": body_id,
                    "englishName": english_name,
                    "aliases": aliases,
                    "semimajorAxis": sma_km,
                    "eccentricity": ecc,
                    "aphelion": aphelion,
                    "perihelion": perihelion,
                    "apogee": aphelion,
                    "perigee": perihelion,
                }
            )
        except Exception:
            continue

    if not bodies:
        raise RuntimeError("Failed to load bodies from Astropy ephemerides.")

    return bodies
