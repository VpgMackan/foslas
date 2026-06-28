import numpy as np

BIG_G = 6.67 * 10 ** (-11)
BIG_M = 1.989 * 10**30


# TODO (fuck)
def calc_kep_3(semi_major_axis=None, orbital_period=None) -> float:
    if orbital_period:
        return (orbital_period**2) ** (1 / 3)
    elif semi_major_axis:
        return np.sqrt(semi_major_axis**3)


def calc_vis_viva(current_radius: float, semi_major_axis: float) -> float:
    return np.sqrt(BIG_G * BIG_M * (2 / current_radius - 1 / semi_major_axis))


def calc_cen_acceleration(current_radius: float) -> float:
    return (BIG_G * BIG_M) / (current_radius**2)


def calc_eccentricity(apogee: float, perigee: float) -> float:
    """
    Beräknar excentriciteten för en elliptisk omloppsbana.

    Args:
        apogee:  Avstånd vid apogeepunkten (från kroppens centrum)
        perigee: Avstånd vid perigeepunkten (från kroppens centrum)

    Returns:
        Excentricitet e (0 = cirkel, 0 < e < 1 = ellips)
    """
    if perigee <= 0 or apogee < perigee:
        raise ValueError("Ogiltiga banparametrar: apogee >= perigee > 0 krävs")

    return (apogee - perigee) / (apogee + perigee)


def calc_semi_minor_axis(eccentricity: float, semi_major_axis: float) -> float:
    return np.sqrt((semi_major_axis**2) / (eccentricity - 1))


def calc_area(semi_minor_axis: float, semi_major_axis: float) -> float:
    """
    Calc area, not that complicated bro. You can figure it out yourself
    """
    return 2 * np.pi * semi_minor_axis * semi_major_axis


def calc_ref_point(apogee: float, perigee: float) -> float:
    """
    I don't know bro, I don't ork

    Args:
        fuck if i know

    Returns:
        something sometimes
    """
    if perigee <= 0:
        raise ValueError("perigee måste vara > 0")
    if apogee < perigee:
        raise ValueError("apogee måste vara >= perigee")

    return apogee - perigee


def calc_angle(target_orbit: float, apogee: float, perigee: float) -> float:
    """
    Computes the satellite's perpendicular height above the major axis,
    using the law of cosines on the triangle formed by the two foci
    and the satellite's current position on the ellipse.

    Args:
        target_orbit: Orbital radius from the central body [km]
        apogee:       Apogee distance from the central body  [km]
        perigee:      Perigee distance from the central body [km]

    Returns:
        Perpendicular distance from the major axis [km]
    """
    focal_distance = calc_ref_point(apogee, perigee)  # d = 2c
    r = target_orbit
    major_axis = apogee + perigee  # 2a

    # Law of cosines: resolve angle θ at the occupied focus
    cos_theta = (r**2 + focal_distance**2 - major_axis**2) / (2 * r * focal_distance)

    return np.arccos(cos_theta)


def calc_area_sector(
    cos_theta: float, r: float, apogee: float, perigee: float
) -> float:
    focal_distance = calc_ref_point(apogee, perigee)
    return (
        np.arctan(r * np.tan(cos_theta) / focal_distance) * r * focal_distance * 1 / 2
    )


def calc_delta_orbital_period(orbital_period: float, area_sector: float, area: float):
    return (orbital_period * area_sector) / area


def calc_delta_theta_ship(orbital_period: float) -> float:
    return 360 / (orbital_period * 365)


def calc_delta_theta_body(orbital_period_body: float) -> float:
    return 360 / (orbital_period_body * 365)


if __name__ == "__main__":
    delta_v = float(input("Enter your delta V: "))
    time_to_apogee = float(input("Enter time to apogee: "))
    angle_between_obj = float(input("Enter angle between objects: "))
    start_body = input("Enter your start body id: ")
    end_body = input("Enter your end body id: ")
