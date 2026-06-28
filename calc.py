import json
import numpy as np

G = 6.67430e-11  # m^3 kg^-1 s^-2
M = 1.98847e30  # Solens massa (kg)
GM = G * M


# TODO (fuck)
def calc_kep_3(semi_major_axis=None, orbital_period=None) -> float:
    if orbital_period:
        return ((orbital_period**2) / ((4 * np.pi**2) / (GM))) ** (1 / 3)
    elif semi_major_axis:
        return np.sqrt(((4 * np.pi**2) / GM) * semi_major_axis**3)


class OrbitalStructure:
    def __init__(
        self,
        apogee: float,
        perigee: float,
    ):
        self.apogee = apogee
        self.perigee = perigee
        self.current_radius = (self.apogee + self.perigee) / 2

        if perigee <= 0 or apogee < perigee:
            raise ValueError("Ogiltiga banparametrar: apogee >= perigee > 0 krävs")
        self.eccentricity = (apogee - perigee) / (apogee + perigee)

        self.semi_major_axis = (self.apogee + self.perigee) / 2
        self.semi_minor_axis = self.semi_major_axis * np.sqrt(1 - self.eccentricity**2)
        self.vis_viva_velocity = np.sqrt(
            G * M * (2 / self.current_radius - 1 / self.semi_major_axis)
        )
        self.acceleration = (G * M) / (self.current_radius**2)

        self.area = np.pi * self.semi_minor_axis * self.semi_major_axis

        if self.perigee <= 0:
            raise ValueError("perigee måste vara > 0")
        elif self.apogee < self.perigee:
            raise ValueError("apogee måste vara >= perigee")
        self.focal_distance = (self.apogee - self.perigee) / 2

    def calc_angle(self, target_orbit: float) -> float:
        """
        Computes the satellite's perpendicular height above the major axis,
        using the law of cosines on the triangle formed by the two foci
        and the satellite's current position on the ellipse.

        Args:
            target_orbit: Orbital radius from the central body [km]

        Returns:
            Perpendicular distance from the major axis [km]
        """
        major_axis = self.apogee + self.perigee  # 2a

        # Law of cosines: resolve angle θ at the occupied focus
        cos_theta = (
            target_orbit**2 + self.focal_distance**2 - (major_axis - target_orbit) ** 2
        ) / (2 * target_orbit * self.focal_distance)

        return np.arccos(cos_theta)

    def calc_area_sector(self, cos_theta: float, r: float) -> float:
        return (
            np.arctan(r * np.tan(cos_theta) / self.focal_distance)
            * r
            * self.focal_distance
            * 1
            / 2
        )

    def calc_delta_orbital_period(
        self, orbital_period: float, area_sector: float, area: float
    ):
        return (orbital_period * area_sector) / area

    def calc_delta_theta(self, orbital_period: float, days=365) -> float:
        return (2 * np.pi) / (orbital_period * days)


def total_delta_v(delta_v, r1, r2):
    """
    Beräknar summan av två Delta V.

    Parametrar:
        delta_v : testvärde på första Delta V (m/s)
        r1      : första banradien (m)
        r2      : andra banradien (m)

    Returnerar:
        Total Delta V (m/s)
    """

    # Cirkulära banhastigheter
    v1 = np.sqrt(GM / r1)
    v2 = np.sqrt(GM / r2)

    # Hastighet efter första manövern
    v1_new = v1 + delta_v

    # Halvstoraxeln efter första manövern
    a = 1 / (2 / r1 - v1_new**2 / GM)

    # Hastigheter på transferbanan
    vt1 = np.sqrt(GM * (2 / r1 - 1 / a))
    vt2 = np.sqrt(GM * (2 / r2 - 1 / a))

    # Total Delta V
    DV = abs(vt1 - v1) + abs(v2 - vt2)

    return DV


if __name__ == "__main__":
    delta_v = float(input("Enter your delta V: "))
    angle_between_obj = float(input("Enter angle between objects: "))

    dataset = json.load(open("./data.json"))["bodies"]
    start_body_id = input("Enter your start body id: ")
    start_body = next((x for x in dataset if x["id"] == start_body_id), None)

    end_body_id = input("Enter your end body id: ")
    end_body = next((x for x in dataset if x["id"] == end_body_id), None)

    start_orbital_structure = OrbitalStructure(
        start_body["apogee"],
        start_body["perigee"],
    )

    end_orbital_structure = OrbitalStructure(
        end_body["apogee"],
        end_body["perigee"],
    )

    print(total_delta_v(delta_v, r1, r2))
