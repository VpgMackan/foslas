import numpy as np

SUN_GRAVITATIONAL_CONSTANT = 6.67e-11  # m^3 kg^-1 s^-2
SUN_MASS = 1.989e30


# TODO (fuck)
def calc_kep_3(semi_major_axis=None, orbital_period=None) -> float:
    if orbital_period:
        return (orbital_period**2) ** (1 / 3)
    elif semi_major_axis:
        return np.sqrt(semi_major_axis**3)


class OrbitalStructure:
    def __init__(
        self,
        semi_major_axis: float,
        current_radius: float,
        apogee: float,
        perigee: float,
    ):
        self.semi_major_axis = semi_major_axis
        self.current_radius = current_radius
        self.apogee = apogee
        self.perigee = perigee

        if perigee <= 0 or apogee < perigee:
            raise ValueError("Ogiltiga banparametrar: apogee >= perigee > 0 krävs")
        else:
            self.eccentricity = (apogee - perigee) / (apogee + perigee)

        self.semi_minor_axis = np.sqrt(
            (self.semi_major_axis**2) / (self.eccentricity - 1)
        )
        self.vis_viva_velocity = np.sqrt(
            SUN_GRAVITATIONAL_CONSTANT
            * SUN_MASS
            * (2 / self.current_radius - 1 / self.semi_major_axis)
        )
        self.acceleration = (SUN_GRAVITATIONAL_CONSTANT * SUN_MASS) / (
            self.current_radius**2
        )

        self.area = 2 * np.pi * self.semi_minor_axis * self.semi_major_axis

        if self.perigee <= 0:
            raise ValueError("perigee måste vara > 0")
        elif self.apogee < self.perigee:
            raise ValueError("apogee måste vara >= perigee")
        else:
            self.focal_distance = self.apogee - self.perigee

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
        cos_theta = (target_orbit**2 + self.focal_distance**2 - major_axis**2) / (
            2 * target_orbit * self.focal_distance
        )

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
        return 360 / (orbital_period * days)


if __name__ == "__main__":
    delta_v = float(input("Enter your delta V: "))
    time_to_apogee = float(input("Enter time to apogee: "))
    angle_between_obj = float(input("Enter angle between objects: "))
    start_body = input("Enter your start body id: ")
    end_body = input("Enter your end body id: ")
