"""Physical constants and unit conversion factors used throughout the library."""

G = 6.67430e-11
M_SUN = 1.98847e30
GM_SUN = G * M_SUN
KM_TO_M = 1000.0
AU_TO_M = 1.496e11
AU_TO_KM = AU_TO_M / KM_TO_M
SEC_TO_DAY = 86400.0
J2000_JD = 2451545.0
JD_EPOCH_OFFSET = 2440587.5
# Transfer computation thresholds
HOHMANN_DV_TOLERANCE = 1.0  # m/s - tolerance for Hohmann vs fast comparison
CIRCULAR_ECC_TOLERANCE = 1e-10  # eccentricity threshold for circular orbits
