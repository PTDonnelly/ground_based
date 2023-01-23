import numpy as np
from Read.ReadGravity import ReadGravity

def CalculateWindShear(temperature, latitude, nlat, nlevel):

    windshear = np.empty((nlevel, nlat))
    # Load Jupiter gravity data to calculate pseudo-windshear using TB and mu array array
    grav, Coriolis, y, _, _, _ = ReadGravity("../inputs/jup_grav.dat", lat=latitude)

    # Calculated the associated windshear from the zonal mean retrieved temperature
    for ilevel in range(nlevel):
        windshear[ilevel,:]=-(grav/(Coriolis*temperature[ilevel,:]))*np.gradient(temperature[ilevel, :],y)

    return windshear