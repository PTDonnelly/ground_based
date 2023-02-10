import numpy as np

def convert_to_planetocentric(lat_planetographic, planet):
    """ Conversion planetographic to planetocentric latitude """
    
    # Ensuring to work with arrays, whereas numpy is grumpy 
    lat_planetographic = np.asarray(lat_planetographic, dtype='float')
    # Retrieve polar and equatorial radius for the current planet
    R_pole, R_equator = planet_polar_equatorial_radius_ratio(planet=planet)
    # Conversion
    e = float((R_equator/R_pole)**2)
    lat_planetocentric = np.arctan(np.tan(lat_planetographic*np.pi/180.)/e)*180./np.pi

    return lat_planetocentric

def convert_to_planetographic(lat_planetocentric, planet):
    """ Conversion planetocentric to planetographic latitude """

    # Ensuring to work with arrays, whereas numpy is grumpy
    lat_planetocentric = np.asarray(lat_planetocentric, dtype='float')
    # Retrieve polar and equatorial radius for the current planet
    R_pole, R_equator = planet_polar_equatorial_radius_ratio(planet=planet)
    # Conversion
    e = float((R_equator/R_pole)**2)
    lat_planetographic = np.arctan(np.tan(lat_planetocentric*np.pi/180.)*e)*180./np.pi

    return lat_planetographic

def planet_polar_equatorial_radius_ratio(planet):

    if planet == 'jupiter':
        R_pol = 66854.
        R_eq  = 71492.

    if planet == 'saturn':
        R_pol = 54364.
        R_eq  = 60268.
    
    return R_pol, R_eq