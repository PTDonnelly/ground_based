import numpy as np
import math
from scipy.interpolate import interp1d

def ReadGravity(filename, lat):
    """ DB: Function to load Jupiter gravity data
            and calculate some planet parameters """
    
    # Load Jupiter gravity data to calculate pseudo-
    # windshear using TB and mu array array
    lines = np.loadtxt(filename)
    g_lat=lines[:,0]
    g_grav=lines[:,2]
    f = interp1d(g_lat, g_grav, kind='cubic')
    inlat=lat
    inlat=inlat.flatten()
    grav=f(inlat)
    # Radians to degree conversion (and reverse)
    deg2rad=math.pi/180.
    rad2deg=180./math.pi
    # Calculate the north-south distance in m
    xellip=1.06937
    xradius = 7.14920e+07 # Radius in m
    # Calculate cosin and sin of latitude
    clatc = np.cos(lat*deg2rad)
    slatc = np.sin(lat*deg2rad)
    # Rr is the ratio of radius at equator to radius at current latitude
    Rr=np.sqrt(clatc**2 + (xellip**2 * slatc**2))
    radius = (xradius/Rr)
    y=radius*lat*deg2rad
    y=y/1e3 # Convert from m to km
    # Planet period
    period=0.41354*24.*60.*60.
    # Planet rotation 
    omega=2.*math.pi/period
    Coriolis = 2. * omega * np.sin(lat*deg2rad)
    
    return grav, Coriolis, y, inlat, rad2deg, omega 
