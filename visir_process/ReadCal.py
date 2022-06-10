import numpy as np 
from scipy.interpolate import interp1d

from BinningInputs import BinningInputs

def ReadCal(filename):
    """Read calibration file and return array of profiles"""

    # Open file
    with open(filename) as f:
        # Read contents
        lines = f.readlines()
        # Save number of filters, Voyager samples, and Cassini samples
        nfilt = int(lines[0])-1     # ignore 0 element (latitude), keep 1-15 (filters)
        niris = int(lines[1])
        ncirs = int(lines[2])-3     # cut off the 60N-60S Q band lines at the end
        # Define temporary variables (needed because of irregular formatting of calfile)
        ilat, lat_count = 0, 0
        newline, caldata = [], []
        for iline, line in enumerate(lines[3:]):
            if lat_count < 3:
                # Gather spectra at each latitude
                l = line.split()
                [newline.append(il) for il in l]
                lat_count += 1
            if lat_count == 3:
                # Store spectral information
                caldata.append(newline)
                # Reset temporary variables
                lat_count, newline = 0, []
                ilat += 1
    # Separate spacecraft profiles (converted to units of W ..)
    irisdata = np.asarray(caldata[0:niris-1], dtype='float')
    cirsdata = np.asarray(caldata[niris:niris+ncirs-1], dtype='float')
    # Interpolate spacecraft profiles onto "lat_grid" (common with data)
    iris = np.zeros((BinningInputs.Nlatbins, nfilt, 2))
    cirs = np.zeros((BinningInputs.Nlatbins, nfilt, 2))
    for ifilt in range(nfilt):
        # Interp IRIS to VISIR
        f = interp1d(irisdata[:, 0], irisdata[:, ifilt+1])
        lmin, lmax = np.min(irisdata[:, 0]), np.max(irisdata[:, 0])
        keep = (BinningInputs.latgrid > lmin) & (BinningInputs.latgrid < lmax)
        # Store interpolated profile
        iris[keep, ifilt, 0] = BinningInputs.latgrid[keep]
        iris[keep, ifilt, 1] = f(BinningInputs.latgrid[keep]) * 10E-8
        # Interp CIRS to VISIR
        f = interp1d(cirsdata[:, 0], cirsdata[:, ifilt+1])
        lmin, lmax = np.min(cirsdata[:, 0]), np.max(cirsdata[:, 0])
        keep = (BinningInputs.latgrid > lmin) & (BinningInputs.latgrid < lmax)
        # Store interpolated profile
        cirs[keep, ifilt, 0] = BinningInputs.latgrid[keep]
        cirs[keep, ifilt, 1] = f(BinningInputs.latgrid[keep]) * 10E-8

    # Throw away zeros
    iris[iris == 0] = np.nan
    cirs[cirs == 0] = np.nan

    return iris, cirs