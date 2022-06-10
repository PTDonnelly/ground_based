import numpy as np
from math import acos, cos, radians, pi

from BinningInputs import BinningInputs
from ReadFits import ReadFits
from SetWave import SetWave
from CalculateErrors import CalculateErrors

def RegisterMaps(files):
    """ Step 1: Read img, cmap and mufiles
        Step 2: Geometric registration of pixel information
        Step 3: Gather pixel information for all files"""
    
    # Define local inputs
    nx, ny = 720, 360                  # Dimensions of an individual cylindrical map (needed for dictionary definition)
    res    = ny / 180                  # Resolution of maps: res = 1 (1 degree), res = 2 (0.5 degree) etc.
    Nfiles = len(files)

    # Create np.array for all pixels in all cmaps and mumaps
    spectrum = np.empty((ny, nx, Nfiles, 7))

    # Define arrays
    viewing_mode   = np.empty(Nfiles)
    wavelength     = np.empty(Nfiles)
    wavenumber     = np.empty(Nfiles)
    LCMIII         = np.empty(Nfiles)

    # Define flags
    pg2pc = 0                   # Optional conversion of latitudes from planetographic to planetocentric

    # Loop over files
    for ifile, fname in enumerate(files):
        print(ifile, fname)
        ## Step 1: Read img, cmap and mufiles
        imghead, imgdata, cylhead, cyldata, muhead, mudata = ReadFits(filename=f"{fname}")

        ## Step 2: Geometric registration of pixel information
        # Save flag depending on Northern (1) or Southern (-1) viewing
        chopang = imghead['HIERARCH ESO TEL CHOP POSANG']
        posang  = imghead['HIERARCH ESO ADA POSANG'] + 360
        view = 1 if chopang == posang else -1
        viewing_mode[ifile] = view
        
        # Store central meridian longitude
        LCMIII[ifile] = cylhead['LCMIII']

        # Assign spatial information to pixels
        naxis1    = cylhead['NAXIS1']
        naxis2    = cylhead['NAXIS2']
        naxis1_mu = muhead['NAXIS1']
        naxis2_mu = muhead['NAXIS2']

        # Set the central wavelengths for each filter. Must be
        # identical to the central wavelength specified for the
        # production of the k-tables
        wavelen, wavenum, _, _  = SetWave(wavelength=cylhead['lambda'], wavenumber=False)
        wavelength[ifile] = wavelen
        wavenumber[ifile] = wavenum
        
        # Loop over each pixel to assign to the structure.
        xstart  = float(naxis1) - BinningInputs.lonrange[0]/(360/naxis1)
        xstop   = float(naxis1) - BinningInputs.lonrange[1]/(360/naxis1) 
        ystart  = (float(naxis2)/2) + BinningInputs.latrange[0]/(180/naxis2)
        ystop   = (float(naxis2)/2) + BinningInputs.latrange[1]/(180/naxis2) 
        x_range = np.arange(xstart, xstop, 1)
        y_range = np.arange(ystart, ystop, 1)
        for ix, x in enumerate(x_range):
            for iy, y in enumerate(y_range): 
                # Only assign latitude and longitude if non-zero pixel value
                if (cyldata[iy, ix] > 0):
                    # Calculate finite spatial element (lat-lon co-ordinates)
                    lat = BinningInputs.latrange[0] + ((180 / naxis2) * y)
                    lon = BinningInputs.lonrange[0] - ((360 / naxis1) * x)
                    # Adjust co-ordinates from edge to centre of bins
                    lat = lat + BinningInputs.latstep/res
                    lon = lon - BinningInputs.latstep/res
                    # Convert from planetographic to planetocentric latitudes
                    mu_ang = mudata[iy, ix]
                    mu  = 180/pi * acos(mu_ang)
                    # Calculate pixel radiance and error
                    rad = cyldata[iy, ix] * 1e-7
                    rad_error = CalculateErrors(imgdata)
                    
                    ## Step 3: Gather pixel information for all files
                    # Store spectral information in spectrum array
                    spectrum[iy, ix, ifile, 0] = lat
                    spectrum[iy, ix, ifile, 1] = LCMIII[ifile]
                    spectrum[iy, ix, ifile, 2] = mu
                    spectrum[iy, ix, ifile, 3] = rad
                    spectrum[iy, ix, ifile, 4] = rad_error * rad
                    spectrum[iy, ix, ifile, 5] = wavenum
                    spectrum[iy, ix, ifile, 6] = view
    # Throw away zeros
    spectrum[spectrum == 0] = np.nan

    return spectrum, wavelength, wavenumber, LCMIII