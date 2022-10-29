import numpy as np
from math import acos, cos, radians, pi
import Globals
from Read.ReadFits import ReadFits
from Tools.SetWave import SetWave
from Tools.CalculateErrors import CalculateErrors

# def RegisterMaps(files):
#     """ Step 1: Read img, cmap and mufiles
#         Step 2: Geometric registration of pixel information
#         Step 3: Gather pixel information for all files"""
    
#     print('Registering maps...')
    
#     # Define local inputs
#     nx, ny = 720, 360                  # Dimensions of an individual cylindrical map (needed for dictionary definition)
#     res    = ny / 180                  # Resolution of maps: res = 1 (1 degree), res = 2 (0.5 degree) etc.
#     Nfiles = len(files)

#     # Create np.array for all pixels in all cmaps and mumaps
#     spectrum = np.empty((ny, nx, Nfiles, 7))

#     # Define arrays
#     viewing_mode   = np.empty(Nfiles)
#     wavelength     = np.empty(Nfiles)
#     wavenumber     = np.empty(Nfiles)
#     LCMIII         = np.empty(Nfiles)

#     # Define flags
#     pg2pc = 0                   # Optional conversion of latitudes from planetographic to planetocentric

#     # Loop over files
#     for ifile, fpath in enumerate(files):
#         print(ifile, fpath)
#         ## Step 1: Read img, cmap and mufiles
#         imghead, imgdata, cylhead, cyldata, muhead, mudata = ReadFits(filepath=f"{fpath}")

#         ## Step 2: Geometric registration of pixel information
#         # Save flag depending on Northern (1) or Southern (-1) viewing
#         chopang = imghead['HIERARCH ESO TEL CHOP POSANG']
#         posang  = imghead['HIERARCH ESO ADA POSANG'] + 360
#         view = 1 if chopang == posang else -1
#         viewing_mode[ifile] = view
        
#         # Store central meridian longitude
#         LCMIII[ifile] = cylhead['LCMIII']

#         # Assign spatial information to pixels
#         naxis1    = cylhead['NAXIS1']
#         naxis2    = cylhead['NAXIS2']
#         naxis1_mu = muhead['NAXIS1']
#         naxis2_mu = muhead['NAXIS2']

#         # Set the central wavelengths for each filter. Must be
#         # identical to the central wavelength specified for the
#         # production of the k-tables
#         _, wavelen, wavenum, _, _  = SetWave(filename=fpath, wavelength=cylhead['lambda'], wavenumber=None, ifilt=None)
#         wavelength[ifile] = wavelen
#         wavenumber[ifile] = wavenum

#         # Calulate radiance error
#         rad_error = CalculateErrors(imgdata, view)
        
#         # Loop over each pixel to assign to the structure.
#         xstart  = float(naxis1) - Globals.lonrange[0]/(360/naxis1)
#         xstop   = float(naxis1) - Globals.lonrange[1]/(360/naxis1) 
#         ystart  = (float(naxis2)/2) + Globals.latrange[0]/(180/naxis2)
#         ystop   = (float(naxis2)/2) + Globals.latrange[1]/(180/naxis2) 
#         x_range = np.arange(xstart, xstop, 1, dtype=int)
#         y_range = np.arange(ystart, ystop, 1, dtype=int)
#         for x in x_range:
#             for y in y_range:
#                 # Only assign latitude and longitude if non-zero pixel value
#                 if (cyldata[y, x] > 0):
#                     # Calculate finite spatial element (lat-lon co-ordinates)
#                     lat = Globals.latrange[0] + ((180 / naxis2) * y)
#                     lon = Globals.lonrange[0] - ((360 / naxis1) * x)
#                     # Adjust co-ordinates from edge to centre of bins
#                     lat = lat + Globals.latstep/res
#                     lon = lon - Globals.latstep/res
#                     # Convert from planetographic to planetocentric latitudes
#                     mu_ang = mudata[y, x]
#                     mu  = 180/pi * acos(mu_ang)
#                     # Calculate pxel radiance and error
#                     rad = cyldata[y, x] * 1e-7
                    
#                     ## Step 3: Gather pxel information for all files
#                     # Store spectral information in spectrum array
#                     spectrum[y, x, ifile, 0] = lat
#                     spectrum[y, x, ifile, 1] = LCMIII[ifile]
#                     spectrum[y, x, ifile, 2] = mu
#                     spectrum[y, x, ifile, 3] = rad
#                     spectrum[y, x, ifile, 4] = rad_error * rad
#                     spectrum[y, x, ifile, 5] = wavenum
#                     spectrum[y, x, ifile, 6] = view
#     # Throw away zeros
#     spectrum[spectrum == 0] = np.nan

#     return spectrum, wavelength, wavenumber, LCMIII

def RegisterMaps(files, binning):
    """ Step 1: Read img, cmap and mufiles
        Step 2: Geometric registration of pixel information
        Step 3: Gather pixel information for all files"""
    
    print('Registering maps...')
    
    # Define local inputs
    nx, ny = 720, 360                  # Dimensions of an individual cylindrical map (needed for dictionary definition)
    res    = ny / 180                  # Resolution of maps: res = 1 (1 degree), res = 2 (0.5 degree) etc.
    Nfiles = len(files)

    # Create np.array for all pixels in all cmaps and mumaps
    spectrum = np.empty((ny, nx, Nfiles, 9))
    # Throw away zeros
    spectrum.fill(np.nan)

    # Define arrays
    viewing_mode   = np.empty(Nfiles)
    wavelength     = np.empty(Nfiles)
    wavenumber     = np.empty(Nfiles)
    LCMIII         = np.empty(Nfiles)

    # Define flags
    pg2pc = 0                   # Optional conversion of latitudes from planetographic to planetocentric

    # Loop over files
    for ifile, fpath in enumerate(files):
        print(ifile, fpath)
        ## Step 1: Read img, cmap and mufiles
        imghead, imgdata, cylhead, cyldata, muhead, mudata = ReadFits(filepath=f"{fpath}")

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
        _, wavelen, wavenum, _, _  = SetWave(filename=fpath, wavelength=cylhead['lambda'], wavenumber=None, ifilt=None)
        wavelength[ifile] = wavelen
        wavenumber[ifile] = wavenum

        # Calulate radiance error
        rad_error = CalculateErrors(imgdata, view)
        
        # Loop over each pixel to assign to the structure.
        xstart  = float(naxis1) - Globals.lonrange[0]/(360/naxis1)
        xstop   = float(naxis1) - Globals.lonrange[1]/(360/naxis1) 
        ystart  = (float(naxis2)/2) + Globals.latrange[0]/(180/naxis2)
        ystop   = (float(naxis2)/2) + Globals.latrange[1]/(180/naxis2) 
        x_range = np.arange(xstart, xstop, 1, dtype=int)
        y_range = np.arange(ystart, ystop, 1, dtype=int)
        for x in x_range:
            for y in y_range:
                # Only assign latitude and longitude if non-zero pixel value
                if (cyldata[y, x] > 0):
                    # Calculate finite spatial element (lat-lon co-ordinates)
                    lat = Globals.latrange[0] + ((180 / naxis2) * y)
                    lon = Globals.lonrange[0] - ((360 / naxis1) * x)
                    # Adjust co-ordinates from edge to centre of bins
                    lat = lat + Globals.latstep/res
                    lon = lon - Globals.latstep/res
                    # Convert from planetographic to planetocentric latitudes
                    mu_ang = mudata[y, x]
                    mu  = 180/pi * acos(mu_ang)
                    # Calculate pxel radiance and error
                    rad = cyldata[y, x] * 1e-7
                    
                    ## Step 3: Gather pxel information for all files
                    # Store spectral information in spectrum array
                    # print(y, x, ifile)
                    # print(LCP, lat, lon, mu, rad, wavenum, view)
                    # input()
                    spectrum[y, x, ifile, 0] = lat
                    spectrum[y, x, ifile, 1] = lon
                    spectrum[y, x, ifile, 2] = Globals.LCP if binning == 'bin_cpara' else np.nan
                    spectrum[y, x, ifile, 3] = LCMIII[ifile]
                    spectrum[y, x, ifile, 4] = mu
                    spectrum[y, x, ifile, 5] = rad
                    spectrum[y, x, ifile, 6] = rad_error * rad
                    spectrum[y, x, ifile, 7] = wavenum
                    spectrum[y, x, ifile, 8] = view
    
    return spectrum, wavelength, wavenumber, LCMIII