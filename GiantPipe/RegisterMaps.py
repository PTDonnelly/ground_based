import os
import numpy as np
from math import acos, cos, radians, pi
import Globals
from Read.ReadFits import ReadFits
from Tools.SetWave import SetWave
from Tools.CalculateErrors import CalculateErrors

def RegisterMaps(dataset, files, binning):
    """ Step 1: Read img, cmap and mufiles
        Step 2: Geometric registration of pixel information
        Step 3: Gather pixel information for all files"""
    
    print('Registering maps...')
    
    # Define local inputs
    nx, ny = 720, 360                  # Dimensions of an individual cylindrical map (needed for dictionary definition)
    res    = ny / 180                  # Resolution of maps: res = 1 (1 degree), res = 2 (0.5 degree) etc.
    Nfiles = len(files)

    # Create np.array for all pixels in all cmaps and mumaps
    spectrum = np.empty((ny, nx, Nfiles, 10))
    # Throw away zeros
    spectrum.fill(np.nan)

    # Define arrays
    viewing_mode   = np.empty(Nfiles)
    wavelength     = np.empty(Nfiles)
    wavenumber     = np.empty(Nfiles)
    LCMIII         = np.empty(Nfiles)
    DATE           = [None]*(Nfiles)

    # Define flags
    pg2pc = 0                   # Optional conversion of latitudes from planetographic to planetocentric

    # If subdirectory does not exist, create it
    dir = f'../outputs/{dataset}/dataset_tables_{binning}/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(f"{dir}dataset_tables.txt", "w") as file:
        file.write(f"Date & Hour & Image ID & $\lambda$ & LCMIII & Hemisphere \n")
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

            # Store observaing date
            DATE[ifile] = cylhead['DATE-OBS']
            
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
                        # Convert from mumap values to emission angle values
                        mu_ang = mudata[y, x]
                        mu  = 180/pi * acos(mu_ang)
                        # Calculate pxel radiance and error
                        rad = cyldata[y, x] * 1e-7
                        
                        
                        # Change date format to store it as a float in singles_av_regions arrays:
                        date = DATE[ifile].replace('T', '')
                        date = date.replace('-', '')
                        date = date.replace(':', '')
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
                        spectrum[y, x, ifile, 9] = date
            
            
            
            info = DATE[ifile].split('T')
            date = info[0].replace('-', ' ')
            infos = info[1].split('.')
            hour = infos[0]
            imageID = infos[1]

            # Write individual mean profiles to textfile
            if viewing_mode[ifile] > 0:
                file.write(f"{date} & {hour} & {imageID} & {wavelength[ifile]} & {int(LCMIII[ifile])} & northern \n")
            elif viewing_mode[ifile] < 0:
                file.write(f"{date} & {hour} & {imageID} & {wavelength[ifile]} & {int(LCMIII[ifile])} & southern \n")
        
    return spectrum, wavelength, wavenumber, LCMIII, DATE