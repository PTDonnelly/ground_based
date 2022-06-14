from cmath import nan
from fileinput import filename
import os

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from scipy import interpolate
from BinningInputs import BinningInputs
from ReadFits import ReadFits
from SetWave import SetWave
from VisirWavenumbers import VisirWavenumbers
from VisirWavelengths import VisirWavelengths
from ConvertBrightnessTemperature import ConvertBrightnessTemperature

def PlotMaps(files, spectrals, ksingles, wavenumber):
    """ DB: Mapping global maps for each VISIR filter """

    print('Correcting and mapping global maps...')
    # If subdirectory does not exist, create it
    dir = '../outputs/global_maps_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    # Define local inputs
    nx, ny = 720, 360                  # Dimensions of an individual cylindrical map (needed for dictionary definition)
    res    = ny / 180                  # Resolution of maps: res = 1 (1 degree), res = 2 (0.5 degree) etc.
    Nfiles = len(files)
    
    # Create np.arrays for all pixels in all cmaps and mumaps
    cmaps      = np.empty((Nfiles, ny, nx))
    mumaps     = np.empty((Nfiles, ny, nx))
    TBmaps     = np.empty((Nfiles, ny, nx))
    globalmaps = np.empty((BinningInputs.nfilters, ny, nx))
    mumin      = np.empty((BinningInputs.nfilters,ny, nx))
    wavelength = np.empty(Nfiles)

    # Loop over file to load individual (and original) cylindrical maps
    for ifile, fname in enumerate(files):
        ## Step 1: Read img, cmap and mufiles
        imghead, _, cylhead, cyldata, _, mudata = ReadFits(filename=f"{fname}")

        ## Step 2: Geometric registration of pixel information
        # Save flag depending on Northern (1) or Southern (-1) viewing
        chopang = imghead['HIERARCH ESO TEL CHOP POSANG']
        posang  = imghead['HIERARCH ESO ADA POSANG'] + 360

        # Set the central wavelengths for each filter. Must be
        # identical to the central wavelength specified for the
        # production of the k-tables
        wavelen, _, _, _  = SetWave(wavelength=cylhead['lambda'], wavenumber=False)
        wavelength[ifile] = wavelen

        # Store corrected spectral information in np.array 
        # with ignoring negative beam on each cyldata maps
        if chopang == posang:
            # Northern view
            cmaps[ifile, int((ny-10)/2):ny, :] = cyldata[int((ny-10)/2):ny, :] * ksingles[ifile, 1]
            cmaps[ifile, 0:int((ny-10)/2), :]  = np.nan
        else:
            # Southern view
            cmaps[ifile, 0:int((ny+10)/2), :]  = cyldata[0:int((ny+10)/2), :] * ksingles[ifile, 1]
            cmaps[ifile, int((ny+10)/2):ny, :] = np.nan
        mumaps[ifile, :, :] = mudata
        # Convert radiance maps to brightness temperature maps
        cmaps[ifile, :, :] = ConvertBrightnessTemperature(cmaps[ifile, :, :],wavelength=wavelength[ifile])

    for ifilt in range(BinningInputs.nfilters):
        # Retrieve emision angle profiles from spectrals, interpolate it over cmaps latitude grid 
        # and create a minimum emission angle map for each filter
        spec_length = np.arange(0, 180)
        mu_prof = spectrals[:, ifilt, 2]
        f = interpolate.interp1d(spec_length, mu_prof, fill_value="extrapolate")
        for x in range(nx):
            lat = np.arange(0,ny)
            mumin[ifilt, :, x] = f(lat)
        # Empty local TBmaps array to avoid overlapping between filter
        TBmaps[:, :, :] = np.nan
        # Get filter index for spectral profiles
        waves = spectrals[:, ifilt, 5]
        wave  = waves[(waves > 0)][0]
        _, _, _, ifilt = SetWave(wavelength=False, wavenumber=wave)
        for ifile, iwave in enumerate(wavenumber):
            if iwave == wave:
                # Store only the cmaps for the current ifilt 
                TBmaps[ifile, :, :] = cmaps[ifile, :, :]                
               
                res = ma.masked_where(mumaps[ifile, :, :] < mumin[ifilt, :, :], TBmaps[ifile, :, :])
                #res = ma.masked_where(((res > 161)), res)
                #res = ma.masked_where(((res < 135)), res)
                TBmaps[ifile,:,:] = res.filled(np.nan)

        # Combinig single cylmaps to store in globalmaps array
        for y in range(ny):
            for x in range(nx):
                globalmaps[ifilt, y, x] = np.nanmax(TBmaps[:, y, x])



        # Plotting global map
        max = np.nanmax(globalmaps[ifilt, :, :]) 
        min = np.nanmin(globalmaps[ifilt, :, :]) 

        im = plt.imshow(globalmaps[ifilt, :, :], origin='lower', vmin=min, vmax=max, cmap='cividis')
        plt.xticks(np.arange(0, nx+1,  step = 60), list(np.arange(360,-1,-30)))
        plt.yticks(np.arange(0, ny+1, step = 60), list(np.arange(-90,91,30)))
        plt.xlabel('System III West Longitude')
        plt.ylabel('Planetocentric Latitude')
        #plt.tick_params(labelsize=15)
        cbar = plt.colorbar(im, extend='both')
        #cbar.ax.tick_params(labelsize=15)
        cbar.set_label("Brightness Temperature [K]")

        # Save figure showing calibation method 
        filt = VisirWavenumbers(ifilt)
        plt.savefig(f"{dir}{filt}_global_maps.png", dpi=900)
        plt.savefig(f"{dir}{filt}_global_maps.eps", dpi=900)
        # Clear figure to avoid overlapping between plotting subroutines
        plt.clf()

    return globalmaps
