from cmath import nan
from fileinput import filename
import os
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import Globals
from Tools.CorrectMaps import PolynomialAdjust
from Tools.SetWave import SetWave

def PlotMaps(files, spectrals):
    """ DB: Mapping global maps for each VISIR filter """

    print('Correcting global maps...')
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
    wavenumber = np.empty(Nfiles)
    TBmaps     = np.empty((Nfiles, ny, nx))
    globalmaps = np.empty((Globals.nfilters, ny, nx))
    #mumin      = np.empty((Globals.nfilters,ny, nx))

    cmaps, mumaps, wavenumber = PolynomialAdjust(dir, files, spectrals)

    print('Mapping global maps...')

    for ifilt in range(Globals.nfilters):
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
               
                res = ma.masked_where(mumaps[ifile, :, :] < 0.02, TBmaps[ifile, :, :])
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

        # Save global map figure of the current filter 
        filt = Wavenumbers(ifilt)
        plt.savefig(f"{dir}{filt}_global_maps.png", dpi=900)
        plt.savefig(f"{dir}{filt}_global_maps.eps", dpi=900)
        # Clear figure to avoid overlapping between plotting subroutines
        plt.clf()

    return globalmaps
