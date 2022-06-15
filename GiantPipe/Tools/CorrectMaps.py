import numpy as np
import matplotlib.pyplot as plt 
import Globals
from Read.ReadFits import ReadFits
from Tools.SetWave import SetWave
from Tools.VisirFilterInfo import Wavenumbers
from Tools.ConvertBrightnessTemperature import ConvertBrightnessTemperature

def PolynomialAdjust(directory, files, wavenumber, spectrals, ksingles):
    # Define local inputs
    nx, ny = 720, 360                   # Dimensions of an individual cylindrical map (needed for dictionary definition)
    res    = ny / 180                   # Resolution of maps: res = 1 (1 degree), res = 2 (0.5 degree) etc.
    Nfiles = len(files)
    lat = np.arange(-89.75,90,step=0.5) # Latitude range from pole-to-pole
    mumin = 0.3
    
    # Create np.arrays for all pixels in all cmaps and mumaps
    cmaps      = np.empty((Nfiles, ny, nx))
    mumaps     = np.empty((Nfiles, ny, nx))
    wavelength = np.empty(Nfiles)
    viewing_mode   = np.empty(Nfiles)

    # Define local arrays to store selected latitude band spectral data
    bandcmaps   = np.empty((Globals.nfilters, ny, nx))
    bandmumaps  = np.empty((Globals.nfilters, ny, nx))
    keepdata  = np.empty((Nfiles, ny, nx))
    keepmu    = np.empty((Nfiles, ny, nx))
    selectdata  = np.empty((Nfiles, ny, nx))
    selectmu    = np.empty((Nfiles, ny, nx))
    # Initialise to nan values
    bandcmaps[:, :, :]  = np.nan
    bandmumaps[:, :, :] = np.nan
    keepdata[:, :, :] = np.nan
    keepmu[:, :, :]   = np.nan
    selectdata[:, :, :] = np.nan
    selectmu[:, :, :]   = np.nan

    # Loop over file to load individual (and original) cylindrical maps
    for ifile, fname in enumerate(files):
        ## Step 1: Read img, cmap and mufiles
        imghead, _, cylhead, cyldata, _, mudata = ReadFits(filename=f"{fname}")

        ## Step 2: Geometric registration of pixel information
        # Save flag depending on Northern (1) or Southern (-1) viewing
        chopang = imghead['HIERARCH ESO TEL CHOP POSANG']
        posang  = imghead['HIERARCH ESO ADA POSANG'] + 360
        view = 1 if chopang == posang else -1
        viewing_mode[ifile] = view

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


    # Select latitude band for polynomial adjustment depending on viewing mode of each cmaps
    for ifile in range(Nfiles):
        keep = ((lat < 30) & (lat > 5)) if viewing_mode[ifile] == 1 else ((lat < -5) & (lat > -30))
        keepdata[ifile, keep, :]  = cmaps[ifile, keep, :]  
        keepmu[ifile, keep, :]    = mumaps[ifile, keep, :]

    for ifilt in range(Globals.nfilters):
        # Get filter index for spectral profiles
        waves = spectrals[:, ifilt, 5]
        wave  = waves[(waves > 0)][0]
        _, _, _, ifilt = SetWave(wavelength=False, wavenumber=wave)
        # Fing all files for the current filter
        for ifile, iwave in enumerate(wavenumber):
            if iwave == wave:
                selectdata[ifile, :, :] = keepdata[ifile, :, :]
                selectmu[ifile, :, :]   = keepmu[ifile, :, :]
                for x in range(nx):
                    for y in range(ny):
                        bandcmaps[ifilt, y, x] = np.nanmean(selectdata[:, y, x])
                        bandmumaps[ifilt, y, x] = np.nanmean(selectmu[:, y, x])
            selectdata[ifile, :, :] = np.nan
            selectmu[ifile, :, :]   = np.nan
        # Define a mask depending on minimum emission angle
        mask = ((bandmumaps[ifilt, :, :] > mumin) & (bandcmaps[ifilt, :, :] > 90.))
        # Calculate polynomial adjustement
        p = np.poly1d(np.polyfit(bandmumaps[ifilt, mask], bandcmaps[ifilt, mask],4))
        # Define a linear space to show the polynomial adjustment variation over all 
        # emission angle range 
        t = np.linspace(mumin, 0.9, 100)
        # Some control printing 
        print(p)
        print(bandcmaps[ifilt, mask])

        # Correct data on the slected latitude band
        cdata_all=bandcmaps[ifilt, mask]*p(1)/p(bandmumaps[ifilt, mask])    
        
        # Plot figure showing limb correction using polynomial adjustment method
        ax1 = plt.subplot2grid((1, 3), (0, 0))
        ax1.scatter(bandmumaps[ifilt, mask],bandcmaps[ifilt, mask])
        ax1.plot(t, p(t), '-',color='red')
        ax2 = plt.subplot2grid((1, 3), (0, 1))
        ax2.plot(t, (p(1))/p(t), '-',color='red')
        ax3 = plt.subplot2grid((1, 3), (0, 2))
        ax3.scatter(bandmumaps[ifilt, mask],cdata_all)
        # Save figure showing limb correction using polynomial adjustment method 
        filt = VisirWavenumbers(ifilt)
        plt.savefig(f"{directory}{filt}_polynomial_adjustment.png", dpi=900)
        plt.savefig(f"{directory}{filt}_polynomial_adjustment.eps", dpi=900)

        # Apply polynomial adjustment over individual cmaps depending of wave value
        for ifile, iwave in enumerate(wavenumber):
            if iwave == wave:
                    cmaps[ifile, :, :] = cmaps[ifile, :, :] * p(1) / p(mumaps[ifile, :, :])
     # Clear figure to avoid overlapping between plotting subroutines
    plt.clf()

    return cmaps, mumaps 
