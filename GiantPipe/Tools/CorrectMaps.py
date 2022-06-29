import numpy as np
import matplotlib.pyplot as plt 
import Globals
from Read.ReadFits import ReadFits
from Tools.SetWave import SetWave
from Tools.VisirFilterInfo import Wavenumbers
from Tools.ConvertBrightnessTemperature import ConvertBrightnessTemperature

def PolynomialAdjust(directory, files, spectrals):
    # Define local inputs
    nx, ny = 720, 360                   # Dimensions of an individual cylindrical map (needed for dictionary definition)
    res    = ny / 180                   # Resolution of maps: res = 1 (1 degree), res = 2 (0.5 degree) etc.
    Nfiles = len(files)
    lat = np.arange(-89.75,90,step=0.5) # Latitude range from pole-to-pole
    mumin = 0.1
    
    # Create np.arrays for all pixels in all cmaps and mumaps
    cmaps      = np.empty((Nfiles, ny, nx))
    mumaps     = np.empty((Nfiles, ny, nx))
    wavelength = np.empty(Nfiles)
    wavenumber = np.empty(Nfiles)
    viewing_mode   = np.empty(Nfiles)

    # Define local arrays to store selected latitude band spectral data
    bandcmaps   = np.empty((Globals.nfilters, ny, nx))
    bandmumaps  = np.empty((Globals.nfilters, ny, nx))
    keepdata  = np.empty((Nfiles, ny, nx))
    keepmu    = np.empty((Nfiles, ny, nx))

    # Initialise to nan values
    bandcmaps[:, :, :]  = np.nan
    bandmumaps[:, :, :] = np.nan
    keepdata[:, :, :] = np.nan
    keepmu[:, :, :]   = np.nan

    # Loop over file to load individual (and original) cylindrical maps
    for ifile, fpath in enumerate(files):
        ## Step 1: Read img, cmap and mufiles
        imghead, _, cylhead, cyldata, _, mudata = ReadFits(filepath=f"{fpath}")

        ## Step 2: Geometric registration of pixel information
        # Save flag depending on Northern (1) or Southern (-1) viewing
        chopang = imghead['HIERARCH ESO TEL CHOP POSANG']
        posang  = imghead['HIERARCH ESO ADA POSANG'] + 360
        view = 1 if chopang == posang else -1
        viewing_mode[ifile] = view

        # Set the central wavelengths for each filter. Must be
        # identical to the central wavelength specified for the
        # production of the k-tables
        wavelen, wavenum, _, _  = SetWave(wavelength=cylhead['lambda'], wavenumber=False)
        wavelength[ifile] = wavelen
        wavenumber[ifile] = wavenum

        # Store corrected spectral information in np.array 
        # with ignoring negative beam on each cyldata maps
        if chopang == posang:
            # Northern view
            cmaps[ifile, int((ny-10)/2):ny, :] = cyldata[int((ny-10)/2):ny, :]
            cmaps[ifile, 0:int((ny-10)/2), :]  = np.nan
        else:
            # Southern view
            cmaps[ifile, 0:int((ny+10)/2), :]  = cyldata[0:int((ny+10)/2), :]
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
        if (ifilt !=  7): # ifilt = 7 cannot be corrected
            # Get filter index for spectral profiles
            waves = spectrals[:, ifilt, 5]
            wave  = waves[(waves > 0)][0]
            _, _, _, ifilt = SetWave(wavelength=False, wavenumber=wave)
            # Find all files for the current filter
            currentdata  = np.empty((ny, nx))   # array to append temperature maps of the current filter 
            currentmu    = np.empty((ny, nx))   # array to append emission angle maps of the current filter
            currentdata.fill(np.nan)            # filled of NaN to avoid strange scatter plotting
            currentmu.fill(np.nan)              # filled of NaN to avoid strange scatter plotting
            currentnfiles = 1 # Number of files for the current filter, initialised at 1 to take into account the initialised nan arrays
            for ifile, iwave in enumerate(wavenumber):
                if iwave == wave:
                    currentdata = np.append(currentdata, keepdata[ifile, :, :], axis = 1)
                    currentmu   = np.append(currentmu, keepmu[ifile, :, :], axis = 1)
                    currentnfiles += 1
            # Reshape the appended arrays to recover map dimensions
            selectdata = np.reshape(currentdata, (currentnfiles, ny, nx))
            selectmu   = np.reshape(currentmu, (currentnfiles, ny, nx))
            # Average selected data and mu over current number files (for this filter) to obtain averaged latitude band 
            for x in range(nx):
                for y in range(ny):
                    bandcmaps[ifilt, y, x] = np.nanmean(selectdata[:, y, x])
                    bandmumaps[ifilt, y, x] = np.nanmean(selectmu[:, y, x])
            # Define a mask depending on minimum emission angle for each hemisphere
            keep_north  = (lat < 30) & (lat > 5)
            bandcnorth  = bandcmaps[ifilt, keep_north, :]
            bandmunorth = bandmumaps[ifilt, keep_north, :]
            mask_north  = (( bandmunorth > mumin) & (bandcnorth > 90.))
            keep_south  = (lat < -5) & (lat > -30)
            bandcsouth  = bandcmaps[ifilt, keep_south, :]
            bandmusouth = bandmumaps[ifilt, keep_south, :]           
            mask_south  = ((bandmusouth > mumin) & (bandcsouth > 90.))
            # Calculate polynomial adjustement for each hemisphere (using mask selections)
            if (ifilt != 6): # ifilt = 6 is only a southern view
                p_north = np.poly1d(np.polyfit(bandmunorth[mask_north], bandcnorth[mask_north],4))
            p_south = np.poly1d(np.polyfit(bandmusouth[mask_south], bandcsouth[mask_south],4))
            # Define a linear space to show the polynomial adjustment variation over all emission angle range 
            t = np.linspace(mumin, 0.9, 100)
            # Some control printing
            if (ifilt != 6): # ifilt = 6 is only a southern view
                print('Northern polynome')
                print(p_north)
                print(bandcnorth[mask_north])
            print('Southern polynome')
            print(p_south)
            print(bandcsouth[mask_south])            
            # Correct data on the slected latitude band
            if (ifilt != 6): # ifilt = 6 is only a southern view
                cdata_north=bandcnorth[mask_north]*p_north(1)/p_north(bandmunorth[mask_north])
            cdata_south=bandcsouth[mask_south]*p_south(1)/p_south(bandmusouth[mask_south])    
            # Plot figure showing limb correction using polynomial adjustment method
            if (ifilt != 6): # ifilt = 6 is only a southern view
                ax1 = plt.subplot2grid((2, 3), (0, 0))
                ax1.scatter(bandmunorth[mask_north], bandcnorth[mask_north])
                ax1.plot(t, p_north(t), '-',color='red')
                ax2 = plt.subplot2grid((2, 3), (0, 1))
                ax2.plot(t, (p_north(1))/p_north(t), '-',color='red')
                ax3 = plt.subplot2grid((2, 3), (0, 2))
                ax3.scatter(bandmunorth[mask_north], cdata_north)
                ax4 = plt.subplot2grid((2, 3), (1, 0))
                ax4.scatter(bandmusouth[mask_south], bandcnorth[mask_south])
                ax4.plot(t, p_south(t), '-',color='red')
                ax5 = plt.subplot2grid((2, 3), (1, 1))
                ax5.plot(t, (p_south(1))/p_south(t), '-',color='red')
                ax6 = plt.subplot2grid((2, 3), (1, 2))
                ax6.scatter(bandmusouth[mask_south], cdata_south)
            else:
                ax1 = plt.subplot2grid((1, 3), (0, 0))
                ax1.scatter(bandmusouth[mask_south], bandcsouth[mask_south])
                ax1.plot(t, p_south(t), '-',color='red')
                ax2 = plt.subplot2grid((1, 3), (0, 1))
                ax2.plot(t, (p_south(1))/p_south(t), '-',color='red')
                ax3 = plt.subplot2grid((1, 3), (0, 2))
                ax3.scatter(bandmusouth[mask_south], cdata_south)
            # Save figure showing limb correction using polynomial adjustment method 
            filt = Wavenumbers(ifilt)
            plt.savefig(f"{directory}{filt}_polynomial_adjustment.png", dpi=900)
            plt.savefig(f"{directory}{filt}_polynomial_adjustment.eps", dpi=900)

            # Apply polynomial adjustment over individual cmaps depending of wave value
            for ifile, iwave in enumerate(wavenumber):
                if iwave == wave:
                    if viewing_mode[ifile] == 1:
                        if (ifilt != 6): # ifilt = 6 is only a southern view
                            cmaps[ifile, :, :] = cmaps[ifile, :, :] * p_north(1) / p_north(mumaps[ifile, :, :])
                    if viewing_mode[ifile] == -1:
                        cmaps[ifile, :, :] = cmaps[ifile, :, :] * p_south(1) / p_south(mumaps[ifile, :, :])

        # Clear figure to avoid overlapping between plotting subroutines
        plt.clf()

    return cmaps, mumaps, wavenumber 
