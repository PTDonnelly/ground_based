import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import Globals
from Read.ReadFits import ReadFits
from Tools.SetWave import SetWave
from Tools.VisirFilterInfo import Wavenumbers
from Tools.ConvertBrightnessTemperature import ConvertBrightnessTemperature

def GetCylandMuMaps(files):
    """ Function to prepare cylindrical and mu maps for correcting and/or plotting """

    # Define local inputs
    Nfiles = len(files)

    # Create np.arrays for all pixels in all cmaps and mumaps
    cmaps      = np.empty((Nfiles, Globals.ny, Globals.nx))
    mumaps     = np.empty((Nfiles, Globals.ny, Globals.nx))
    wavelength = np.empty(Nfiles)
    wavenumber = np.empty(Nfiles)
    viewing_mode   = np.empty(Nfiles)

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
            cmaps[ifile, int((Globals.ny)/2):Globals.ny, :] = cyldata[int((Globals.ny)/2):Globals.ny, :]
            cmaps[ifile, 0:int((Globals.ny)/2), :]  = np.nan
        else:
            # Southern view
            cmaps[ifile, 0:int((Globals.ny)/2), :]  = cyldata[0:int((Globals.ny)/2), :]
            cmaps[ifile, int((Globals.ny)/2):Globals.ny, :] = np.nan
        mumaps[ifile, :, :] = mudata

    return cmaps, mumaps, wavelength, wavenumber, viewing_mode

def PolynomialAdjust(directory, files, spectrals):
    """ Function to calculate and apply an unique polynomial 
        ajustment per filter over the entire dataset. 
        To date, it is only use for Jupiter 2018May dataset """

    # Define local inputs
    Nfiles = len(files)
    lat = np.arange(-89.75,90,step=0.5) # Latitude range from pole-to-pole
    mumin = [0.15, 0.15, 0.02, 0.37, 0.3, 0.2, 0.5, 0.5, 0.15, 0.3, 0.6, 0.5, 0.05]

    # Define local arrays to store selected latitude band spectral data
    bandcmaps   = np.empty((Globals.nfilters, Globals.ny, Globals.nx))
    bandmumaps  = np.empty((Globals.nfilters, Globals.ny, Globals.nx))
    keepdata  = np.empty((Nfiles, Globals.ny, Globals.nx))
    keepmu    = np.empty((Nfiles, Globals.ny, Globals.nx))

    # Initialise to nan values
    bandcmaps[:, :, :]  = np.nan
    bandmumaps[:, :, :] = np.nan
    keepdata[:, :, :] = np.nan
    keepmu[:, :, :]   = np.nan

    #adj_location = 'northern'    
    #adj_location = 'southern'
    adj_location = 'hemispheric'
    #adj_location = 'average'

    cmaps, mumaps, wavelength, wavenumber, viewing_mode = GetCylandMuMaps(files)

    # Loop over file to convert radiance maps to brightness temperature maps
    for ifile in range(Nfiles):
        cmaps[ifile, :, :] = ConvertBrightnessTemperature(cmaps[ifile, :, :], wavelength=wavelength[ifile])

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
            currentdata  = np.empty((Globals.ny, Globals.nx))   # array to append temperature maps of the current filter 
            currentmu    = np.empty((Globals.ny, Globals.nx))   # array to append emission angle maps of the current filter
            currentdata.fill(np.nan)            # filled of NaN to avoid strange scatter plotting
            currentmu.fill(np.nan)              # filled of NaN to avoid strange scatter plotting
            currentnfiles = 1 # Number of files for the current filter, initialised at 1 to take into account the initialised nan arrays
            for ifile, iwave in enumerate(wavenumber):
                if iwave == wave:
                    currentdata = np.append(currentdata, keepdata[ifile, :, :], axis = 1)
                    currentmu   = np.append(currentmu, keepmu[ifile, :, :], axis = 1)
                    currentnfiles += 1
            # Reshape the appended arrays to recover map dimensions
            selectdata = np.reshape(currentdata, (currentnfiles, Globals.ny, Globals.nx))
            selectmu   = np.reshape(currentmu, (currentnfiles, Globals.ny, Globals.nx))
            # Average selected data and mu over current number files (for this filter) to obtain averaged latitude band 
            for x in range(Globals.nx):
                for y in range(Globals.ny):
                    bandcmaps[ifilt, y, x] = np.nanmean(selectdata[:, y, x])
                    bandmumaps[ifilt, y, x] = np.nanmean(selectmu[:, y, x])
            # Four posibilities to calculate the polynomial adjustemnt
            # Hemispheric, Northern, Southern, Average:
            if adj_location == 'hemispheric':
                # Define a mask depending on minimum emission angle for each hemisphere
                if (ifilt == 3):
                    keep_north = ((lat < 15) & (lat > 10))
                    keep_south = ((lat < -10 ) & (lat > -15))
                elif (ifilt == 4):
                    keep_north = ((lat < 25) & (lat > 10))
                    keep_south = ((lat < -10) & (lat > -25))
                else:
                    keep_north  = ((lat < 30) & (lat > 5))
                    keep_south  = ((lat < -5) & (lat > -30))
                bandcnorth  = bandcmaps[ifilt, keep_north, :]
                bandmunorth = bandmumaps[ifilt, keep_north, :]
                mask_north  = (( bandmunorth >  mumin[ifilt]) & (bandcnorth > 90.))
                bandcsouth  = bandcmaps[ifilt, keep_south, :]
                bandmusouth = bandmumaps[ifilt, keep_south, :]           
                mask_south  = ((bandmusouth > mumin[ifilt]) & (bandcsouth > 90.))
                # Calculate polynomial adjustement for each hemisphere (using mask selections)
                if (ifilt != 6): # ifilt = 6 is only a southern view
                    p_north = np.poly1d(np.polyfit(bandmunorth[mask_north], bandcnorth[mask_north],4))
                p_south = np.poly1d(np.polyfit(bandmusouth[mask_south], bandcsouth[mask_south],4))
                # Define a linear space to show the polynomial adjustment variation over all emission angle range
                t = np.linspace(mumin[ifilt], 0.9, 100)
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
                ax4.scatter(bandmusouth[mask_south], bandcsouth[mask_south])
                ax4.plot(t, p_south(t), '-',color='red')
                ax5 = plt.subplot2grid((2, 3), (1, 1))
                ax5.plot(t, (p_south(1))/p_south(t), '-',color='red')
                ax6 = plt.subplot2grid((2, 3), (1, 2))
                ax6.scatter(bandmusouth[mask_south], cdata_south)
            else:
                # Define a mask depending on minimum emission angle and the adjustement location chosen
                if adj_location == 'northern':
                    if (ifilt != 6): # ifilt = 6 is only a southern view
                        if (ifilt == 3):
                            keep = ((lat < 15) & (lat > 10))
                        elif (ifilt == 4):
                            keep = ((lat < 25) & (lat > 10))
                        else:
                            keep = ((lat < 30) & (lat > 5))
                if adj_location == 'southern':
                    if (ifilt == 3):
                        keep = ((lat < -10 ) & (lat > -15))
                    elif (ifilt == 4):
                        keep = ((lat < -10) & (lat > -25))
                    else:
                        keep  = ((lat < -5) & (lat > -30))
                bandc  = bandcmaps[ifilt, keep, :]
                bandmu = bandmumaps[ifilt, keep, :]
                if adj_location == 'average':
                    if (ifilt == 3):
                        keep_north = ((lat < 15) & (lat > 10))
                        keep_south = ((lat < -10 ) & (lat > -15))
                    elif (ifilt == 4):
                        keep_north = ((lat < 25) & (lat > 10))
                        keep_south = ((lat < -10) & (lat > -25))
                    else:
                        keep_north  = ((lat < 30) & (lat > 5))
                        keep_south  = ((lat < -5) & (lat > -30))
                    bandcnorth  = bandcmaps[ifilt, keep_north, :]
                    bandmunorth = bandmumaps[ifilt, keep_north, :]
                    mask_north  = (( bandmunorth >  mumin[ifilt]) & (bandcnorth > 90.))
                    bandcsouth  = bandcmaps[ifilt, keep_south, :]
                    bandmusouth = bandmumaps[ifilt, keep_south, :]           
                    mask_south  = ((bandmusouth > mumin[ifilt]) & (bandcsouth > 90.))
                    # Average the two hemispheric bands
                    bandc_combined = np.append(bandcnorth, bandcsouth[::-1,:], axis = 1)
                    bandmu_combined = np.append(bandmunorth, bandmusouth[::-1,:], axis = 1)
                    iy = len(bandcnorth[:,0])
                    bandc_tmp = np.reshape(bandc_combined, (2, iy, Globals.nx))  
                    bandmu_tmp = np.reshape(bandmu_combined, (2, iy, Globals.nx))
                    for x in range(Globals.nx):
                        for y in range(iy):
                            bandc[y, x] = np.nanmean(bandc_tmp[:, y, x])
                            bandmu[y, x] = np.nanmax(bandmu_tmp[:, y, x])
                mask  = (( bandmu > mumin[ifilt]) & (bandc > 90.))
                if adj_location == 'northern':
                    if (ifilt != 6): # ifilt = 6 is only a southern view
                        # Calculate polynomial adjustement for each hemisphere (using mask selections)
                        p = np.poly1d(np.polyfit(bandmu[mask], bandc[mask],4))
                        coeff = np.polyfit(bandmu[mask], bandc[mask],4)
                        # Define a linear space to show the polynomial adjustment variation over all emission angle range
                        t = np.linspace(mumin[ifilt], 0.9, 100)
                        # Some control printing
                        print(f"{adj_location} polynome")
                        print(p)
                        print(bandc[mask])
                        # Correct data on the slected latitude band
                        cdata=bandc[mask]*p(1)/p(bandmu[mask])
                        # Plot figure showing limb correction using polynomial adjustment method
                        ax1 = plt.subplot2grid((1, 3), (0, 0))
                        ax1.scatter(bandmu[mask], bandc[mask])
                        ax1.plot(t, p(t), '-',color='red')
                        ax2 = plt.subplot2grid((1, 3), (0, 1))
                        ax2.plot(t, (p(1))/p(t), '-',color='red')
                        ax3 = plt.subplot2grid((1, 3), (0, 2))
                        ax3.scatter(bandmu[mask], cdata)
                else:
                    # Calculate polynomial adjustement for each hemisphere (using mask selections)
                    p = np.poly1d(np.polyfit(bandmu[mask], bandc[mask],4))
                    coeff = np.polyfit(bandmu[mask], bandc[mask],4)
                    # Define a linear space to show the polynomial adjustment variation over all emission angle range
                    t = np.linspace(mumin[ifilt], 0.9, 100)
                    # Some control printing
                    print(f"{adj_location} polynome")
                    print(p)
                    print(bandc[mask])
                    # Correct data on the slected latitude band
                    cdata=bandc[mask]*p(1)/p(bandmu[mask])
                    # Plot figure showing limb correction using polynomial adjustment method
                    ax1 = plt.subplot2grid((1, 3), (0, 0))
                    ax1.scatter(bandmu[mask], bandc[mask])
                    ax1.plot(t, p(t), '-',color='red')
                    ax2 = plt.subplot2grid((1, 3), (0, 1))
                    ax2.plot(t, (p(1))/p(t), '-',color='red')
                    ax3 = plt.subplot2grid((1, 3), (0, 2))
                    ax3.scatter(bandmu[mask], cdata)
            # Save figure showing limb correction using polynomial adjustment method 
            filt = Wavenumbers(ifilt)
            plt.savefig(f"{directory}calib_{filt}_polynomial_adjustment_{adj_location}.png", dpi=300)
            plt.savefig(f"{directory}calib_{filt}_polynomial_adjustment_{adj_location}.eps", dpi=300)
            # Save polynomial coefficients
            if adj_location != 'hemispheric':
                np.save(f"{directory}calib_{filt}_polynomial_coefficients_{adj_location}", coeff)
                np.savetxt(f"{directory}calib_{filt}_polynomial_coefficients_{adj_location}.txt", coeff)            
            # Apply polynomial adjustment over individual cmaps depending of wave value
            for ifile, iwave in enumerate(wavenumber):
                if iwave == wave:
                    if adj_location == 'hemispheric':
                        if viewing_mode[ifile] == 1:
                            if (ifilt != 6): # ifilt = 6 is only a southern view
                                cmaps[ifile, :, :] = cmaps[ifile, :, :] * p_north(1) / p_north(mumaps[ifile, :, :])
                        if viewing_mode[ifile] == -1:
                            cmaps[ifile, :, :] = cmaps[ifile, :, :] * p_south(1) / p_south(mumaps[ifile, :, :])
                    elif adj_location == 'northern':
                        if (ifilt != 6): # ifilt = 6 is only a southern view
                            cmaps[ifile, :, :] = cmaps[ifile, :, :] * p(1) / p(mumaps[ifile, :, :])
                    else: # adj_location == 'southern' or adj_location == 'average':
                        cmaps[ifile, :, :] = cmaps[ifile, :, :] * p(1) / p(mumaps[ifile, :, :])
                        
        # Clear figure to avoid overlapping between plotting subroutines
        plt.clf()

    return cmaps, mumaps, wavenumber, adj_location 

def ApplyPolynom(directory, files, spectrals):
    """ Function to apply a pre-calculated polynomial adjustment with 
        2018May dataset """

    Nfiles = len(files)
    # Get cylindrical and mu maps 
    cmaps, mumaps, wavelength, wavenumber, viewing_mode = GetCylandMuMaps(files)

    # Loop over file to convert radiance maps to brightness temperature maps
    for ifile in range(Nfiles):
        cmaps[ifile, :, :] = ConvertBrightnessTemperature(cmaps[ifile, :, :], wavelength=wavelength[ifile])

    for ifilt in range(Globals.nfilters):
        # Get filter index for spectral profiles
        waves = spectrals[:, ifilt, 5]
        wave  = waves[(waves > 0)][0]
        _, _, _, ifilt = SetWave(wavelength=False, wavenumber=wave)
        # Get polynome coefficient calculating for 2018May dataset 
        coeff = np.load(f'../outputs/2018May/global_maps_figures/calib_{wave}_polynomial_coefficients_average.npy')
        # Calculate polynomial adjustement for each hemisphere (using mask selections)
        p = np.poly1d(coeff)
        # Some control printing
        print("Polynome")
        print(p)
        # Apply polynomial adjustment over individual cmaps depending of wave value
        for ifile, iwave in enumerate(wavenumber):
                if iwave == wave:
                    cmaps[ifile, :, :] = cmaps[ifile, :, :] * p(1) / p(mumaps[ifile, :, :])

    return cmaps, mumaps, wavenumber

def BlackLineRemoving(directory, files, cblack):
    """ Function to interpolate observation over the black line latitude
        (when the visir_cleanjup.pro program is not able to properly correct 
        the central brust, we mask it with a black line) """
    
    # If subdirectory does not exist, create it
    dir = f'{directory}/black_line_removing_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    lat = np.arange(-89.75,90,step=0.5) # Latitude range from pole-to-pole
    Nfiles = len(files)

    # Get cylindrical and mu maps 
    cmaps, mumaps, wavelength, wavenumber, viewing_mode = GetCylandMuMaps(files)

    for ifile in range(Nfiles):
        # Loop over file to convert radiance maps to brightness temperature maps
        cmaps[ifile, :, :] = ConvertBrightnessTemperature(cmaps[ifile, :, :], wavelength=wavelength[ifile])
        # Interpolation over latitudes for each longitude containing data
        print(files[ifile])
        lonkeep = []
        for x in range(Globals.nx):
            # Find longitude contaning data
            if not np.isnan(cmaps[ifile, :, x]).all():
                # Interpolation needs a minimum number of data pixels to do the interpolation
                y =(cmaps[ifile, :, x] > 0)
                check_len = cmaps[ifile, y, x]
                # We set this minimum number of data pixels to 50
                if len(check_len) > 50:
                    # Replace black line low value by NaN values
                    lblack = (lat < (cblack+4)) & (lat > (cblack-4))
                    cmaps[ifile, lblack, x] = np.nan
                    # Keep longitude index to limit the interpolation area, regarding the required minimum number of data pixels
                    lonkeep.append(x) 
        # Interpolate observations over the black line location, only for NaN surrounding by data
        yp = pd.DataFrame(cmaps[ifile, :, lonkeep])
        yp = yp.interpolate(limit_area='inside', method='linear', axis=1)
        # Fill cmaps array with interpolated black line area
        cmaps[ifile, :, lonkeep] = yp

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(lat, cmaps[ifile, :, x], 'o',label="Observation")
        # ax.plot(lat, yp, 'x',label="pandas interpolation")
        # plt.legend()
        # # Save figure showing black line correction by interpolation 
        # plt.savefig(f"{dir}calib_{wavenumber[ifile]}_black_line_removing_lat{x}.png", dpi=300)
        # #plt.savefig(f"{dir}calib_{wavenumber[ifile]}_black_line_removing_lat{x}.eps", dpi=300)
        # plt.close() 

    
    return cmaps, mumaps, wavenumber