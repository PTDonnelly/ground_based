import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import Globals
from Read.ReadFits import ReadFits
from Tools.SetWave import SetWave
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
        _, wavelen, wavenum, _, _  = SetWave(filename=fpath, wavelength=cylhead['lambda'], wavenumber=None, ifilt=None)
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

def MuNormalization(files):
    """ Function to normalise by the variation 
        of the emission angle at power 'mu_power' """

    # Define local inputs
    Nfiles = len(files)
    # Load the cylindrical and mu maps
    cmaps, mumaps, wavelength, wavenumber, viewing_mode = GetCylandMuMaps(files)

    # Loop over file to convert radiance maps to brightness temperature maps
    for ifile, fpath in enumerate(files):
        # retrieve name of the filter to set the correct power for mumaps
        filename = fpath
        filter_name = filename.split('visir_')
        filter_name = filter_name[-1].split('_20')
        filter_name = filter_name[0]
        # Set the power of mumaps
        if filter_name == 'J7.9':
            mu_power = 0.005
        if filter_name == 'PAH1' or filter_name == 'ARIII':
            mu_power = 0.3
        if filter_name =='SIV' or filter_name =='SIV_1' or filter_name == 'SIV_2' or filter_name == 'NEII_1' or filter_name == 'NEII_2':
            mu_power = 0.2
        if filter_name =='Q1' or filter_name == 'Q2' or filter_name =='Q3':
            mu_power = 0.18
        # Normalization by mumaps to the power mu_power
        cmaps[ifile, :, :] = cmaps[ifile, :, :] / mumaps[ifile, :, :]**mu_power
        # Convert radiance cmaps to brightness temperature 
        cmaps[ifile, :, :] = ConvertBrightnessTemperature(cmaps[ifile, :, :], wavelength=wavelength[ifile])

    return cmaps, mumaps, wavenumber

def PolynomialAdjust(directory, files, spectrals):
    """ Function to calculate and apply an unique polynomial 
        ajustment per filter over the entire dataset. 
        To date, it is only use for Jupiter 2018May dataset """

    # Define local inputs
    Nfiles = len(files)
    lat = np.arange(-89.75,90,step=0.5) # Latitude range from pole-to-pole
    mumin = [0.15, 0.15, 0.02, 0.25, 0.3, 0.2, 0.5, 0.5, 0.15, 0.3, 0.6, 0.5, 0.05]

    # Define local arrays to store selected latitude band spectral data
    bandcmaps   = np.empty((Globals.nfilters, Globals.ny, Globals.nx))
    bandmumaps  = np.empty((Globals.nfilters, Globals.ny, Globals.nx))
    keepdata  = np.empty((Nfiles, Globals.ny, Globals.nx))
    keepmu    = np.empty((Nfiles, Globals.ny, Globals.nx))
    # Define location adjustment array to store the latitude band information
    adj_location = np.empty(Globals.nfilters)
    # Initialise to nan values
    bandcmaps[:, :, :]  = np.nan
    bandmumaps[:, :, :] = np.nan
    keepdata[:, :, :] = np.nan
    keepmu[:, :, :]   = np.nan

    # Load the cylindrical and mu maps
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
        # After several tests, best results are obtained with an average polynomial adjustment
        # on the N-Band filter and a southern polynomial adjustment on the Q-Band Filters for
        # the VLT/VISIR Jupiter 2018May24-27 dataset. 
        adj_location= 'average' if ifilt < 10 else 'southern'
        if (ifilt !=  7) and (ifilt != 6): # ifilt = 7 cannot be corrected
            # Get filter index for spectral profiles
            waves = spectrals[:, ifilt, 5]
            wave  = waves[(waves > 0)][0]
            # print(ifilt)
            # _, _, _, _, ifilt = SetWave(filename=None, wavelength=False, wavenumber=wave, ifilt=False)
            # print(ifilt)
            # Find all files for the current filter
            currentdata  = np.empty((Globals.ny, Globals.nx))   # array to append temperature maps of the current filter 
            currentmu    = np.empty((Globals.ny, Globals.nx))   # array to append emission angle maps of the current filter
            currentdata.fill(np.nan)                            # filled of NaN to avoid strange scatter plotting
            currentmu.fill(np.nan)                              # filled of NaN to avoid strange scatter plotting
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
            # Northern, Southern, Hemispheric, Average:
            if adj_location == 'northern':
                if (ifilt != 6): # ifilt = 6 is only a southern view
                    if (ifilt == 3):
                        keep = ((lat < 15) & (lat > 10))
                    elif (ifilt == 4):
                        keep = ((lat < 25) & (lat > 10))
                    else:
                        keep = ((lat < 30) & (lat > 5))
                    # Store the latitude bands chosen for cyldata and mudata arrays
                    bandc  = bandcmaps[ifilt, keep, :]
                    bandmu = bandmumaps[ifilt, keep, :]
            
            if adj_location == 'southern':
                if (ifilt == 3):
                    keep = ((lat < -10 ) & (lat > -15))
                elif (ifilt == 4):
                    keep = ((lat < -10) & (lat > -25))
                else:
                    keep  = ((lat < -5) & (lat > -30))
                # Store the latitude bands chosen for cyldata and mudata arrays
                bandc  = bandcmaps[ifilt, keep, :]
                bandmu = bandmumaps[ifilt, keep, :]
            
            if adj_location == 'average' or adj_location == 'hemispheric':
                if (ifilt == 3):
                    keep_north = ((lat < 15) & (lat > 10))
                    keep_south = ((lat < -10 ) & (lat > -15))
                elif (ifilt == 4):
                    keep_north = ((lat < 35) & (lat > 10))
                    keep_south = ((lat < -10) & (lat > -35))
                else:
                    keep_north  = ((lat < 30) & (lat > 5))
                    keep_south  = ((lat < -5) & (lat > -30))
                # Store the north latitude bands chosen for cyldata and mudata arrays
                bandcnorth  = bandcmaps[ifilt, keep_north, :]
                bandmunorth = bandmumaps[ifilt, keep_north, :]
                # Define the northern region where the polynomial adjustment will be calculated
                mask_north  = (( bandmunorth >  mumin[ifilt]) & (bandcnorth > 90.))
                # Store the south latitude bands chosen for cyldata and mudata arrays
                bandcsouth  = bandcmaps[ifilt, keep_south, :]
                bandmusouth = bandmumaps[ifilt, keep_south, :]
                # Define the southern region where the polynomial adjustment will be calculated           
                mask_south  = ((bandmusouth > mumin[ifilt]) & (bandcsouth > 90.))

                # Average the two hemispheric band in case of average method is chosen
                # For the hemispheric one, a polynomial adjustment will be calculted 
                # on the mask_north area and another one on the mask_south area
                if adj_location == 'average':
                    # Average the two hemispheric bands, considering the poleward variation of hemispheric bands [::-1]
                    bandc_combined = np.append(bandcnorth, bandcsouth[::-1,:], axis = 1)
                    bandmu_combined = np.append(bandmunorth, bandmusouth[::-1,:], axis = 1)
                    iy = len(bandcnorth[:,0])
                    bandc_tmp = np.reshape(bandc_combined, (2, iy, Globals.nx))  
                    bandmu_tmp = np.reshape(bandmu_combined, (2, iy, Globals.nx))
                    bandc = np.empty((iy, Globals.nx))
                    bandmu = np.empty((iy, Globals.nx))
                    for x in range(Globals.nx):
                        for y in range(iy):
                            bandc[y, x] = np.nanmean(bandc_tmp[:, y, x])
                            bandmu[y, x] = np.nanmax(bandmu_tmp[:, y, x])
            # Define the region where the polynomial adjustment will be calculated
            if adj_location == 'northern' or adj_location =='southern' or adj_location == 'average':
                mask  = (( bandmu > mumin[ifilt]) & (bandc > 90.))
            
            # Calculate polynomial adjustement for the hemispheric method
            if adj_location == 'hemispheric':
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
            # Calculate polynomial adjustement for the northern method (igoring filters with only southern views)
            elif adj_location == 'northern':
                if (ifilt != 6): # ifilt = 6 is only a southern view
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
            # Calculate polynomial adjustement for the southern or average method
            else:
                p = np.poly1d(np.polyfit(bandmu[mask], bandc[mask],4))
                coeff = np.polyfit(bandmu[mask], bandc[mask],4)
                # Define a linear space to show the polynomial adjustment variation over all emission angle range
                t = np.linspace(mumin[ifilt], 1, 100)
                # Some control printing
                print(f"{adj_location} polynome")
                print(p)
                print(bandc[mask])
                # Correct data on the slected latitude band
                cdata=bandc[mask]*p(1)/p(bandmu[mask])
                # Plot figure showing limb correction using polynomial adjustment method
                fig, axes = plt.subplots(1, 3, figsize=(8,3), sharex=True, constrained_layout = True)
                
                axes[0].plot(bandmu[mask], bandc[mask], lw=0, marker='.', markersize=0.5, color = 'black')
                axes[0].set_ylabel(r'Observed T$_B$ [K]', size=15)
                axes[0].set_xlabel("Emission angle", size=15)
                axes[0].plot(t, p(t), '-',color='red')
                axes[0].tick_params(labelsize=12)
                

                axes[1].plot(t, (p(1))/p(t), '-',color='red')
                axes[1].set_ylabel("Polynome adjustment", size=15)
                axes[1].set_xlabel("Emission angle", size=15)
                axes[1].tick_params(labelsize=12)
                
                axes[2].plot(bandmu[mask], cdata, lw=0, marker='.', markersize=0.5, color = 'black')
                axes[2].set_ylabel(r'Corrected T$_B$ [K]', size=15)
                axes[2].set_xlabel("Emission angle", size=15)
                axes[2].tick_params(labelsize=12)

            # Save figure showing limb correction using polynomial adjustment method 
            _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
            plt.savefig(f"{directory}calib_{wavnb}_polynomial_adjustment_{adj_location}.png", dpi=150, bbox_inches='tight')
            # plt.savefig(f"{directory}calib_{wavnb}_polynomial_adjustment_{adj_location}.eps", dpi=150, bbox_inches='tight')
            # Save polynomial coefficients
            if adj_location != 'hemispheric':
                np.save(f"{directory}calib_{wavnb}_polynomial_coefficients_{adj_location}", coeff)
                # np.savetxt(f"{directory}calib_{wavnb}_polynomial_coefficients_{adj_location}.txt", coeff)            
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
        plt.close()

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
        adj_location= 'average' if ifilt < 10 else 'southern'
        if ifilt != 6 and ifilt != 7:
            # Get filter index for spectral profiles
            waves = spectrals[:, ifilt, 5]
            wave  = waves[(waves > 0)][0]
            _, _, _, _, ifilt = SetWave(filename=None, wavelength=False, wavenumber=wave, ifilt=False)
            # Get polynome coefficient calculating for 2018May dataset 
            coeff = np.load(f'../outputs/2018May/global_maps_figures/calib_{wave}_polynomial_coefficients_{adj_location}.npy')
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

def BlackLineRemoving(directory, files, cblack, mu_scaling=False):
    """ Function to interpolate observation over the black line latitude
        (when the visir_cleanjup.pro program is not able to properly correct 
        the central brust, we mask it with a black line) """
    
    # If subdirectory does not exist, create it
    dir = f'{directory}/black_line_removing_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    lat = np.arange(-89.75,90,step=0.5) # Latitude range from pole-to-pole
    Nfiles = len(files)

    if mu_scaling == True: 
        # Normalisation by mumaps to the power something 
        # conversion into brightness temperature 
        cmaps, mumaps, wavenumber = MuNormalization(files)
    else:
        # Get cylindrical and mu maps 
        cmaps, mumaps, wavelength, wavenumber, viewing_mode = GetCylandMuMaps(files)
    
    for ifile in range(Nfiles):
        if mu_scaling == False:
            # Loop over file to convert radiance maps to brightness temperature maps
            cmaps[ifile, :, :] = ConvertBrightnessTemperature(cmaps[ifile, :, :], wavelength=wavelength)
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
