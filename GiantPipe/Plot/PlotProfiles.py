import os 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from copy import copy
from scipy.interpolate import UnivariateSpline, interp1d
import Globals
from Read.ReadCal import ReadCal
from Tools.SetWave import SetWave

# Colormap definition
cmap = get_cmap("magma")

def PlotMeridProfiles(dataset, mode, files, singles, spectrals):
    """ Plot meridian profiles and spacecraft data to illustrate 
            the calibration method """

    print('Plotting profiles...')

    # Read in Voyager and Cassini data into arrays
    calfile = "../inputs/visir.jup.filtered-iris-cirs.10-12-15.data.v3"
    iris, cirs = ReadCal(calfile)

    # If subdirectory does not exist, create it
    dir = f'../outputs/{dataset}/calibration_profiles_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    for ifilt in range(Globals.nfilters):
        # Get filter index for plotting spacecraft and calibrated data
        _, _, wave, ifilt_sc, ifilt_v = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=ifilt)
        
        # Create a figure per filter
        fig = plt.figure(figsize=(8, 3))
        # subplot showing the averaging of each singles merid profiles (ignoring negative beam)
        for ifile, fname in enumerate(files):
            _, _, iwave, _, _ = SetWave(filename=fname, wavelength=None, wavenumber=None, ifilt=None)
            if iwave == wave:
                plt.plot(singles[:, ifile, 0], singles[:, ifile, 3], color='black', lw=0, marker='.', markersize=2)
        # Select the suitable spacecraft meridian profile
        if ifilt_sc < 12:
            # Use CIRS for N-Band
            plt.plot(cirs[:, ifilt_sc, 0], cirs[:, ifilt_sc, 1], color='green', lw=2, label='Cassini/CIRS')
        else:
            # Use IRIS for Q-Band
            plt.plot(iris[:, ifilt_sc, 0], iris[:, ifilt_sc, 1], color='green', lw=2, label='Voyager/IRIS')
        # Plot the VLT/VISIR pole-to-pole meridian profile
        plt.plot(spectrals[:, ifilt_v, 0], spectrals[:, ifilt_v, 3], color='orange', lw=0, marker='o', markersize=3, label=f"averaged VLT/VISIR at {int(wave)}"+" cm$^{-1}$")
        plt.xlim((-90, 90))
        plt.tick_params(labelsize=12)
        plt.grid()
        plt.legend(fontsize=12)
        plt.xlabel("Planetocentric Latitude", size=15)
        plt.ylabel("Radiance (W cm$^{-1}$ sr$^{-1}$)", size=15)

        # Save figure showing calibation method 
        # _, _, wave, _, _ = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=ifilt)
        plt.savefig(f"{dir}{wave}_calibration_merid_profiles.png", dpi=150, bbox_inches='tight')
        #plt.savefig(f"{dir}{wave}_calibration_merid_profiles.eps", dpi=900)
        # Clear figure to avoid overlapping between plotting subroutines
        plt.clf()

def PlotParaProfiles(dataset, mode, files, singles, spectrals):
    """ Plot parallel profiles """

    print('Plotting parallel profiles...')

    # If subdirectory does not exist, create it
    dir = f'../outputs/{dataset}/parallel_profiles_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    for ifilt in range(Globals.nfilters):
        # Get filter index for plotting spacecraft and calibrated data
        _, _, wave, _, ifilt_v = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=ifilt)
        # Create a figure per filter
        fig = plt.subplots(1, 1, figsize=(8, 3), sharex=True, sharey=True)
        # subplot showing the averaging of each singles merid profiles (ignoring negative beam)
        for ifile, fname in enumerate(files):
            _, _, iwave, _, _ = SetWave(filename=fname, wavelength=None, wavenumber=None, ifilt=None)
            if iwave == wave:
                plt.plot(singles[:, ifile, 1], singles[:, ifile, 3]*1.e9, lw=0, marker='.', markersize=2, color = 'black')
        plt.plot(spectrals[:, ifilt_v, 1], spectrals[:, ifilt_v, 3]*1.e9, color='orange', lw=0, marker='o', markersize=2, label=f"{int(wave)}"+" cm$^{-1}$ VLT/VISIR profile at "+f"{Globals.LCP}"+"$^{\circ}$")
        plt.legend(fontsize=12)
        plt.grid()
        plt.tick_params(labelsize=12) 
        plt.xlabel("System III West Longitude", size=15)
        print(spectrals[0, ifilt_v, 1], spectrals[-1, ifilt_v, 1])
        if spectrals[0, ifilt_v, 1] >=0 and spectrals[-1, ifilt_v, 1]>=0:
            plt.xlim(spectrals[0, ifilt_v, 1], spectrals[-1, ifilt_v, 1]) 
            plt.xticks(ticks=np.arange(360,-1,-30), labels=list(np.arange(360,-1,-30)))
        plt.ylabel("Radiance (nW cm$^{-1}$ sr$^{-1}$)", size=15)
        # Save figure showing calibation method 
        plt.savefig(f"{dir}{wave}_parallel_profiles.png", dpi=150, bbox_inches='tight')
        #plt.savefig(f"{dir}{wave}_calibration_merid_profiles.eps", dpi=900)
        # Clear figure to avoid overlapping between plotting subroutines
        plt.clf()

def PlotGlobalSpectrals(dataset, spectrals):
    """Basic code to plot the central meridian profiles with wavenumber
    (or spectral profiles with latitude, depending on the perspective).
    Displays the global pseudo-spectrum in a way resembling a normal spectrum."""
    
    # If subdirectory does not exist, create it
    dir = f'../outputs/{dataset}/calibration_profiles_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    lat = copy(spectrals[:, :, 0])
    wave = copy(spectrals[:, :, 5])
    rad =  copy(spectrals[:, :, 3])
    rad_res1 = copy(spectrals[:, :, 3])
    rad_res2 = copy(spectrals[:, :, 3])
    for i in range(Globals.nfilters):
        rad_res1[:, i] = rad[:, i] - np.nanmean(rad[:, i])
        rad_res2[:, i] = (rad_res1[:, i]/np.nanmean(rad[:, i]))*100

    plt.figure
    ax1 = plt.contourf(wave, lat, rad, levels=200, cmap='nipy_spectral')
    for i in range(Globals.nfilters):
        plt.plot((wave[i], wave[i]), (-90, 90), ls=':', lw=0.7, color='white')
    cbar = plt.colorbar(ax1)
    cbar.set_label('Radiance (W cm$^{-2}$ sr$^{-1}$ (cm$^{-1}$)$^{-1}$)', size=15)
    plt.xlabel('Wavenumber (cm$^{-1}$)', size=15)
    #plt.xlim((xmin, xmax))
    plt.ylabel('Latitude', size=15)
    plt.ylim((-90, 90))
    plt.savefig(f"{dir}global_spectrals.png", dpi=150, bbox_inches='tight')
    plt.close()
 
    plt.figure
    ax2 = plt.contourf(wave, lat, rad_res1, vmin=-1*np.nanmax(rad_res1),vmax=np.nanmax(rad_res1), levels=200, cmap='seismic')
    for i in range(Globals.nfilters):
        plt.plot((wave[i], wave[i]), (-90, 90), ls=':', lw=0.7, color='black')
    cbar = plt.colorbar(ax2)
    cbar.set_label('Residual radiance (W cm$^{-2}$ sr$^{-1}$ (cm$^{-1}$)$^{-1}$)', size=15)
    plt.xlabel('Wavenumber (cm$^{-1}$)', size=15)
    #plt.xlim((xmin, xmax))
    plt.ylabel('Latitude', size=15)
    plt.ylim((-90, 90))
    plt.savefig(f"{dir}global_spectrals_res1.png", dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure
    ax3 = plt.contourf(wave, lat, rad_res2, vmin=-1*np.nanmax(rad_res2),vmax=np.nanmax(rad_res2), levels=200, cmap='seismic')
    for i in range(Globals.nfilters):
        plt.plot((wave[i], wave[i]), (-90, 90), ls=':', lw=0.7, color='black')
    cbar = plt.colorbar(ax3)
    cbar.set_label('Difference (pourcent)', size=15)
    plt.xlabel('Wavenumber (cm$^{-1}$)', size=15)
    #plt.xlim((xmin, xmax))
    plt.ylabel('Latitude', size=15)
    plt.ylim((-90, 90))
    plt.savefig(f"{dir}global_spectrals_res2.png", dpi=150, bbox_inches='tight')
    plt.close()

def PlotCentreTotLimbProfiles(mode, singles, spectrals):
    print('Plotting profiles...')
    
    a = 1

def PlotBiDimMaps(dataset, mode, spectrals):
    """ Plot radiance maps """

    print('Plotting radiances maps...')

    # If subdirectory does not exist, create it
    dir = f'../outputs/{dataset}/maps_radiance_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
   
    for ifilt in range(Globals.nfilters):
        # Get retrieve wavenumber value from ifilt index
        _, _, wave, _, _ = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=ifilt)  
        # Set extreme values for mapping
        spectrals[:, :, ifilt, 3] *= 1.e9
        max = np.nanmax(spectrals[:, :, ifilt, 3]) 
        min = np.nanmin(spectrals[:, :, ifilt, 3])
        # Create a figure per filter
        fig = plt.figure(figsize=(8, 3))
        plt.imshow(spectrals[:, :, ifilt, 3], vmin=min, vmax=max, origin='lower', extent = [360,0,-90,90],  cmap='inferno')
        plt.xlim(Globals.lon_target+Globals.merid_width, Globals.lon_target-Globals.merid_width)
        plt.xticks(np.arange(Globals.lon_target-Globals.merid_width, Globals.lon_target+Globals.merid_width+1,  step = Globals.merid_width/2))
        plt.xlabel('System III West Longitude', size=15)
        plt.ylim(Globals.lat_target-Globals.para_width, Globals.lat_target+Globals.para_width)
        plt.yticks(np.arange(Globals.lat_target-Globals.para_width, Globals.lat_target+Globals.para_width+1, step = 5))
        plt.ylabel('Planetocentric Latitude', size=15)
        plt.tick_params(labelsize=12)
        cbar = plt.colorbar(extend='both', fraction=0.04, pad=0.05)#, orientation='horizontal')
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.locator_params(nbins=6)
        cbar.set_label("Radiance (nW cm$^{-1}$ sr$^{-1}$)", size=15)
        # Save figure showing calibation method 
        plt.savefig(f"{dir}{wave}_radiance_maps.png", dpi=150, bbox_inches='tight')
        #plt.savefig(f"{dir}{wave}_calibration_merid_profiles.eps", dpi=900)
        # Clear figure to avoid overlapping between plotting subroutines
        plt.clf()
