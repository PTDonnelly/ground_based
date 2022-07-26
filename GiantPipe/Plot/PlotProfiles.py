import os 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from copy import copy
import Globals
from Read.ReadCal import ReadCal
from Tools.SetWave import SetWave

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
        fig, axes = plt.subplots(2, 1, sharex=True, sharey=True)
        # subplot showing the averaging of each singles merid profiles (ignoring negative beam)
        for ifile, fname in enumerate(files):
            _, _, iwave, _, _ = SetWave(filename=fname, wavelength=None, wavenumber=None, ifilt=None)
            if iwave == wave:
                axes[0].plot(singles[:, ifile, 0], singles[:, ifile, 3], color='black', lw=0, marker='.', markersize=2)
        axes[0].plot(spectrals[:, ifilt_v, 0], spectrals[:, ifilt_v, 3], color='orange', lw=0, marker='o', markersize=2, label='VLT/VISIR av')
        axes[0].set_title(f"{wave}"+" cm$^{-1}$")
        axes[0].set_xlim((-90, 90))
        axes[0].legend()

        # subplot showing the calibration of the spectral merid profile to spacecraft data
        if ifilt_sc < 12:
            # Use CIRS for N-Band
            axes[1].plot(cirs[:, ifilt_sc, 0], cirs[:, ifilt_sc, 1], color='k', lw=1, label='Cassini/CIRS')
        else:
            # Use IRIS for Q-Band
            axes[1].plot(iris[:, ifilt_sc, 0], iris[:, ifilt_sc, 1], color='k', lw=1, label='Voyager/IRIS')
        axes[1].plot(spectrals[:, ifilt_v, 0], spectrals[:, ifilt_v, 3], color='orange', lw=0, marker='o', markersize=3, label='VLT/VISIR av')
        axes[1].set_xlim((-90, 90))
        axes[1].legend()

        # add a big axis, hide frame
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Latitude", size=15)
        plt.ylabel("Radiance (W cm$^{-2}$ sr$^{-1}$ (cm$^{-1}$)$^{-1}$)", size=15)

        # Save figure showing calibation method 
        _, _, wave, _, _ = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=ifilt)
        plt.savefig(f"{dir}{wave}_calibration_merid_profiles.png", dpi=900)
        #plt.savefig(f"{dir}{wave}_calibration_merid_profiles.eps", dpi=900)
        # Clear figure to avoid overlapping between plotting subroutines
        plt.clf()

def PlotGlobalSpectrals(dataset, spectrals):
    """Basic code to plot the central meridian profiles with wavenumber
    (or spectral profiles with latitude, depending on the persepctive).
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
    plt.savefig(f"{dir}global_spectrals.png", dpi=900)
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
    plt.savefig(f"{dir}global_spectrals_res1.png", dpi=900)
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
    plt.savefig(f"{dir}global_spectrals_res2.png", dpi=900)
    plt.close()

def PlotCentreTotLimbProfiles(mode, singles, spectrals):
    print('Plotting profiles...')
    
    a = 1

def ColorNuance(colorm, ncolor, i):
    pal = get_cmap(name=colorm)
    coltab = [pal(icolor) for icolor in np.linspace(0,0.9,ncolor)]
    
    return coltab[i]

