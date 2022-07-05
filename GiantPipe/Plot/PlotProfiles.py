import os 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from copy import copy
import Globals
from Read.ReadCal import ReadCal
from Tools.SetWave import SetWave

def PlotMeridProfiles(mode, files, singles, spectrals):
    """ DB: Plot meridian profiles and spacecraft data to illustrate 
            the calibration method """

    print('Plotting profiles...')

    # Read in Voyager and Cassini data into arrays
    calfile = "../inputs/visir.jup.filtered-iris-cirs.10-12-15.data.v3"
    iris, cirs = ReadCal(calfile)

    # If subdirectory does not exist, create it
    dir = '../outputs/calibration_profiles_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    for ifilt in range(Globals.nfilters):
        # Get filter index for plotting spacecraft and calibrated data
        _, _, wave, ifilt_sc, ifilt_v = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=ifilt)
        
        # Create a figure per filter
        #plt.figure(dpi=900)
        # subplot showing the averaging of each singles merid profiles (ignoring negative beam)
        ax1 = plt.subplot2grid((2, 1), (0, 0))
        for ifile, fname in enumerate(files):
            _, _, iwave, _, _ = SetWave(filename=fname, wavelength=None, wavenumber=None, ifilt=None)
            if iwave == wave:
                ax1.plot(singles[:, ifile, 0], singles[:, ifile, 3], color='black', lw=0, marker='.', markersize=2)
        ax1.plot(spectrals[:, ifilt_v, 0], spectrals[:, ifilt_v, 3], color='orange', lw=0, marker='o', markersize=3, label='VLT/VISIR av')
        ax1.set_title(wave)
        ax1.set_xlim((-90, 90))
        #ax1.set_ylim((0, 20e-8))
        ax1.set_ylabel('Radiance (W)', size=15)
        ax1.legend()

        # subplot showing the calibration of the spectral merid profile to spacecraft data
        ax2 = plt.subplot2grid((2, 1), (1, 0))
        if ifilt_sc < 12:
            # Use CIRS for N-Band
            ax2.plot(cirs[:, ifilt_sc, 0], cirs[:, ifilt_sc, 1], color='k', lw=1, label='Cassini/CIRS')
        else:
            # Use IRIS for Q-Band
            ax2.plot(iris[:, ifilt_sc, 0], iris[:, ifilt_sc, 1], color='k', lw=1, label='Voyager/IRIS')
        ax2.plot(spectrals[:, ifilt_v, 0], spectrals[:, ifilt_v, 3], color='orange', lw=0, marker='o', markersize=3, label='VLT/VISIR calib')
        # ax2.plot(spectrals[:, ifilt_v, 0], spectrals[:, ifilt_v, 3], color='skyblue', lw=0,  marker='o', markersize=3, label='VLT/VISIR av')
        ax2.set_xlim((-90, 90))
        ax2.set_xlabel('Latitude', size=15)
        #ax2.set_ylim((0, 20e-8))
        ax2.set_ylabel('Radiance (W)', size=15)
        ax2.legend()

        # Save figure showing calibation method 
        _, _, wave, _, _ = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=ifilt)
        plt.savefig(f"{dir}{wave}_calibration_merid_profiles.png", dpi=900)
        # plt.savefig(f"{dir}{filt}_calibration_merid_profiles.eps", dpi=900)
    # Clear figure to avoid overlapping between plotting subroutines
    plt.clf()

def PlotCalMeridProfiles(singles, spectrals):
    """ DB: Plot calibrated meridian profiles and spacecraft data FOR PAPER! """
    
    print('Plotting profiles...')

    # Read in Voyager and Cassini data into arrays
    calfile = "../inputs/visir.jup.filtered-iris-cirs.10-12-15.data.v3"
    iris, cirs = ReadCal(calfile)

    # If subdirectory does not exist, create it
    dir = '../outputs/calibration_profiles_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    for ifilt in range(Globals.nfilters):
        # Get filter index for plotting spacecraft and calibrated data
        waves = spectrals[:, ifilt, 5]
        wave  = waves[(waves > 0)][0]

        # Get filter index for spectral profiles
        _, _, _, ifilt_sc, ifilt_v = SetWave(filename=False, wavelength=False, wavenumber=wave)
        # Create a figure per filter
        #plt.figure(dpi=900)
        # subplot showing the averaging of each singles merid profiles (ignoring negative beam)
        ax1 = plt.subplot2grid((2, 1), (0, 0))
        for ifile, ifname in enumerate(files):
            _, _, wave, _, _ = SetWave(filename=fname, wavelength=False, wavenumber=False)
            if iwave == wave:
                ax1.plot(singles[:, ifile, 0], singles[:, ifile, 3], color='midnightblue', lw=0, marker='.', markersize=3)
        ax1.plot(spectrals[:, ifilt_v, 0], spectrals[:, ifilt_v, 3], color='skyblue', lw=0, marker='o', markersize=3, label='VLT/VISIR')
        ax1.set_xlim((-90, 90))
        #ax1.set_ylim((0, 20e-8))
        ax1.set_ylabel('Radiance (W)', size=15)
        ax1.legend()
        # subplot showing the calibration of the spectral merid profile to spacecraft data
        ax2 = plt.subplot2grid((2, 1), (1, 0))
        if ifilt_sc < 12:
            # Use CIRS for N-Band
            ax2.plot(cirs[:, ifilt_sc, 0], cirs[:, ifilt_sc, 1], color='k', lw=1, label='Cassini/CIRS')
        else:
            # Use IRIS for Q-Band
            ax2.plot(iris[:, ifilt_sc, 0], iris[:, ifilt_sc, 1], color='k', lw=1, label='Voyager/IRIS')
        ax2.plot(spectrals[:, ifilt_v, 0], spectrals[:, ifilt_v, 3], color='skyblue', lw=0,  marker='o', markersize=3, label='VLT/VISIR')
        ax2.set_xlim((-90, 90))
        ax2.set_xlabel('Latitude', size=15)
        #ax2.set_ylim((0, 20e-8))
        ax2.set_ylabel('Radiance (W)', size=15)
        ax2.legend()
        # Save figure showing calibation method 
        filt = Wavenumbers(ifilt)
        plt.savefig(f"{dir}{filt}_calibrated_merid_profiles.png", dpi=900)
        plt.savefig(f"{dir}{filt}_calibrated_merid_profiles.eps", dpi=900)
    # Clear figure to avoid overlapping between plotting subroutines
    plt.clf()

def PlotGlobalSpectrals(spectrals):
    """Basic code to plot the central meridian profiles with wavenumber
    (or spectral profiles with latitude, depending on the persepctive).
    Displays the global pseudo-spectrum in a way resembling a normal spectrum."""
    
    # If subdirectory does not exist, create it
    dir = '../outputs/calibration_profiles_figures/'
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
    plt.colorbar(ax1)
    plt.xlim((xmin, xmax))
    plt.ylim((-90, 90))
    plt.savefig(f"{dir}global_spectrals.png", dpi=900)
    plt.close()
 
    plt.figure
    ax2 = plt.contourf(wave, lat, rad_res1, vmin=-1*np.nanmax(rad_res1),vmax=np.nanmax(rad_res1), levels=200, cmap='seismic')
    for i in range(Globals.nfilters):
        plt.plot((wave[i], wave[i]), (-90, 90), ls=':', lw=0.7, color='black')
    plt.colorbar(ax2)
    plt.xlim((xmin, xmax))
    plt.ylim((-90, 90))
    plt.savefig(f"{dir}global_spectrals_res1.png", dpi=900)
    plt.close()

    plt.figure
    ax3 = plt.contourf(wave, lat, rad_res2, vmin=-1*np.nanmax(rad_res2),vmax=np.nanmax(rad_res2), levels=200, cmap='seismic')
    for i in range(Globals.nfilters):
        plt.plot((wave[i], wave[i]), (-90, 90), ls=':', lw=0.7, color='black')
    plt.colorbar(ax3)
    plt.xlim((xmin, xmax))
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

