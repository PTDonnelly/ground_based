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
cmap = plt.get_cmap("magma")

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
        axes[0].plot(spectrals[:, ifilt_v, 0], spectrals[:, ifilt_v, 3], color='orange', lw=0, marker='o', markersize=2, label=f"averaged VLT/VISIR at {int(wave)}"+" cm$^{-1}$")
        # axes[0].set_title(f"{wave}"+" cm$^{-1}$")
        axes[0].set_xlim((-90, 90))
        axes[0].legend(fontsize=20)

        # subplot showing the calibration of the spectral merid profile to spacecraft data
        if ifilt_sc < 12:
            # Use CIRS for N-Band
            axes[1].plot(cirs[:, ifilt_sc, 0], cirs[:, ifilt_sc, 1], color='k', lw=1, label='Cassini/CIRS')
        else:
            # Use IRIS for Q-Band
            axes[1].plot(iris[:, ifilt_sc, 0], iris[:, ifilt_sc, 1], color='k', lw=1, label='Voyager/IRIS')
        axes[1].plot(spectrals[:, ifilt_v, 0], spectrals[:, ifilt_v, 3], color='orange', lw=0, marker='o', markersize=3, label=f"averaged VLT/VISIR at {int(wave)}"+" cm$^{-1}$")
        axes[1].set_xlim((-90, 90))
        axes[1].legend(fontsize=20)

        # add a big axis, hide frame
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Latitude", size=20)
        plt.ylabel("Radiance (W cm$^{-2}$ sr$^{-1}$ (cm$^{-1}$)$^{-1}$)", size=20)

        # Save figure showing calibation method 
        _, _, wave, _, _ = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=ifilt)
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
                plt.plot(singles[:, ifile, 1], singles[:, ifile, 3], lw=0, marker='.', markersize=2, color = 'black')
        plt.plot(spectrals[:, ifilt_v, 1], spectrals[:, ifilt_v, 3], color='orange', lw=0, marker='o', markersize=2, label=f"{int(wave)}"+" cm$^{-1}$ VLT/VISIR profile at "+f"{Globals.LCP}"+"$^{\circ}$")
        # x = np.arange(0.5, 360.5, 1)
        # y = np.flipud(spectrals[:, ifilt_v, 3])
        # w = np.isnan(y)
        # y[w] = 0.
        # spl = UnivariateSpline(x, y, w=~w, k=5, s=0.01)
        # xs = np.arange(0.5, 360.5, 1)
        # plt.plot(np.flipud(xs), np.flipud(spl(xs)), color='blue', lw=0, marker='.', markersize=1, label=f"1-D smoothing spline fit")
        # # spl2 = UnivariateSpline(x, y, w=~w, k=5, s=1.e9)
        # # plt.plot(np.flipud(xs), np.flipud(spl2(xs)), color='green', lw=0, marker='.', markersize=1, label=f"SPL")
        plt.legend(fontsize=12)
        plt.grid()
        plt.tick_params(labelsize=12) 
        plt.xlabel("System III West Longitude", size=15)
        print(spectrals[0, ifilt_v, 1], spectrals[-1, ifilt_v, 1])
        if spectrals[0, ifilt_v, 1] >=0 and spectrals[-1, ifilt_v, 1]>=0:
            plt.xlim(spectrals[0, ifilt_v, 1], spectrals[-1, ifilt_v, 1]) 
            plt.xticks(ticks=np.arange(360,-1,-30), labels=list(np.arange(360,-1,-30)))
        plt.ylabel("Radiance (W cm$^{-1}$ sr$^{-1}$)", size=15)
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
        # Get filter index for plotting spacecraft and calibrated data
        _, _, wave, _, ifilt = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=ifilt)
        # Create a figure per filter
        fig = plt.subplots(1, 1, figsize=(8, 3))
        im = plt.imshow(spectrals[:, ifilt, 3], origin='lower', vmin=min, vmax=max, cmap='cividis')
        plt.tick_params(labelsize=12) 
        plt.xlabel("System III West Longitude", size=15)
        print(spectrals[0, ifilt, 1], spectrals[-1, ifilt, 1])
        if spectrals[0, ifilt, 1] >=0 and spectrals[-1, ifilt, 1]>=0:
            plt.xlim(spectrals[0, ifilt, 1], spectrals[-1, ifilt, 1]) 
            plt.xticks(ticks=np.arange(360,-1,-30), labels=list(np.arange(360,-1,-30)))
        plt.ylabel("Latitude", size=15)
        cbar = plt.colorbar(im, extend='both')
        cbar.ax.tick_params(labelsize=12)
        cbar.set_label("Radiance (W cm$^{-1}$ sr$^{-1}$)")
        # Save figure showing calibation method 
        plt.savefig(f"{dir}{wave}_parallel_profiles.png", dpi=150, bbox_inches='tight')
        #plt.savefig(f"{dir}{wave}_calibration_merid_profiles.eps", dpi=900)
        # Clear figure to avoid overlapping between plotting subroutines
        plt.clf()
