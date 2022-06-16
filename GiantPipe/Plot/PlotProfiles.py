import os 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import Globals
from Read.ReadCal import ReadCal
from Tools.VisirFilterInfo import Wavenumbers
from Tools.SetWave import SetWave

def PlotMeridProfiles(singles, spectrals, ksingles, kspectrals, wavenumber):
    """ DB: Plot meridian profiles and spacecraft data to illustrate 
            the calibration method """
    
    # If wavenumber=False, we need to retrieve wavenumber from each "singles" file/array
    #if wavenumber=False:
    #    print("blablabla, you have to code it!")

    # Read in Voyager and Cassini data into arrays
    calfile = "../inputs/visir.jup.filtered-iris-cirs.10-12-15.data.v3"
    iris, cirs = ReadCal(calfile)

    print('Plotting profiles...')
    # If subdirectory does not exist, create it
    dir = '../outputs/calibration_profiles_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    for ifilt in range(Globals.nfilters):
        # Get filter index for plotting spacecraft and calibrated data
        waves = spectrals[:, ifilt, 5]
        wave  = waves[(waves > 0)][0]

        # Get filter index for spectral profiles
        _, _, ifilt_sc, ifilt_v = SetWave(wavelength=False, wavenumber=wave)
        # Create a figure per filter
        #plt.figure(dpi=900)
        # subplot showing the averaging of each singles merid profiles (ignoring negative beam)
        ax1 = plt.subplot2grid((2, 1), (0, 0))
        for ifile, iwave in enumerate(wavenumber):
            if iwave == wave:
                ax1.plot(singles[:, ifile, 0], singles[:, ifile, 3]*ksingles[ifile, 1], color='midnightblue', lw=0, marker='.', markersize=3)
        ax1.plot(spectrals[:, ifilt_v, 0], spectrals[:, ifilt_v, 3]*kspectrals[ifilt_v, 1], color='skyblue', lw=0, marker='o', markersize=3, label='VLT/VISIR av')
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
        ax2.plot(spectrals[:, ifilt_v, 0], spectrals[:, ifilt_v, 3], color='orange', lw=0, marker='v', markersize=3, label='VLT/VISIR calib')
        ax2.plot(spectrals[:, ifilt_v, 0], spectrals[:, ifilt_v, 3]*kspectrals[ifilt_v, 1], color='skyblue', lw=0,  marker='o', markersize=3, label='VLT/VISIR av')
        ax2.set_xlim((-90, 90))
        ax2.set_xlabel('Latitude', size=15)
        #ax2.set_ylim((0, 20e-8))
        ax2.set_ylabel('Radiance (W)', size=15)
        ax2.legend()
        # Save figure showing calibation method 
        filt = Wavenumbers(ifilt)
        plt.savefig(f"{dir}{filt}_calibration_merid_profiles.png", dpi=900)
        plt.savefig(f"{dir}{filt}_calibration_merid_profiles.eps", dpi=900)
    # Clear figure to avoid overlapping between plotting subroutines
    plt.clf()

def PlotCalMeridProfiles(singles, spectrals, wavenumber):
    """ DB: Plot calibrated meridian profiles and spacecraft data FOR PAPER! """
    
    # If wavenumber=False, we need to retrieve wavenumber from each "singles" file/array
    #if wavenumber=False:
    #    print("blablabla, you have to code it!")

    # Read in Voyager and Cassini data into arrays
    calfile = "../inputs/visir.jup.filtered-iris-cirs.10-12-15.data.v3"
    iris, cirs = ReadCal(calfile)

    print('Plotting profiles...')
    # If subdirectory does not exist, create it
    dir = '../outputs/calibration_profiles_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    for ifilt in range(Globals.nfilters):
        # Get filter index for plotting spacecraft and calibrated data
        waves = spectrals[:, ifilt, 5]
        wave  = waves[(waves > 0)][0]

        # Get filter index for spectral profiles
        _, _, ifilt_sc, ifilt_v = SetWave(wavelength=False, wavenumber=wave)
        # Create a figure per filter
        #plt.figure(dpi=900)
        # subplot showing the averaging of each singles merid profiles (ignoring negative beam)
        ax1 = plt.subplot2grid((2, 1), (0, 0))
        for ifile, iwave in enumerate(wavenumber):
            if iwave == wave:
                ax1.plot(singles[:, ifile, 0], singles[:, ifile, 3], color='chocolate', lw=0, marker='.', markersize=3)
        ax1.plot(spectrals[:, ifilt_v, 0], spectrals[:, ifilt_v, 3], color='orange', lw=0, marker='o', markersize=3, label='VLT/VISIR')
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
        ax2.plot(spectrals[:, ifilt_v, 0], spectrals[:, ifilt_v, 3], color='orange', lw=0,  marker='o', markersize=3, label='VLT/VISIR')
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

def ColorNuance(colorm, ncolor, i):
    pal = get_cmap(name=colorm)
    coltab = [pal(icolor) for icolor in np.linspace(0,0.9,ncolor)]
    
    return coltab[i]

