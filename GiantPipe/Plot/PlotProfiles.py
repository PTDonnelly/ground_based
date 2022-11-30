import os 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import Globals
from Read.ReadCal import ReadCal
from Tools.VisirFilterInfo import Wavenumbers
from Tools.SetWave import SetWave

def PlotMeridProfiles(dataset, files, singles, spectrals):
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
        filt = Wavenumbers(ifilt)
        plt.savefig(f"{dir}{filt}_calibration_merid_profiles.png", dpi=150, bbox_inches='tight')
        plt.savefig(f"{dir}{filt}_calibration_merid_profiles.eps", dpi=150, bbox_inches='tight')
    # Clear figure to avoid overlapping between plotting subroutines
    plt.clf()


def ColorNuance(colorm, ncolor, i):
    pal = get_cmap(name=colorm)
    coltab = [pal(icolor) for icolor in np.linspace(0,0.9,ncolor)]
    
    return coltab[i]

