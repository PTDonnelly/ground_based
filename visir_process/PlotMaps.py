from fileinput import filename
import os

import numpy as np
import matplotlib.pyplot as plt
from BinningInputs import BinningInputs
from ReadFits import ReadFits
from VisirWavenumbers import VisirWavenumbers
from VisirWavelengths import VisirWavelengths
from ConvertBrightnessTemperature import ConvertBrightnessTemperature

def PlotMaps(files, ksingles, kspectrals):
    """ DB: Mapping global maps for each VISIR filter """

    globalmaps = np.empty(shape=(BinningInputs.nfilters, 360,720))

    print('Mapping and correcting global maps...')
    # If subdirectory does not exist, create it
    dir = '../outputs/global_maps_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Loop over files
    for ifile, fname in enumerate(files):
        imghead, imgdata, cylhead, cyldata, muhead, mudata = ReadFits(filename=f"{fname}")


        for ifilt in range(BinningInputs.nfilters):
            wave = VisirWavelengths(ifilt=ifilt)
            TBmaps = ConvertBrightnessTemperature(cyldata,wavelength=wave)

            # Combinig single cylmaps to store in globalmaps array
            for i in range(0,360):
                for j in range(0,720):
                    globalmaps[i,j] = np.nanmax(TBmaps[:,i,j])

           # Plotting global map
            TBmax = np.nanmax(globalmaps[ifilt,:,:]) 
            TBmin = np.nanmin(globalmaps[ifilt,:,:]) 

            im = plt.imshow(globalmaps, origin='lower', vmin=TBmin, vmax=TBmax, cmap='cividis')
            plt.xticks(np.arange(0, len(cyldata[0])+1, step = 60), list(np.arange(360,-1,-30)))
            plt.yticks(np.arange(0, len(cyldata)+1, step = 60), list(np.arange(-90,91,30)))
            plt.set_xlabel('System III West Longitude', size=15)
            plt.set_ylabel('Planetocentric Latitude', size=15)
            plt.tick_params(labelsize=15)
            cbar = plt.colorbar(im, extend='both')
            cbar.ax.tick_params(labelsize=15)
            cbar.set_label("Brightness Temperature [K]", size=15)

            # Save figure showing calibation method 
            filt = VisirWavenumbers(ifilt)
            plt.savefig(f"{dir}{filt}_global_maps.png", dpi=900)
            plt.savefig(f"{dir}{filt}_global_maps.eps", dpi=900)


    return globalmaps
