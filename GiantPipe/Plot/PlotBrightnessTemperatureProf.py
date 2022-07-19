import os
import numpy as np
import matplotlib.pyplot as plt
import Globals
from Tools.VisirFilterInfo import Wavenumbers, Wavelengths
from Read.ReadZonalWind import ReadZonalWind

def PlotCompositeTBprofile():
    """ Plotting thermal shear using stored global maps numpy array """

    print('Plotting composite figure of brightness temperature profiles...')
    # If subdirectory does not exist, create it
    dir = '../outputs/brightness_temperature_profiles/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Initialize some local variales
    nx, ny = 720, 360
    lat = np.arange(-89.75,90,step=0.5)               # Latitude range from pole-to-pole
    globalmaps = np.empty((Globals.nfilters, ny, nx))
    zonalmean = np.empty((Globals.nfilters, ny))
    #  Load Jupiter zonal jets data to determine belts and zones location
    ejets_c, wjets_c, nejet, nwjet = ReadZonalWind("../inputs/jupiter_jets.dat")
    
    for ifilt in range(Globals.nfilters):
        filt = Wavenumbers(ifilt)
        adj_location = 'average' if ifilt < 10 else 'southern'
        globalmaps[ifilt, :, :] = np.load(f'../outputs/global_maps_figures/calib_{filt}_global_maps_{adj_location}_adj.npy')
        # Zonal mean of the global maps
        for iy in range(ny):
            zonalmean[ifilt, iy] = np.nanmean(globalmaps[ifilt, iy, :])
    
    # Create a composite figure with all filters
    fig, axes = plt.subplots(Globals.nfilters, 1, figsize=(12,16), sharex=True)
    iaxes = 0
    for ifilt in [0,10,11,12,5,4,6,7,8,9,3,2,1]:
        wavelength = Wavelengths(ifilt)
        axes[iaxes].plot(lat[:],zonalmean[iaxes,:],linewidth=3.0,color="black",label=f"{wavelength}"+"$\mu$m")
        for iejet in range(0,nejet):
            axes[iaxes].plot([ejets_c[iejet],ejets_c[iejet]],[np.nanmin(zonalmean[iaxes, :]),np.nanmax(zonalmean[iaxes, :])],color='black',linestyle="dashed")
        for iwjet in range(0,nwjet):
            axes[iaxes].plot([wjets_c[iwjet],wjets_c[iwjet]],[np.nanmin(zonalmean[iaxes, :]),np.nanmax(zonalmean[iaxes, :])],color='black',linestyle="dotted")
        axes[iaxes].plot([-90,90],[np.nanmean(zonalmean[iaxes, :]),np.nanmean(zonalmean[iaxes, :])],linewidth=1.0,color="grey")
        axes[iaxes].set_xlim(-90,90)
        axes[iaxes].legend(loc="upper right", fontsize=12, handletextpad=0, handlelength=0, markerscale=0)
        axes[iaxes].tick_params(labelsize=20)
        # hide tick and tick label of the big axis
        plt.axes([0.1, 0.1, 0.8, 0.8], frameon=False) 
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)        
        plt.xlabel("Planetocentric Latitude", size=25)
        plt.ylabel("Zonal-mean Brightness Temperature [K]", size=25)
        iaxes += 1
    # Save figure 
    plt.savefig(f"{dir}calib_birghtness_temperature.png", dpi=300)
    plt.savefig(f"{dir}calib_birghtness_temperature.eps", dpi=300)
    plt.clf()