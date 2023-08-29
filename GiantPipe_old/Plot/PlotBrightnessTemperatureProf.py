import os
import numpy as np
import matplotlib.pyplot as plt
import Globals
from Tools.VisirFilterInfo import Wavenumbers, Wavelengths
from Read.ReadZonalWind import ReadZonalWind

def PlotCompositeTBprofile(dataset):
    """ Plotting thermal shear using stored global maps numpy array """

    print('Plotting composite figure of brightness temperature profiles...')
    # If subdirectory does not exist, create it
    dir = f'../outputs/{dataset}/brightness_temperature_profiles/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Initialize some local variales
    lat = np.arange(-89.75,90,step=0.5)               # Latitude range from pole-to-pole
    globalmaps = np.empty((Globals.nfilters, Globals.ny, Globals.nx))
    zonalmean = np.empty((Globals.nfilters, Globals.ny))
    globalmaps.fill(np.nan)
    zonalmean.fill(np.nan)
    Nfilters = Globals.nfilters if dataset == '2018May' else 11
    #  Load Jupiter zonal jets data to determine belts and zones location
    ejets_c, wjets_c, nejet, nwjet = ReadZonalWind("../inputs/jupiter_jets.dat")
    
    for ifilt in range(Nfilters):
        if dataset == '2018May':
            filt = Wavenumbers(ifilt)
            adj_location = 'average' if ifilt < 10 else 'southern'
            globalmaps[ifilt, :, :] = np.load(f'../outputs/{dataset}/global_maps_figures/calib_{filt}_global_maps_{adj_location}_adj.npy')
        elif dataset == '2022July' or dataset == '2022August':
            if ifilt == 4: 
                filt = Wavenumbers(ifilt+1)
                ifilt_up = ifilt+1
            elif ifilt > 5:
                filt = Wavenumbers(ifilt+2)
                ifilt_up = ifilt+2
            else:
                filt = Wavenumbers(ifilt)
                ifilt_up = ifilt
            globalmaps[ifilt_up, :, :] = np.load(f'../outputs/{dataset}/global_maps_figures/calib_{filt}_global_maps.npy')
    for ifilt in range(Globals.nfilters):
        # Zonal mean of the global maps
        for iy in range(Globals.ny):
            zonalmean[ifilt, iy] = np.nanmean(globalmaps[ifilt, iy, :])
    
    # Create a composite figure with all filters
    Nlines = Globals.nfilters if dataset == '2018May' else 10
    fig, axes = plt.subplots(Nlines, 1, figsize=(12,16), sharex=True)
    iaxes = 0
    subplot_array = [0,10,11,12,5,4,6,7,8,9,3,2,1] if dataset == '2018May' else [0,10,11,12,5,8,9,3,2,1]
    for ifilt in subplot_array:
        wavelength = Wavelengths(ifilt)
        axes[iaxes].plot(lat[:],zonalmean[ifilt,:],linewidth=3.0,color="black",label=f"{wavelength}"+"$\mu$m")
        for iejet in range(0,nejet):
            axes[iaxes].plot([ejets_c[iejet],ejets_c[iejet]],[np.nanmin(zonalmean[ifilt, :]),np.nanmax(zonalmean[ifilt, :])],color='black',linestyle="dashed")
        for iwjet in range(0,nwjet):
            axes[iaxes].plot([wjets_c[iwjet],wjets_c[iwjet]],[np.nanmin(zonalmean[ifilt, :]),np.nanmax(zonalmean[ifilt, :])],color='black',linestyle="dotted")
        axes[iaxes].plot([-90,90],[np.nanmean(zonalmean[ifilt, :]),np.nanmean(zonalmean[ifilt, :])],linewidth=1.0,color="grey")
        axes[iaxes].set_xlim(-90,90)
        axes[iaxes].legend(loc="upper right", fontsize=12, handletextpad=0, handlelength=0, markerscale=0)
        axes[iaxes].tick_params(labelsize=20)
        # hide tick and tick label of the big axis
        plt.axes([0.08, 0.1, 0.8, 0.8], frameon=False) 
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)        
        plt.xlabel("Planetocentric Latitude", size=25)
        plt.ylabel("Zonal-mean Brightness Temperature [K]", size=25)
        iaxes += 1
    # Save figure 
    plt.savefig(f"{dir}calib_birghtness_temperature.png", dpi=300)
    plt.savefig(f"{dir}calib_birghtness_temperature.eps", dpi=300)
    plt.close()