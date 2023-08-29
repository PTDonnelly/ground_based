import os
import numpy as np
import matplotlib.pyplot as plt
import Globals
from Tools.SetWave import SetWave
from Read.ReadZonalWind import ReadZonalWind
from Read.ReadGravity import ReadGravity


def PlotPseudoWindShear(dataset):
    """ Plotting thermal shear using calculated global maps array """

    print('Plotting pseudo wind shear...')
    # If subdirectory does not exist, create it
    dir = f'../outputs/{dataset}/pseudo_wind_shear_profiles_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Initialize some local variales
    lat = np.arange(-89.75,90,step=0.5)               # Latitude range from pole-to-pole
    globalmaps = np.empty((Globals.nfilters, Globals.ny, Globals.nx))
    zonalmean = np.empty((Globals.nfilters, Globals.ny))
    windshear = np.empty((Globals.nfilters, Globals.ny))
    Nfilters = Globals.nfilters if dataset == '2018May' else 11
    #  Load Jupiter zonal jets data to determine belts and zones location
    ejets_c, wjets_c, nejet, nwjet = ReadZonalWind("../inputs/jupiter_jets.dat")
    # Load Jupiter gravity data to calculate pseudo-windshear using TB and mu array array
    grav, Coriolis, y, _, _, _ = ReadGravity("../inputs/jup_grav.dat", lat=lat)


    for ifilt in range(Nfilters):
        if dataset == '2018May':
            _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
            adj_location = 'average' if ifilt < 10 else 'southern'
            globalmaps[ifilt, :, :] = np.load(f'../outputs/{dataset}/global_maps_figures/calib_{wavnb}_global_maps_{adj_location}_adj.npy')
        elif dataset == '2022July' or dataset == '2022August':
            if ifilt == 4: 
                _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt+1)
            elif ifilt > 5:
                _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt+2)
            else:
                _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
            globalmaps[ifilt, :, :] = np.load(f'../outputs/{dataset}/global_maps_figures/calib_{wavnb}_global_maps.npy')
        # Zonal mean of the gloal maps
        for iy in range(Globals.ny):
            zonalmean[ifilt, iy] = np.nanmean(globalmaps[ifilt, iy, :])
        # Calculated the associated thermal/pseudo-windshear
        windshear[ifilt,:]=-(grav/(Coriolis*zonalmean[ifilt,:]))*np.gradient(zonalmean[ifilt, :],y)

    # Create a figure per filter
    for ifilt in range(Nfilters):
        if dataset == '2018May':
            _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
        elif dataset == '2022July' or dataset == '2022August':
            if ifilt == 4:  _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt+1)
            elif ifilt > 5:
                _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt+2)
            else:
                _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
        latkeep = (lat <-5)
        axes[0].plot(lat[latkeep],windshear[ifilt,latkeep],linewidth=3.0,color="black")
        negkeep = (lat <-5) & (windshear[ifilt,:] < 0)
        axes[0].plot(lat[negkeep],windshear[ifilt,negkeep],"bo")
        poskeep = (lat <-5) & (windshear[ifilt,:] > 0)
        axes[0].plot(lat[poskeep],windshear[ifilt,poskeep],"ro")
        for iejet in range(0,nejet):
            axes[0].plot([ejets_c[iejet],ejets_c[iejet]],[-15,15],color='black',linestyle="dashed")
        for iwjet in range(0,nwjet):
            axes[0].plot([wjets_c[iwjet],wjets_c[iwjet]],[-15,15],color='black',linestyle="dotted")
        axes[0].plot([-90,-10],[0,0],linewidth=1.0,color="grey")
        axes[0].set_xlim(-60,-20)
        axes[0].set_ylim(-0.7,0.7)
        # Subplot for the northern hemisphere
        latkeep = (lat > 5)       
        axes[1].plot(lat[latkeep],windshear[ifilt,latkeep],linewidth=3.0,color="black")
        negkeep = (lat > 5) & (windshear[ifilt,:] < 0)
        axes[1].plot(lat[negkeep],windshear[ifilt,negkeep],"bo")
        poskeep = (lat > 5) & (windshear[ifilt,:] > 0)
        axes[1].plot(lat[poskeep],windshear[ifilt,poskeep],"ro")
        axes[1].set_xlim(20,60)
        axes[1].set_ylim(-0.7,0.7)
        for iejet in range(0,nejet):
            axes[1].plot([ejets_c[iejet],ejets_c[iejet]],[-15,15],color='black',linestyle="dashed")
        for iwjet in range(0,nwjet):
            axes[1].plot([wjets_c[iwjet],wjets_c[iwjet]],[-15,15],color='black',linestyle="dotted")
        axes[1].plot([10,90],[0,0],linewidth=1.0,color="grey")
        # add a big axis, hide frame
        fig.add_subplot(111, frameon=False)
        # hide tick and tick label of the big axis
        plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
        plt.title(f"{wavnb}"+" (cm$^{-1}$)")
        plt.xlabel("Latitude", size=15)
        plt.ylabel("Pseudo-shear (m s$^{-1}$ km$^{-1}$)", size=15)
        # Save figure
        if dataset == '2018May':
            adj_location = 'average' if ifilt < 10 else 'southern'
            plt.savefig(f"{dir}calib_{wavnb}_pseudo_wind_shear_{adj_location}_adj.png", dpi=150, bbox_inches='tight')
            plt.savefig(f"{dir}calib_{wavnb}_pseudo_wind_shear_{adj_location}_adj.eps", dpi=150, bbox_inches='tight')
        else:
            plt.savefig(f"{dir}calib_{wavnb}_pseudo_wind_shear.png", dpi=150, bbox_inches='tight')
            plt.savefig(f"{dir}calib_{wavnb}_pseudo_wind_shear.eps", dpi=150, bbox_inches='tight')

def PlotCompositePseudoWindShear(dataset):
    """ Plotting thermal shear using stored global maps numpy array """

    print('Plotting composite figure of pseudo wind shear...')
    # If subdirectory does not exist, create it
    dir = f'../outputs/{dataset}/pseudo_wind_shear_profiles_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Initialize some local variales
    lat = np.arange(-89.75,90,step=0.5)               # Latitude range from pole-to-pole
    globalmaps = np.empty((Globals.nfilters, Globals.ny, Globals.nx))
    zonalmean = np.empty((Globals.nfilters, Globals.ny))
    windshear = np.empty((Globals.nfilters, Globals.ny))
    globalmaps.fill(np.nan)
    zonalmean.fill(np.nan)
    windshear.fill(np.nan)
    Nfilters = Globals.nfilters if dataset == '2018May' else 11
    #  Load Jupiter zonal jets data to determine belts and zones location
    ejets_c, wjets_c, nejet, nwjet = ReadZonalWind("../inputs/jupiter_jets.dat")
    # Load Jupiter gravity data to calculate pseudo-windshear using TB and mu array array
    grav, Coriolis, y, _, _, _ = ReadGravity("../inputs/jup_grav.dat", lat=lat)

    for ifilt in range(Nfilters):
        if dataset == '2018May':
            _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
            adj_location = 'average' if ifilt < 10 else 'southern'
            globalmaps[ifilt, :, :] = np.load(f'../outputs/{dataset}/global_maps_figures/calib_{wavnb}_global_maps_{adj_location}_adj.npy')
        elif dataset == '2022July' or dataset == '2022August':
            if ifilt == 4: 
                _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt+1)
                ifilt_up = ifilt+1
            elif ifilt > 5:
                _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt+2)
                ifilt_up = ifilt+2
            else:
                _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
                ifilt_up = ifilt
            globalmaps[ifilt_up, :, :] = np.load(f'../outputs/{dataset}/global_maps_figures/calib_{wavnb}_global_maps.npy')
    for ifilt in range(Globals.nfilters):
        # Zonal mean of the global maps
        for iy in range(Globals.ny):
            zonalmean[ifilt, iy] = np.nanmean(globalmaps[ifilt, iy, :])
        # Calculated the associated thermal/pseudo-windshear
        windshear[ifilt,:]=-(grav/(Coriolis*zonalmean[ifilt,:]))*np.gradient(zonalmean[ifilt, :],y)
    print(np.shape(windshear))
    # Create a composite figure with all filters
    Nlines = Globals.nfilters if dataset == '2018May' else 10
    fig, axes = plt.subplots(Nlines, 2, figsize=(12,16), sharey=True)
    iaxes = 0
    subplot_array = [0,10,11,12,5,4,6,7,8,9,3,2,1] if dataset == '2018May' else [0,10,11,12,5,8,9,3,2,1]
    for ifilt in subplot_array:
        _, wavelength, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
        # Subplot for the southern hemisphere
        latkeep = (lat <-5)
        axes[iaxes,0].plot(lat[latkeep],windshear[ifilt,latkeep],linewidth=3.0,color="black")
        negkeep = (lat <-5) & (windshear[ifilt,:] < 0)
        axes[iaxes,0].plot(lat[negkeep],windshear[ifilt,negkeep],"bo")
        poskeep = (lat <-5) & (windshear[ifilt,:] > 0)
        axes[iaxes,0].plot(lat[poskeep],windshear[ifilt,poskeep],"ro")
        for iejet in range(0,nejet):
            axes[iaxes,0].plot([ejets_c[iejet],ejets_c[iejet]],[-15,15],color='black',linestyle="dashed")
        for iwjet in range(0,nwjet):
            axes[iaxes,0].plot([wjets_c[iwjet],wjets_c[iwjet]],[-15,15],color='black',linestyle="dotted")
        axes[iaxes,0].plot([-90,-10],[0,0],linewidth=1.0,color="grey")
        axes[iaxes,0].set_xlim(-60,-20)
        axes[iaxes,0].xaxis.set_ticklabels([]) if (iaxes < len(subplot_array)-1) else axes[iaxes,0].tick_params(labelsize=20)
        axes[iaxes,0].set_ylim(-0.7,0.7)
        axes[iaxes,0].tick_params(labelsize=20)
        # Subplot for the northern hemisphere
        latkeep = (lat > 5)       
        axes[iaxes,1].plot(lat[latkeep],windshear[ifilt,latkeep],linewidth=3.0,color="black",label=f"{wavelength}"+"$\mu$m")
        negkeep = (lat > 5) & (windshear[ifilt,:] < 0)
        axes[iaxes,1].plot(lat[negkeep],windshear[ifilt,negkeep],"bo")
        poskeep = (lat > 5) & (windshear[ifilt,:] > 0)
        axes[iaxes,1].plot(lat[poskeep],windshear[ifilt,poskeep],"ro")
        axes[iaxes,1].set_xlim(20,60)
        axes[iaxes,1].xaxis.set_ticklabels([]) if (iaxes < len(subplot_array)-1) else axes[iaxes,1].tick_params(labelsize=20)
        axes[iaxes,1].set_ylim(-0.7,0.7)
        axes[iaxes,1].tick_params(labelsize=20)
        axes[iaxes,1].legend(loc="upper right", fontsize=12, handletextpad=0, handlelength=0, markerscale=0)
        for iejet in range(0,nejet):
            axes[iaxes,1].plot([ejets_c[iejet],ejets_c[iejet]],[-15,15],color='black',linestyle="dashed")
        for iwjet in range(0,nwjet):
            axes[iaxes,1].plot([wjets_c[iwjet],wjets_c[iwjet]],[-15,15],color='black',linestyle="dotted")
        axes[iaxes,1].plot([10,90],[0,0],linewidth=1.0,color="grey")
        # hide tick and tick label of the big axis
        plt.axes([0.1, 0.1, 0.8, 0.8], frameon=False) 
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.xlabel("Planetocentric Latitude", size=25)
        plt.ylabel("Pseudo-shear (m s$^{-1}$ km$^{-1}$)", size=25)
        print(iaxes, ifilt)
        iaxes += 1
    # Save figure 
    plt.savefig(f"{dir}calib_pseudo_wind_shear.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{dir}calib_pseudo_wind_shear.eps", dpi=150, bbox_inches='tight')
    plt.close()
