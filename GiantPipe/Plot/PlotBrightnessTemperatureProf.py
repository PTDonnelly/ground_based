import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import Globals
from Tools.SetWave import SetWave
from Read.ReadZonalWind import ReadZonalWind
from Read.ReadGravity import ReadGravity

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
    dT_dy = np.empty((Globals.nfilters, Globals.ny))
    globalmaps.fill(np.nan)
    zonalmean.fill(np.nan)
    dT_dy.fill(np.nan)
    Nfilters = Globals.nfilters if dataset == '2018May' else 11
    #  Load Jupiter zonal jets data to determine belts and zones location
    ejets_c, wjets_c, nejet, nwjet = ReadZonalWind("../inputs/jupiter_jets.dat")
    # Load Jupiter gravity data to calculate pseudo-windshear using TB and mu array array
    grav, Coriolis, y, _, _, _ = ReadGravity("../inputs/jup_grav.dat", lat=lat)
    
    for ifilt in range(Nfilters):
        if (ifilt !=  7) and (ifilt != 6):
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
        dT_dy[ifilt,:] = np.gradient(zonalmean[ifilt, :],y, edge_order=2)
    # Removing the large spike located at the equator coming from the transition between two cylindrcal maps 
    latremoved = (lat <= 0.25) & (lat >= -0.25)
    for ifilt in range(Globals.nfilters):
        # Zonal mean of the global maps
        dT_dy[ifilt,latremoved] = np.nan
    # Interpolating the removing value
    latinterp = (lat <= 1.25) & (lat >= -1.25)
    rawlat = lat[latinterp]
    xnew = np.linspace(np.min(rawlat), np.max(rawlat), num=len(rawlat))
    print(np.shape(xnew))
    for ifilt in range(Globals.nfilters):
        rawdata = dT_dy[ifilt,latinterp]
        interpdata = np.interp(xnew, rawdata,rawlat)
        print(rawlat, rawdata, interpdata)
        # dT_dy[ifilt,latinterp] = interpdata
    # Create a composite figure with all filters
    Nlines = 11
    fig, axes = plt.subplots(Nlines, 1, figsize=(8,18), sharex=True)
    iaxes = 0
    subplot_array = [0,10,11,12,5,4,8,9,3,2,1] if dataset == '2018May' else [0,10,11,12,5,8,9,3,2,1]
    ititle = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)']
    for ifilt in subplot_array:
        if (ifilt !=  7) and (ifilt != 6):
            _, wavelength, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
            axes[iaxes].plot(lat[:],dT_dy[ifilt,:],linewidth=3.0,color="black",label=f"{wavelength}"+"$\mu$m")
            for iejet in range(0,nejet):
                axes[iaxes].plot([ejets_c[iejet],ejets_c[iejet]],[np.nanmin(dT_dy[ifilt, :]),np.nanmax(dT_dy[ifilt, :])],color='black',linestyle="dashed")
                if iaxes==0: print('eastward',ejets_c[iejet])  
            for iwjet in range(0,nwjet):
                axes[iaxes].plot([wjets_c[iwjet],wjets_c[iwjet]],[np.nanmin(dT_dy[ifilt, :]),np.nanmax(dT_dy[ifilt, :])],color='black',linestyle="dotted")
                if iaxes==0: print('westward',wjets_c[iwjet])
            axes[iaxes].plot([-90,90],[np.nanmean(dT_dy[ifilt, :]),np.nanmean(dT_dy[ifilt, :])],linewidth=1.0,color="grey")
            axes[iaxes].set_xlim(-70, 70)
            axes[iaxes].set_xticks([-70, -60, -40, -20, 0, 20, 40, 60, 70])
            axes[iaxes].tick_params(axis='x', labelrotation=45)
            axes[iaxes].tick_params(labelsize=20)
            axes[iaxes].set_title(ititle[iaxes]+f"    {wavelength}"+r" $\mu$m", fontfamily='sans-serif', loc='left', fontsize=20)
            iaxes += 1
    plt.subplots_adjust(hspace=0.5)
    # hide tick and tick label of the big axis
    plt.axes([0.01, 0.09, 0.9, 0.8], frameon=False) 
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)        
    plt.xlabel("Planetocentric Latitude", size=18)
    plt.ylabel("Zonal-mean Brightness Temperature gradient [K/km]", size=18)
    # Save figure 
    plt.savefig(f"{dir}calib_birghtness_temperature_gradient.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{dir}calib_birghtness_temperature_gradient.pdf", dpi=500, bbox_inches='tight')
    plt.close()


    for ifilt in range(Globals.nfilters):
        if (ifilt !=  7) and (ifilt != 6):
            fig = plt.figure( figsize=(8,5))
            subplot_array = [0,10,11,12,5,4,8,9,3,2,1] if dataset == '2018May' else [0,10,11,12,5,8,9,3,2,1]
            
            _, wavelength, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
            plt.plot(lat[:],dT_dy[ifilt,:],linewidth=3.0,color="black",label=f"{wavelength}"+"$\mu$m")
            for iejet in range(0,nejet):
                plt.plot([ejets_c[iejet],ejets_c[iejet]],[np.nanmin(dT_dy[ifilt, :]),np.nanmax(dT_dy[ifilt, :])],color='black',linestyle="dashed")
            for iwjet in range(0,nwjet):
                plt.plot([wjets_c[iwjet],wjets_c[iwjet]],[np.nanmin(dT_dy[ifilt, :]),np.nanmax(dT_dy[ifilt, :])],color='black',linestyle="dotted")
            plt.plot([-90,90],[np.nanmean(dT_dy[ifilt, :]),np.nanmean(dT_dy[ifilt, :])],linewidth=1.0,color="grey")
            plt.xlim(-90,90)
            plt.xticks([-90, -80, -60, -40, -20, 0, 20, 40, 60, 80, 90])
            # plt.ylim(np.nanmin(dT_dy[ifilt, :]),np.nanmax(dT_dy[ifilt, :]))
            plt.tick_params(labelsize=15)       
            plt.xlabel("Planetocentric Latitude", size=18)
            plt.ylabel('[K/km]', size=18)
            plt.title(r'$\frac{\partial \overline{T_B}}{\partial y}$'+f" at {wavelength}"+r'$\mu$m', fontfamily='sans-serif', loc='left', fontsize=18)
            # Save figure 
            plt.savefig(f"{dir}calib_{wavnb}_birghtness_temperature_gradient.png", dpi=150, bbox_inches='tight')
            plt.savefig(f"{dir}calib_{wavnb}_birghtness_temperature_gradient.pdf", dpi=500, bbox_inches='tight')
            plt.close()