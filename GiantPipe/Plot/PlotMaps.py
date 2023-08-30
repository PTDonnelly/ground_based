import os
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
from matplotlib import colors
from matplotlib import ticker
from matplotlib.ticker import FormatStrFormatter
import cartopy.crs as ccrs
import operator
import datetime
import Globals
from Tools.CorrectMaps import GetCylandMuMaps, PolynomialAdjust, ApplyPolynom, BlackLineRemoving, MuNormalization
from Tools.SetWave import SetWave


def GlobalMapsNetCDF(dir, filt, globalmaps):
    """Function to save into NetCDF file"""

    import netCDF4 as nc

    fn = f"{dir}{filt}_global_maps.nc"
    data = nc.Dataset(fn, 'w', format='NETCDF4')

    time = data.createDimension('time', None)
    lat = data.createDimension('lat', Globals.ny)
    lon = data.createDimension('lon', Globals.nx)

    times = data.createVariable('time', 'f4', ('time',))
    lats = data.createVariable('lat', 'f4', ('lat',))
    lons = data.createVariable('lon', 'f4', ('lon',))
    value = data.createVariable('value', 'f4', ('time', 'lat', 'lon',))
    value.units = 'Unknown'
    lats[:] = np.arange(-89.75,90,step=0.5)
    lons[:] = np.arange(0, 360, step=0.5)
    value[0, :, :] = globalmaps[:, :]
    data.close()

def PlotMaps(dataset, files, spectrals):
    """ Mapping global maps for each VISIR filter """

    print('Correcting global maps...')
    # If subdirectory does not exist, create it
    dir = f'../outputs/{dataset}/global_maps_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    # Define local inputs
    Nfiles = len(files)
    
    # Create np.arrays for all pixels in all cmaps and mumaps
    cmaps      = np.empty((Nfiles, Globals.ny, Globals.nx))
    mumaps     = np.empty((Nfiles, Globals.ny, Globals.nx))
    wavenumber = np.empty(Nfiles)
    TBmaps     = np.empty((Nfiles, Globals.ny, Globals.nx))
    globalmaps = np.empty((Globals.nfilters, Globals.ny, Globals.nx))

    # Calling the correction method chosen for this dataset
    if dataset == '2018May': # or dataset == '2018May_completed':
        cmaps, mumaps, wavenumber, adj_location = PolynomialAdjust(dir, files, spectrals)
        mumin = [0.02, 0.02, 0.1, 0.08, 0.01, 0.05, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01] if dataset == '2018May' else [0.02, 0.02, 0.1, 0.08, 0.01, 0.05, 0.02, 0.02, 0.02, 0.02, 0.0, 0.0, 0.01]
        Nfilters = Globals.nfilters
    elif dataset == '2022July':
        cmaps, mumaps, wavenumber = BlackLineRemoving(dir, files, cblack=-60, mu_scaling=True)
        mumin = np.empty(13)
        mumin.fill(0.01)
        Nfilters = 10
    elif dataset == '2018May_completed':
        cmaps, mumaps, wavenumber = ApplyPolynom(dir, files, spectrals)
        mumin = [0.02, 0.02, 0.1, 0.08, 0.01, 0.05, 0.02, 0.02, 0.02, 0.02, 0.0, 0.0, 0.01]
        Nfilters = Globals.nfilters
    # else:
    #     cmaps, mumaps, wavenumber = MuNormalization(files)
    #     mumin = [0.02, 0.05, 0.1, 0.08, 0.05, 0.05, 0.0, 0.0, 0.1, 0.08, 0.15, 0.05, 0.05]
    #     Nfilters = 13 if dataset == '2018May_completed' else 10

    print('Mapping global maps...')

    for ifilt in range(Nfilters):
        if (ifilt !=  7) and (ifilt != 6):
            # Empty local TBmaps array to avoid overlapping between filter
            TBmaps[:, :, :] = np.nan
            # Get filter index for spectral profiles
            waves = spectrals[:, ifilt, 5]
            if waves[(waves > 0)] != []:
                wave  = waves[(waves > 0)][0]
                for ifile, iwave in enumerate(wavenumber):
                    if iwave == wave:
                        # Store only the cmaps for the current ifilt 
                        TBmaps[ifile, :, :] = cmaps[ifile, :, :]                
                    
                        res = ma.masked_where(mumaps[ifile, :, :] < mumin[ifilt], TBmaps[ifile, :, :])
                        res = ma.masked_where(((res > 201)), res)
                        res = ma.masked_where(((res < 100)), res)
                        TBmaps[ifile,:,:] = res.filled(np.nan)

                # Combinig single cylmaps to store in globalmaps array
                for y in range(Globals.ny):
                    for x in range(Globals.nx):
                        globalmaps[ifilt, y, x] = np.nanmax(TBmaps[:, y, x])

                # Setting brightness temperature extremes to plot global map
                max = np.nanmax(globalmaps[ifilt, :, :]) 
                min = np.nanmin(globalmaps[ifilt, :, :]) 
                # Plotting global map
                fig = plt.figure(figsize=(8, 3))
                plt.imshow(globalmaps[ifilt, :, :], origin='lower', vmin=min, vmax=max, cmap='inferno')
                plt.xticks(np.arange(0, Globals.nx+1,  step = 60), list(np.arange(360,-1,-30)))
                plt.yticks(np.arange(0, Globals.ny+1, step = 60), list(np.arange(-90,91,30)))
                plt.xlabel('System III West Longitude', size=15)
                plt.ylabel('Planetocentric Latitude', size=15)
                plt.tick_params(labelsize=12)
                cbar = plt.colorbar(extend='both', fraction=0.046, pad=0.014)
                cbar.ax.tick_params(labelsize=12)
                cbar.set_label("Brightness Temperature [K]")

                # Save global map figure of the current filter 
                _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
                if dataset == '2018May':
                    adj_location = 'average' if ifilt < 10 else 'southern'
                    plt.savefig(f"{dir}calib_{wavnb}_global_maps_{adj_location}_adj.png", dpi=150, bbox_inches='tight')
                    # plt.savefig(f"{dir}calib_{wavnb}_global_maps_{adj_location}_adj.eps", dpi=150, bbox_inches='tight')
                    # Clear figure to avoid overlapping between plotting subroutines
                    plt.close()
                    # Write global maps to np.array
                    np.save(f"{dir}calib_{wavnb}_global_maps_{adj_location}_adj", globalmaps[ifilt, :, :])
                    # Write global maps to txtfiles
                    # np.savetxt(f"{dir}calib_{wavnb}_global_maps_{adj_location}_adj.txt", globalmaps[ifilt, :, :])
                    # Write global maps to NetCDF files
                    #GlobalMapsNetCDF(dir, wavnb, globalmaps=globalmaps[ifilt, :, :])
                else:
                    plt.savefig(f"{dir}calib_{wavnb}_global_maps.png", dpi=150, bbox_inches='tight')
                    # plt.savefig(f"{dir}calib_{wavnb}_global_maps.eps", dpi=150, bbox_inches='tight')
                    # Clear figure to avoid overlapping between plotting subroutines
                    plt.close()
                    # Write global maps to np.array
                    np.save(f"{dir}calib_{wavnb}_global_maps", globalmaps[ifilt, :, :])
                    # Write global maps to txtfiles
                    # np.savetxt(f"{dir}calib_{wavnb}_global_maps.txt", globalmaps[ifilt, :, :])


def PlotMontageGlobalMaps(dataset):

    """ Plotting Montage Global maps using stored global maps array """

    print('Mapping Montage Global maps...')
    # If subdirectory does not exist, create it
    dir = f'../outputs/{dataset}/global_maps_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    # Define local inputs    
    lat           = np.arange(-89.75,90,step=0.5)               # Latitude range from pole-to-pole
    central_lon   = 0.                                          # Central longitude for polar projection
    central_lat   = 90.                                         # Absolute central latitude value for polar projection 
    lat_lim       = 10.                                         # Absolute latitude limit for polar projection 
    dmeridian     = 30                                          # Meridian lines step, interger to please xlocs parameter in gridlines
    dparallel     = 10                                          # Parallel lines step, interger to please ylocs in gridlines
    num_merid     = int(360/dmeridian + 1)                      # Number of meridian lines
    num_parra     = int((90-np.abs(lat_lim)) / dparallel + 1)   # Number of parallel lines per hemisphere
    lon_to_write  = 45                                          # Array to set on which longitude will be written latitude labels
    
    globalmap     = np.empty((Globals.nfilters, Globals.ny, Globals.nx))
    Nfilters = Globals.nfilters if dataset == '2018May' or '2018May_completed' else 11
    
    #  Subplot figure with both hemisphere
    for ifilt in range(Nfilters):
        if ifilt < 6 or ifilt > 7:
            if dataset == '2018May':
                # Retrive wavenumber corresponding to ifilt
                _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
                adj_location = 'average' if ifilt < 10 else 'southern'
                globalmap[ifilt, :, :] = np.load(f'../outputs/{dataset}/global_maps_figures/calib_{wavnb}_global_maps_{adj_location}_adj.npy')
            elif dataset == '2018May_completed':
                # Retrive wavenumber corresponding to ifilt
                _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
                globalmap[ifilt, :, :] = np.load(f'../outputs/{dataset}/global_maps_figures/calib_{wavnb}_global_maps.npy')
            elif dataset == '2022July' or dataset == '2022August':
                if ifilt == 4: 
                    _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt+1)
                elif ifilt > 5: 
                    _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt+2)
                else:
                    _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
                globalmap[ifilt, :, :] = np.load(f'../outputs/{dataset}/global_maps_figures/calib_{wavnb}_global_maps.npy')

    fig, ax = plt.subplots(6, 2, figsize=(10, 16), sharex=True, sharey=True)
    iax = 0
    for ifilt in [0,10,11,12,5,4,6,7,8,9,3,2,1]:
        if ifilt < 6 or ifilt > 7:
            _, wavlg, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
                   
            irow = [0,1,1,2,2,3,3,4,4,5,5]
            icol = [0,0,1,0,1,0,1,0,1,0,1]
            ititle = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)']
            
            # Remove the frame of the empty subplot
            ax[0][1].set_frame_on(False)
            ax[0][1].tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        

            # Setting brightness temperature extremes to plot global map
            max = np.nanmax(globalmap[ifilt, :, :]) 
            min = np.nanmin(globalmap[ifilt, :, :]) 
            # Plotting global map
           
            im = ax[irow[iax]][icol[iax]].imshow(globalmap[ifilt, :, :], origin='lower', vmin=min, vmax=max, cmap='inferno')
            ax[irow[iax]][icol[iax]].set_xticks(np.arange(0, Globals.nx+1,  step = 120), list(np.arange(360,-1,-60)))
            ax[irow[iax]][icol[iax]].set_yticks(np.arange(0, Globals.ny+1, step = 60), list(np.arange(-90,91,30)))
            ax[irow[iax]][icol[iax]].set_title(ititle[iax]+f"              {wavlg}"+r" $\mu$m", fontfamily='sans-serif', loc='left', fontsize=15)
            ax[irow[iax]][icol[iax]].tick_params(labelsize=14)
            cbar = fig.colorbar(im, ax=ax[irow[iax]][icol[iax]], extend='both', fraction=0.04, pad=0.05, format="%.0f")#, orientation='horizontal')
            cbar.ax.tick_params(labelsize=12)
            cbar.ax.locator_params(nbins=6)
            cbar.ax.set_title(r" T$_{B}$ [K]", size=15, pad=10)
            iax +=1
    plt.axes([0.1, 0.1, 0.8, 0.8], frameon=False) 
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel("Planetocentric Latitude", size=20)
    plt.xlabel("System III West Longitude", size=20)
    # Save south pole map figure of the current filter 
    if dataset == '2018May':
        plt.savefig(f"{dir}calib_all_global_maps.png", dpi=150, bbox_inches='tight')
        # plt.savefig(f"{dir}calib_all_global_maps.eps", dpi=150, bbox_inches='tight')
    else:
        plt.savefig(f"{dir}calib_all_global_maps.png", dpi=150, bbox_inches='tight')
        # plt.savefig(f"{dir}calib_all_global_maps.eps", dpi=150, bbox_inches='tight')
        # Clear figure to avoid overlapping between plotting subroutines
        plt.close()



    mumin = [0.15, 0.15, 0.02, 0.25, 0.3, 0.2, 0.5, 0.5, 0.15, 0.3, 0.6, 0.5, 0.05]
    plt.figure(figsize=(20,22))
    G = gridspec.GridSpec(6, 6, wspace=0.7, hspace=0.7)
    iax = 0
    for ifilt in [0,10,11,12,5,4,6,7,8,9,3,2,1]:
        if ifilt < 6 or ifilt > 7:
            adj_location= 'average' if ifilt < 10 else 'southern'
            # Get filter index for spectral profiles
            _, wavl, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
            # Get polynome coefficient calculating for 2018May dataset 
            
            coeff = np.load(f'../outputs/2018May/global_maps_figures/calib_{wavnb}_polynomial_coefficients_{adj_location}.npy')
            bandmu = np.load(f"../outputs/2018May/global_maps_figures/calib_{wavnb}_bandmu_{adj_location}.npy")
            bandc = np.load(f"../outputs/2018May/global_maps_figures/calib_{wavnb}_bandc_{adj_location}.npy")
            cdata = np.load(f"../outputs/2018May/global_maps_figures/calib_{wavnb}_cdata_{adj_location}.npy")
                
            # Calculate polynomial adjustement for each hemisphere (using mask selections)
            p = np.poly1d(coeff)
            # Define a linear space to show the polynomial adjustment variation over all emission angle range
            t = np.linspace(mumin[ifilt], 1, 100)

            irow = [0,1,1,2,2,3,3,4,4,5,5]
            icol = [0,0,3,0,3,0,3,0,3,0,3]
            ititle = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)']

            # Plot figure showing limb correction using polynomial adjustment method
            ax = plt.subplot(G[irow[iax],icol[iax]])
            ax.plot(bandmu, bandc, lw=0, marker='.', markersize=0.5, color = 'black')
            ax.plot(t, p(t), '-',color='red')
            ax.set_title(ititle[iax]+f"    {(wavl)}"+" $\mu$m \n"+'Observed \n T$_B$ [K]', fontfamily='sans-serif', loc='left', fontsize=18)
            ax.tick_params(labelsize=18)
            
            ax = plt.subplot(G[irow[iax],icol[iax]+1])
            ax.plot(t, (p(1))/p(t), '-',color='red')
            ax.set_title('Polynomial \n'+'Fucntion', fontfamily='sans-serif', loc='left', fontsize=18)
            ax.tick_params(labelsize=18)

            ax = plt.subplot(G[irow[iax],icol[iax]+2])
            ax.plot(bandmu, cdata, lw=0, marker='.', markersize=0.5, color = 'black')
            ax.set_title('Corrected \nT$_B$ [K]', fontfamily='sans-serif', loc='left', fontsize=18)
            ax.tick_params(labelsize=18)
            iax+=1
    plt.axes([0.1, 0.1, 0.8, 0.8], frameon=False) 
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Emission angle cosine", size=20)
    # Save figure showing limb correction using polynomial adjustment method 
    plt.savefig(f"{dir}calib_all_wavl_polynomial_adjustment.png", dpi=150, bbox_inches='tight')
    # plt.savefig(f"{directory}calib_all_wavl_polynomial_adjustment.eps", dpi=150, bbox_inches='tight')   




def PlotZoomMaps(dataset, central_lon, lat_target, lon_target, lat_window, lon_window):
    """ Mapping zoom maps for each VISIR filter """

    print('Mapping zoom maps...')
    # If subdirectory does not exist, create it
    dir = f'../outputs/{dataset}/zoom_maps_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    # Initialize some local variables and arrays
    lat         = np.arange(-89.75,90,step=0.5) # Latitude range from pole-to-pole
    lon         = np.arange(360,0, -0.5)       # Longitude range in System III West
    globalmap   = np.empty((Globals.nfilters, Globals.ny, Globals.nx))
    Nfilters    = Globals.nfilters if dataset == '2018May' else 11
    #  Subplot figure with both hemisphere
    for ifilt in range(Nfilters):
        if dataset == '2018May':
            # Retrive wavenumber corresponding to ifilt
            _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
            adj_location = 'average' if ifilt < 10 else 'southern'
            globalmap[ifilt, :, :] = np.load(f'../outputs/{dataset}/global_maps_figures/calib_{wavnb}_global_maps_{adj_location}_adj.npy')
        elif dataset == '2022July' or dataset == '2022August':
            if ifilt == 4: 
                _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt+1)
            elif ifilt > 5: 
                _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt+2)
            else:
                _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
            globalmap[ifilt, :, :] = np.load(f'../outputs/{dataset}/global_maps_figures/calib_{wavnb}_global_maps.npy')

        # Set extreme values for mapping
        if central_lon > lon_target:
            lonmax = np.asarray(np.where((lon == lon_target+lon_window+central_lon)))
            lonmin = np.asarray(np.where((lon == lon_target-lon_window+central_lon)))
        else:
            lonmax = np.asarray(np.where((lon == lon_target+lon_window)))
            lonmin = np.asarray(np.where((lon == lon_target-lon_window)))
        latmax = np.asarray(np.where((lat == lat_target+lat_window+0.25)))
        latmin = np.asarray(np.where((lat == lat_target-lat_window+0.25)))
        max = np.nanmax(globalmap[ifilt, int(latmin):int(latmax), int(lonmax):int(lonmin)]) 
        min = np.nanmin(globalmap[ifilt, int(latmin):int(latmax), int(lonmax):int(lonmin)])
        # Mapping zoom area
        plt.figure(figsize=(8, 3), dpi=150)
        projection = ccrs.PlateCarree(central_longitude=central_lon)
        ax = plt.axes(projection = projection)
        im = ax.imshow(globalmap[ifilt, :, :], \
                        transform=ccrs.PlateCarree(central_longitude=central_lon), \
                        origin='lower', extent=[0, 360, -90, 90], vmin=min, vmax=max, \
                        regrid_shape=1000, cmap='inferno')
        ax.set_extent([lon_target-lon_window, lon_target+lon_window, lat_target-lat_window, lat_target+lat_window], \
                        crs = ccrs.PlateCarree())
        if central_lon > lon_target:
            plt.xticks(np.arange(lon_target-lon_window-central_lon, lon_target+lon_window+1-central_lon,  step = lon_window/2),
                        list(np.arange(central_lon-lon_target+lon_window,central_lon-lon_target-lon_window-1,-lon_window/2)))
            # plt.xlim(180-lon_target-lon_window, 180-lon_target+lon_window)
        else:
            plt.xticks(np.arange(360-lon_target-lon_window, 360-lon_target+lon_window+1,  step = lon_window/2),
                        list(np.arange(lon_target+lon_window,lon_target-lon_window-1,-lon_window/2)))
            plt.xlim(360-lon_target-lon_window, 360-lon_target+lon_window)
        plt.yticks(np.arange(lat_target-lat_window, lat_target+lat_window+1, step = 5))
        
        plt.xlabel('System III West Longitude', size=15)
        plt.ylabel('Planetocentric Latitude', size=15)
        plt.tick_params(labelsize=12)
        cbar = plt.colorbar(im, extend='both', fraction=0.046, pad=0.15, orientation='horizontal', format="%.0f")
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.locator_params(nbins=6)
        cbar.set_label("Brightness Temperature [K]")

        # Save global map figure of the current filter 
        if dataset == '2022July' or dataset == '2022August':
            if ifilt == 4: 
                _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt+1)
            elif ifilt > 5: 
                _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt+2)
            else:
                _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
        else:
            _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
        if dataset == '2018May':
            plt.savefig(f"{dir}calib_{wavnb}_zoom_lon{lon_target}-lat{lat_target}_maps_{adj_location}_adj.png", dpi=150, bbox_inches='tight')
            plt.savefig(f"{dir}calib_{wavnb}_zoom_lon{lon_target}-lat{lat_target}_maps_{adj_location}_adj.eps", dpi=150, bbox_inches='tight')
        else:
            plt.savefig(f"{dir}calib_{wavnb}_zoom_lon{lon_target}-lat{lat_target}_maps.png", dpi=150, bbox_inches='tight')
            plt.savefig(f"{dir}calib_{wavnb}_zoom_lon{lon_target}-lat{lat_target}_maps.eps", dpi=150, bbox_inches='tight')
        # Clear figure to avoid overlapping between plotting subroutines
        plt.close()
            
def PlotMapsPerNight(dataset, files, spectrals):
    """ In the case of a multiple consecutive night dataset, 
        we could try to track the planetary-scale waves, for
        that we need to explore the global dynamics for each 
        night, therefore we need a global map per night."""

    print('Correcting and plotting global maps per night...')
    # If subdirectory does not exist, create it
    dir = f'../outputs/{dataset}/global_maps_per_night_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    # Define local inputs
    Nfiles = len(files)
    
    # Create np.arrays for all pixels in all cmaps and mumaps
    lat         = np.arange(-89.75,90,step=0.5) # Latitude range from pole-to-pole
    lon         = np.arange(360,0, -0.5)       # Longitude range in System III West
    tobs_ifile = np.empty((Nfiles, 2)) # First column: night+hour and second column: ifile
    TBmaps     = np.empty((Nfiles, Globals.ny, Globals.nx))
    globalmaps = np.empty((Globals.nfilters, Globals.ny, Globals.nx))
    zonalmean  = np.empty((Globals.nfilters, Globals.ny))
    zonalpert  = np.empty((Globals.nfilters, Globals.ny, Globals.nx))
    
    # # Load the cylindrical and mu maps
    cmaps, mumaps, wavenumber = ApplyPolynom(dir, files, spectrals)
    # cmaps, mumaps, wavenumber = MuNormalization(files)
    mumin = np.empty(13)
    mumin.fill(0.09)

    list_time = []
    list_ifile = []
    for ifile, fpath in enumerate(files):
        print(fpath)
        # retrieve date, hour of the cylmaps corresponding to 
        # the current file in an array to be sorted
        hour = fpath.split('_201')
        hour  = hour[-1].split('.')
        hour  = hour[0]
        hour = hour.replace('T', '')
        hour = hour.replace('-', '')
        hour = hour.replace('_', '')
        # Append hour and ifile into lists
        list_time.append(hour)
        list_ifile.append(ifile)
    # Store in the tobs_ifile array
    tobs_ifile[:, 0] = [float(x) for x in list_time]
    tobs_ifile[:, 1] = list_ifile
    # Sort the arrays in function of time 
    tobs_ifile = sorted(tobs_ifile, key=operator.itemgetter(1))
    tobs_ifile = np.asarray(tobs_ifile, dtype='float')

    # Set night limits 
    night_limits = [0, 80524120000, 80525120000, 80526120000, 80527120000]
                    
    for inight in range(len(night_limits)-1):
        for ifilt in range(Globals.nfilters):
            if ifilt != 6 and ifilt != 7:
                # Empty local TBmaps array to avoid overlapping between filter
                TBmaps[:, :, :] = np.nan
                # Get filter index for spectral profiles
                waves = spectrals[:, ifilt, 5]
                if waves[(waves > 0)] != []:
                    wave  = waves[(waves > 0)][0]
                    for ifile, iwave in enumerate(wavenumber):
                        # Store only cmaps for the current filter ifilt and current night inight
                        if tobs_ifile[ifile, 0] > night_limits[inight] and tobs_ifile[ifile, 0] < night_limits[inight+1]:
                            if iwave == wave:
                                TBmaps[ifile, :, :] = cmaps[ifile, :, :]                
                                res = ma.masked_where(mumaps[ifile, :, :] < mumin[ifilt], TBmaps[ifile, :, :])
                                res = ma.masked_where(((res > 201)), res)
                                res = ma.masked_where(((res < 100)), res)
                                TBmaps[ifile,:,:] = res.filled(np.nan)
    ####                
    # Mapping of the bightness temperature for each observation night:
                    # Combinig single cylmaps to store in globalmaps array
                    for y in range(Globals.ny):
                        for x in range(Globals.nx):
                            globalmaps[ifilt, y, x] = np.nanmax(TBmaps[:, y, x])
                    # Setting brightness temperature extremes to plot global map
                    max = np.nanmax(globalmaps[ifilt, :, :]) 
                    min = np.nanmin(globalmaps[ifilt, :, :]) 
                    # Plotting global map
                    im = plt.imshow(globalmaps[ifilt, :, :], origin='lower', vmin=min, vmax=max, cmap='inferno')
                    plt.xticks(np.arange(0, Globals.nx+1,  step = 60), list(np.arange(360,-1,-30)))
                    plt.yticks(np.arange(0, Globals.ny+1, step = 60), list(np.arange(-90,91,30)))
                    plt.xlabel('System III West Longitude')
                    plt.ylabel('Planetocentric Latitude')
                    #plt.tick_params(labelsize=15)
                    cbar = plt.colorbar(im, extend='both', fraction=0.046, pad=0.014)
                    #cbar.ax.tick_params(labelsize=15)
                    cbar.set_label("Brightness Temperature [K]")

                    # Save global map figure of the current filter 
                    _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
                    if dataset == '2018May_completed':
                        plt.savefig(f"{dir}calib_{wavnb}_global_maps_night_{inight}.png", dpi=150, bbox_inches='tight')
                        # plt.savefig(f"{dir}calib_{wavnb}_global_maps_night_{inight}.eps", dpi=150, bbox_inches='tight')
                        # Clear figure to avoid overlapping between plotting subroutines
                        plt.close()
                        # Write global maps to np.array
                        np.save(f"{dir}calib_{wavnb}_global_maps_night_{inight}", globalmaps[ifilt, :, :])
                        # Write global maps to txtfiles
                        # np.savetxt(f"{dir}calib_{wavnb}_global_maps_night_{inight}.txt", globalmaps[ifilt, :, :])
                        # Write global maps to NetCDF files
                        #GlobalMapsNetCDF(dir, wavnb, globalmaps=globalmaps[ifilt, :, :])
    ####                
    # Mapping of the zonal perturbation of bightness temperature for each observation night:
                    for y in range(Globals.ny):
                        # Zonal mean of the daily global maps
                        zonalmean[ifilt, y] = np.nanmean(globalmaps[ifilt, y, :])
                        # Corresponding zonal perturbation in brightness temperature
                        for x in range (Globals.nx):
                            zonalpert[ifilt, y, x] = globalmaps[ifilt, y, x] - zonalmean[ifilt, y]
                    # Setting brightness temperature extremes to plot global map
                    max = np.nanmax(zonalpert[ifilt, :, :]) 
                    min = np.nanmin(zonalpert[ifilt, :, :])
                    norm = colors.TwoSlopeNorm(vmin=-8, vmax=6, vcenter=0) 
                    # Plotting global map
                    im = plt.imshow(zonalpert[ifilt, :, :], origin='lower', norm=norm, cmap='seismic')
                    plt.xticks(np.arange(0, Globals.nx+1,  step = 60), list(np.arange(360,-1,-30)))
                    plt.yticks(np.arange(0, Globals.ny+1, step = 60), list(np.arange(-90,91,30)))
                    plt.xlabel('System III West Longitude')
                    plt.ylabel('Planetocentric Latitude')
                    #plt.tick_params(labelsize=15)
                    cbar = plt.colorbar(im, extend='both', fraction=0.046, pad=0.014)
                    #cbar.ax.tick_params(labelsize=15)
                    cbar.set_label("Brightness Temperature anomalies [K]")

                    # Save global map figure of the current filter 
                    _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
                    if dataset == '2018May_completed':
                        plt.savefig(f"{dir}calib_{wavnb}_zonalpert_maps_night_{inight}.png", dpi=150, bbox_inches='tight')
                        # plt.savefig(f"{dir}calib_{wavnb}_zonalpert_maps_night_{inight}.eps", dpi=150, bbox_inches='tight')
                        # Clear figure to avoid overlapping between plotting subroutines
                        plt.close()
                        # Write global maps to np.array
                        np.save(f"{dir}calib_{wavnb}_zonalpert_maps_night_{inight}", zonalpert[ifilt, :, :])
                        # Write global maps to txtfiles
                        # np.savetxt(f"{dir}calib_{wavnb}_zonalpert_maps_night_{inight}.txt", zonalpert[ifilt, :, :])

                    # Set extreme values for mapping
                    central_lon=180
                    lat_target=18
                    lon_target=310
                    lat_window=10
                    lon_window=50

                    if central_lon > lon_target:
                        lonmax = np.asarray(np.where((lon == lon_target+lon_window+central_lon)))
                        lonmin = np.asarray(np.where((lon == lon_target-lon_window+central_lon)))
                    else:
                        lonmax = np.asarray(np.where((lon == lon_target+lon_window)))
                        lonmin = np.asarray(np.where((lon == lon_target-lon_window)))
                    latmax = np.asarray(np.where((lat == lat_target+lat_window+0.25)))
                    latmin = np.asarray(np.where((lat == lat_target-lat_window+0.25)))
                    max = np.nanmax(zonalpert[ifilt, int(latmin):int(latmax), int(lonmax):int(lonmin)]) 
                    min = np.nanmin(zonalpert[ifilt, int(latmin):int(latmax), int(lonmax):int(lonmin)])
                    norm = colors.TwoSlopeNorm(vmin=-8, vmax=6, vcenter=0)
                    # Mapping zoom area
                    plt.figure(figsize=(8, 3), dpi=150)
                    projection = ccrs.PlateCarree(central_longitude=central_lon)
                    ax = plt.axes(projection = projection)
                    im = ax.imshow(zonalpert[ifilt, :, :], \
                                    transform=ccrs.PlateCarree(central_longitude=central_lon), \
                                    origin='lower', extent=[0, 360, -90, 90], norm=norm, \
                                    regrid_shape=1000, cmap='seismic')
                    ax.set_extent([lon_target-lon_window, lon_target+lon_window, lat_target-lat_window, lat_target+lat_window], \
                                    crs = ccrs.PlateCarree())
                    if central_lon > lon_target:
                        plt.xticks(np.arange(lon_target-lon_window-central_lon, lon_target+lon_window+1-central_lon,  step = lon_window/2),
                                    list(np.arange(central_lon-lon_target+lon_window,central_lon-lon_target-lon_window-1,-lon_window/2)))
                        # plt.xlim(180-lon_target-lon_window, 180-lon_target+lon_window)
                    else:
                        plt.xticks(np.arange(360-lon_target-lon_window, 360-lon_target+lon_window+1,  step = lon_window/2),
                                    list(np.arange(lon_target+lon_window,lon_target-lon_window-1,-lon_window/2)))
                        plt.xlim(360-lon_target-lon_window, 360-lon_target+lon_window)
                    plt.yticks(np.arange(lat_target-lat_window, lat_target+lat_window+1, step = 5))
                    
                    plt.xlabel('System III West Longitude', size=15)
                    plt.ylabel('Planetocentric Latitude', size=15)
                    plt.tick_params(labelsize=12)
                    cbar = plt.colorbar(im, extend='both', fraction=0.046, pad=0.014)
                    cbar.ax.tick_params(labelsize=12)
                    cbar.ax.locator_params(nbins=6)
                    #cbar.ax.tick_params(labelsize=15)
                    cbar.set_label("Brightness Temperature anomalies [K]")
                    if inight ==0:
                        plt.title('2018 May 24')
                    elif inight ==1:
                        plt.title('2018 May 24-25')
                    elif inight ==2:
                        plt.title('2018 May 25-26')
                    elif inight ==3:
                        plt.title('2018 May 26-27')

                    # Save global map figure of the current filter 
                    _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
                    if dataset == '2018May_completed':
                        plt.savefig(f"{dir}calib_{wavnb}_zonalpert_maps_lon{lon_target}_lat{lat_target}_night_{inight}.png", dpi=150, bbox_inches='tight')
                        # plt.savefig(f"{dir}calib_{wavnb}_zonalpert_maps_night_{inight}.eps", dpi=150, bbox_inches='tight')
                        # Clear figure to avoid overlapping between plotting subroutines
                        plt.close()
                        # Write global maps to np.array
                        np.save(f"{dir}calib_{wavnb}_zonalpert_maps_lon{lon_target}_lat{lat_target}_night_{inight}", zonalpert[ifilt, :, :])
                        
def PlotSubplotMapsPerNight(dataset):
    """ In the case of a multiple consecutive night dataset, 
        we could try to track the planetary-scale waves, for
        that we need to explore the global dynamics for each 
        night, therefore we need a global map per night."""

    print('Plotting subplot of zoom maps per night...')

    # If subdirectory does not exist, create it
    dir = f'../outputs/{dataset}/global_maps_per_night_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    # Define local inputs    
    lat = np.arange(-89.75,90,step=0.5) # Latitude range from pole-to-pole
    lon = np.arange(360,0, -0.5)        # Longitude range in System III West
    # Set extreme values for mapping
    central_lon=180
    lat_target=18
    lon_target=310
    lat_window=10
    lon_window=50
    
    Nnight = 4
    Nfilters = Globals.nfilters if dataset == '2018May' or '2018May_completed' else 11
    zonalpert     = np.empty((Nnight, Globals.nfilters, Globals.ny, Globals.nx))
    title = [r'2018 May 24$^{th}$', r'2018 May 24$^{th}$-25$^{th}$', r'2018 May 25$^{th}$-26$^{th}$', r'2018 May 26$^{th}$-27$^{th}$']
    #  Subplot figure with both hemisphere
    for ifilt in range(Nfilters):
        if ifilt < 6 or ifilt > 7:
            if dataset == '2018May_completed':
                # Retrive wavenumber corresponding to ifilt
                for inight in range(Nnight):
                    _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
                    zonalpert[inight, ifilt, :, :] = np.load(f'../outputs/{dataset}/global_maps_per_night_figures/calib_{wavnb}_zonalpert_maps_lon310_lat18_night_{inight}.npy')
                    
    
    for ifilt in range(Globals.nfilters):
        if ifilt != 6 and ifilt != 7:
            # A figure per filter and per night
            fig = plt.figure(figsize=(8, 10))
            projection = ccrs.PlateCarree(central_longitude=central_lon)
            for inight in range(Nnight):
                ax = plt.subplot2grid((4,1), (inight,0),  projection = projection)
                # Set the limit of the zoom maps
                if central_lon > lon_target:
                    lonmax = np.asarray(np.where((lon == lon_target+lon_window+central_lon)))
                    lonmin = np.asarray(np.where((lon == lon_target-lon_window+central_lon)))
                else:
                    lonmax = np.asarray(np.where((lon == lon_target+lon_window)))
                    lonmin = np.asarray(np.where((lon == lon_target-lon_window)))
                latmax = np.asarray(np.where((lat == lat_target+lat_window+0.25)))
                latmin = np.asarray(np.where((lat == lat_target-lat_window+0.25)))
                # Set extreme values of the zoom maps
                max = np.nanmax(zonalpert[inight, ifilt, int(latmin):int(latmax), int(lonmax):int(lonmin)]) 
                min = np.nanmin(zonalpert[inight, ifilt, int(latmin):int(latmax), int(lonmax):int(lonmin)])
                norm = colors.TwoSlopeNorm(vmin=-8, vmax=6, vcenter=0)
                # Mapping zoom area
                im = ax.imshow(zonalpert[inight, ifilt, :, :], \
                                transform=ccrs.PlateCarree(central_longitude=central_lon), \
                                origin='lower', extent=[0, 360, -90, 90], norm=norm, \
                                regrid_shape=1000, cmap='seismic')
                ax.set_extent([lon_target-lon_window, lon_target+lon_window, lat_target-lat_window, lat_target+lat_window], \
                                crs = ccrs.PlateCarree())
                ax.set_title(title[inight], size=20)
                ax.tick_params(labelsize=15)
                if central_lon > lon_target:
                    if inight < 3:
                        plt.xticks(np.arange(lon_target-lon_window-central_lon, lon_target+lon_window+1-central_lon,  step = lon_window/2),\
                                    [])
                        # plt.xlim(360-lon_target-lon_window, 360-lon_target+lon_window)
                    else:
                        plt.xticks(np.arange(lon_target-lon_window-central_lon, lon_target+lon_window+1-central_lon,  step = lon_window/2),\
                                    list(np.arange(central_lon-lon_target+lon_window,central_lon-lon_target-lon_window-1,-lon_window/2)))
                        # ax.xlim(180-lon_target-lon_window, 180-lon_target+lon_window)
                else:
                    if inight < 3:
                        plt.xticks(np.arange(360-lon_target-lon_window, 360-lon_target+lon_window+1,  step = lon_window/2),
                                    [])
                        plt.xlim(360-lon_target-lon_window, 360-lon_target+lon_window)
                    else: 
                        plt.xticks(np.arange(360-lon_target-lon_window, 360-lon_target+lon_window+1,  step = lon_window/2),\
                                    list(np.arange(lon_target+lon_window,lon_target-lon_window-1,-lon_window/2)))
                        plt.xlim(360-lon_target-lon_window, 360-lon_target+lon_window)
                plt.yticks(np.arange(lat_target-lat_window, lat_target+lat_window+1, step = 5))
            plt.axes([0.1, 0.1, 0.8, 0.8], frameon=False) 
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel('System III West Longitude', size=20)
            plt.ylabel('Planetocentric Latitude', size=20)
            cax = plt.axes([0.95, 0.1, 0.05, 0.8])
            cbar = plt.colorbar(im, cax=cax, format="%.0f", extend='both', fraction=0.046, pad=0.15)
            cbar.ax.tick_params(labelsize=20)
            cbar.locator = ticker.MaxNLocator(nbins=10)
            cbar.update_ticks()
            cbar.ax.set_title(r" T$_{B}^{'}$ [K]", size=20, pad=30)
            # Save global map figure of the current filter 
            _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
            plt.savefig(f"{dir}calib_{wavnb}_zonalpert_maps_lon{lon_target}_lat{lat_target}_all_nights.png", dpi=150, bbox_inches='tight')
            # plt.savefig(f"{dir}calib_{wavnb}_zonalpert_maps_all_nights.eps", dpi=150, bbox_inches='tight')
            # Clear figure to avoid overlapping between plotting subroutines
            plt.close()
            # Write global maps to np.array
            # np.save(f"{dir}calib_{wavnb}_zonalpert_maps_lon{lon_target}_lat{lat_target}_all_nights", zonalpert[ifilt, :, :])

def PlotSubplotMapsPerNightForJGRPaper(dataset):
    """ In the case of a multiple consecutive night dataset, 
        we could try to track the planetary-scale waves, for
        that we need to explore the global dynamics for each 
        night, therefore we need a global map per night."""

    print('Plotting subplot of zoom maps per night in one figure...')

    # If subdirectory does not exist, create it
    dir = f'../outputs/{dataset}/global_maps_per_night_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    # Define local inputs    
    lat = np.arange(-89.75,90,step=0.5) # Latitude range from pole-to-pole
    lon = np.arange(360,0, -0.5)        # Longitude range in System III West
    # Set extreme values for mapping
    central_lon=180
    lat_target=18
    lon_target=310
    lat_window=10
    lon_window=50
    
    Nnight = 4
    Nfilters = Globals.nfilters if dataset == '2018May' or '2018May_completed' else 11
    zonalpert     = np.empty((Nnight, Globals.nfilters, Globals.ny, Globals.nx))
    title = [r'2018 May 24$^{th}$', r'2018 May 24$^{th}$-25$^{th}$', r'2018 May 25$^{th}$-26$^{th}$', r'2018 May 26$^{th}$-27$^{th}$']
    #  Subplot figure with both hemisphere
    for ifilt in range(Nfilters):
        if ifilt < 6 or ifilt > 7:
            if dataset == '2018May_completed':
                # Retrive wavenumber corresponding to ifilt
                for inight in range(Nnight):
                    _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
                    zonalpert[inight, ifilt, :, :] = np.load(f'../outputs/{dataset}/global_maps_per_night_figures/calib_{wavnb}_zonalpert_maps_lon310_lat18_night_{inight}.npy')
                    
    
   
    fig = plt.figure(figsize=(16,7))
    projection = ccrs.PlateCarree(central_longitude=central_lon)
    for inight in range(Nnight):
        
        # Set the limit of the zoom maps
        if central_lon > lon_target:
            lonmax = np.asarray(np.where((lon == lon_target+lon_window+central_lon)))
            lonmin = np.asarray(np.where((lon == lon_target-lon_window+central_lon)))
        else:
            lonmax = np.asarray(np.where((lon == lon_target+lon_window)))
            lonmin = np.asarray(np.where((lon == lon_target-lon_window)))
        latmax = np.asarray(np.where((lat == lat_target+lat_window+0.25)))
        latmin = np.asarray(np.where((lat == lat_target-lat_window+0.25)))

        #8.59mu
        ax = plt.subplot2grid((4,3), (inight,0),  projection = projection)
        # Set extreme values of the zoom maps
        max = np.nanmax(zonalpert[inight, 1, int(latmin):int(latmax), int(lonmax):int(lonmin)]) 
        min = np.nanmin(zonalpert[inight, 1, int(latmin):int(latmax), int(lonmax):int(lonmin)])
        norm = colors.TwoSlopeNorm(vmin=-8, vmax=6, vcenter=0)
        # Mapping zoom area
        im = ax.imshow(zonalpert[inight, 1, :, :], \
                        transform=ccrs.PlateCarree(central_longitude=central_lon), \
                        origin='lower', extent=[0, 360, -90, 90], norm=norm, \
                        regrid_shape=1000, cmap='seismic')
        ax.set_extent([lon_target-lon_window, lon_target+lon_window, lat_target-lat_window, lat_target+lat_window], \
                        crs = ccrs.PlateCarree())
        ax.set_title(title[inight], size=20)
        ax.tick_params(labelsize=20)
        if central_lon > lon_target:
            if inight < 3:
                plt.xticks(np.arange(lon_target-lon_window-central_lon, lon_target+lon_window+1-central_lon,  step = lon_window/2),\
                            [])
                # plt.xlim(360-lon_target-lon_window, 360-lon_target+lon_window)
            else:
                plt.xticks(np.arange(lon_target-lon_window-central_lon, lon_target+lon_window+1-central_lon,  step = lon_window/2),\
                            list([int(x) for x in np.arange(central_lon-lon_target+lon_window,central_lon-lon_target-lon_window-1,-lon_window/2)]))
                # ax.xlim(180-lon_target-lon_window, 180-lon_target+lon_window)
        else:
            if inight < 3:
                plt.xticks(np.arange(360-lon_target-lon_window, 360-lon_target+lon_window+1,  step = lon_window/2),
                            [])
                plt.xlim(360-lon_target-lon_window, 360-lon_target+lon_window)
            else: 
                plt.xticks(np.arange(360-lon_target-lon_window, 360-lon_target+lon_window+1,  step = lon_window/2),\
                            list([int(x) for x in np.arange(lon_target+lon_window,lon_target-lon_window-1,-lon_window/2)]))
                plt.xlim(360-lon_target-lon_window, 360-lon_target+lon_window)
        plt.yticks(np.arange(lat_target-lat_window, lat_target+lat_window+1, step = 10))
        _, wavl, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=1)
        plt.figtext(0.1,0.91,f"(a)  {wavl} "+r"$\mu$m", size=20)

        #10.77mu
        ax = plt.subplot2grid((4,3), (inight,1),  projection = projection)
        # Set extreme values of the zoom maps
        max = np.nanmax(zonalpert[inight, 5, int(latmin):int(latmax), int(lonmax):int(lonmin)]) 
        min = np.nanmin(zonalpert[inight, 5, int(latmin):int(latmax), int(lonmax):int(lonmin)])
        norm = colors.TwoSlopeNorm(vmin=-8, vmax=6, vcenter=0)
        # Mapping zoom area
        im = ax.imshow(zonalpert[inight, 5, :, :], \
                        transform=ccrs.PlateCarree(central_longitude=central_lon), \
                        origin='lower', extent=[0, 360, -90, 90], norm=norm, \
                        regrid_shape=1000, cmap='seismic')
        ax.set_extent([lon_target-lon_window, lon_target+lon_window, lat_target-lat_window, lat_target+lat_window], \
                        crs = ccrs.PlateCarree())
        ax.set_title(title[inight], size=20)
        ax.tick_params(labelsize=20)
        if central_lon > lon_target:
            if inight < 3:
                plt.xticks(np.arange(lon_target-lon_window-central_lon, lon_target+lon_window+1-central_lon,  step = lon_window/2),\
                            [])
                # plt.xlim(360-lon_target-lon_window, 360-lon_target+lon_window)
            else:
                plt.xticks(np.arange(lon_target-lon_window-central_lon, lon_target+lon_window+1-central_lon,  step = lon_window/2),\
                            list([int(x) for x in np.arange(central_lon-lon_target+lon_window,central_lon-lon_target-lon_window-1,-lon_window/2)]))
                # ax.xlim(180-lon_target-lon_window, 180-lon_target+lon_window)
        else:
            if inight < 3:
                plt.xticks(np.arange(360-lon_target-lon_window, 360-lon_target+lon_window+1,  step = lon_window/2),
                            [])
                plt.xlim(360-lon_target-lon_window, 360-lon_target+lon_window)
            else: 
                plt.xticks(np.arange(360-lon_target-lon_window, 360-lon_target+lon_window+1,  step = lon_window/2),\
                            list([int(x) for x in np.arange(lon_target+lon_window,lon_target-lon_window-1,-lon_window/2)]))
                plt.xlim(360-lon_target-lon_window, 360-lon_target+lon_window)
        #plt.yticks(np.arange(lat_target-lat_window, lat_target+lat_window+1, step = 5))
        _, wavl, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=5)
        plt.figtext(0.37,0.91,f"(b)  {wavl} "+r"$\mu$m", size=20)

        #7.9mu
        ax2 = plt.subplot2grid((4,3), (inight,2),  projection = projection)
        # Set extreme values of the zoom maps
        max = np.nanmax(zonalpert[inight, 0, int(latmin):int(latmax), int(lonmax):int(lonmin)]) 
        min = np.nanmin(zonalpert[inight, 0, int(latmin):int(latmax), int(lonmax):int(lonmin)])
        norm = colors.TwoSlopeNorm(vmin=-8, vmax=6, vcenter=0)
        # Mapping zoom area
        im = ax2.imshow(zonalpert[inight, 0, :, :], \
                        transform=ccrs.PlateCarree(central_longitude=central_lon), \
                        origin='lower', extent=[0, 360, -90, 90], norm=norm, \
                        regrid_shape=1000, cmap='seismic')
        ax2.set_extent([lon_target-lon_window, lon_target+lon_window, lat_target-lat_window, lat_target+lat_window], \
                        crs = ccrs.PlateCarree())
        ax2.set_title(title[inight], size=20)
        ax2.tick_params(labelsize=20)


        if central_lon > lon_target:
            if inight < 3:
                plt.xticks(np.arange(lon_target-lon_window-central_lon, lon_target+lon_window+1-central_lon,  step = lon_window/2),\
                            [])
                # plt.xlim(360-lon_target-lon_window, 360-lon_target+lon_window)
            else:
                plt.xticks(np.arange(lon_target-lon_window-central_lon, lon_target+lon_window+1-central_lon,  step = lon_window/2),\
                            list([int(x) for x in np.arange(central_lon-lon_target+lon_window,central_lon-lon_target-lon_window-1,-lon_window/2)]))
                # ax.xlim(180-lon_target-lon_window, 180-lon_target+lon_window)
        else:
            if inight < 3:
                plt.xticks(np.arange(360-lon_target-lon_window, 360-lon_target+lon_window+1,  step = lon_window/2),
                            [])
                plt.xlim(360-lon_target-lon_window, 360-lon_target+lon_window)
            else: 
                plt.xticks(np.arange(360-lon_target-lon_window, 360-lon_target+lon_window+1,  step = lon_window/2),\
                            list([int(x) for x in np.arange(lon_target+lon_window,lon_target-lon_window-1,-lon_window/2)]))
                plt.xlim(360-lon_target-lon_window, 360-lon_target+lon_window)
        # plt.yticks(np.arange(lat_target-lat_window, lat_target+lat_window+1, step = 5))
        _, wavl, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=0)
        plt.figtext(0.65,0.91,f"(c)  {wavl} "+r"$\mu$m", size=20)
        
    plt.axes([0.115, 0.1, 0.8, 0.8], frameon=False) 
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel('System III West Longitude', size=20)
    plt.ylabel('Planetocentric Latitude', size=20)
    cax = plt.axes([0.92, 0.2, 0.02, 0.6])
    cbar = plt.colorbar(im, cax=cax, format="%.0f", extend='both', fraction=0.015, pad=0.05)
    cbar.ax.tick_params(labelsize=20)
    cbar.locator = ticker.MaxNLocator(nbins=10)
    cbar.update_ticks()
    cbar.ax.set_title(r" T$_{B}^{'}$ [K]", size=20, pad=30)
    
    plt.savefig(f"{dir}calib_zonalpert_maps_lon{lon_target}_lat{lat_target}_all_nights.png", dpi=150, bbox_inches='tight')
    # plt.savefig(f"{dir}calib_{wavnb}_zonalpert_maps_all_nights.eps", dpi=150, bbox_inches='tight')
    # Clear figure to avoid overlapping between plotting subroutines
    plt.close()
    # Write global maps to np.array
    # np.save(f"{dir}calib_{wavnb}_zonalpert_maps_lon{lon_target}_lat{lat_target}_all_nights", zonalpert[ifilt, :, :])


    # For the baroclinic warm anomalies at 55N-70W
    # Set extreme values for mapping
    central_lon=180
    lat_target=45
    lon_target=100
    lat_window=20
    lon_window=50
    
    Nnight = 4
    Nfilters = Globals.nfilters if dataset == '2018May' or '2018May_completed' else 11
    zonalpert     = np.empty((Nnight, Globals.nfilters, Globals.ny, Globals.nx))
    title = [r'2018 May 24$^{th}$', r'2018 May 24$^{th}$-25$^{th}$', r'2018 May 25$^{th}$-26$^{th}$', r'2018 May 26$^{th}$-27$^{th}$']
    #  Subplot figure with both hemisphere
    for ifilt in range(Nfilters):
        if ifilt < 6 or ifilt > 7:
            if dataset == '2018May_completed':
                # Retrive wavenumber corresponding to ifilt
                for inight in range(Nnight):
                    _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
                    zonalpert[inight, ifilt, :, :] = np.load(f'../outputs/{dataset}/global_maps_per_night_figures/calib_{wavnb}_zonalpert_maps_night_{inight}.npy')
                    
    
   
    fig = plt.figure(figsize=(6, 8))
    projection = ccrs.PlateCarree(central_longitude=central_lon)
    for inight in range(3):
        
        # Set the limit of the zoom maps
        if central_lon > lon_target:
            lonmax = np.asarray(np.where((lon == lon_target+lon_window+central_lon)))
            lonmin = np.asarray(np.where((lon == lon_target-lon_window+central_lon)))
        else:
            lonmax = np.asarray(np.where((lon == lon_target+lon_window)))
            lonmin = np.asarray(np.where((lon == lon_target-lon_window)))
        latmax = np.asarray(np.where((lat == lat_target+lat_window+0.25)))
        latmin = np.asarray(np.where((lat == lat_target-lat_window+0.25)))

        #8.59mu
        ax = plt.subplot2grid((3,1), (inight,0),  projection = projection)
        # Set extreme values of the zoom maps
        max = np.nanmax(zonalpert[inight, 0, int(latmin):int(latmax), int(lonmax):int(lonmin)]) 
        min = np.nanmin(zonalpert[inight, 0, int(latmin):int(latmax), int(lonmax):int(lonmin)])
        norm = colors.TwoSlopeNorm(vmin=-8, vmax=6, vcenter=0)
        # Mapping zoom area
        im = ax.imshow(zonalpert[inight, 0, :, :], \
                        transform=ccrs.PlateCarree(central_longitude=central_lon), \
                        origin='lower', extent=[0, 360, -90, 90], norm=norm, \
                        regrid_shape=1000, cmap='seismic')
        ax.set_extent([lon_target-lon_window, lon_target+lon_window, lat_target-lat_window, lat_target+lat_window], \
                        crs = ccrs.PlateCarree())
        ax.set_title(title[inight], size=20)
        ax.tick_params(labelsize=20)
        if central_lon > lon_target:
            if inight < 2:
                plt.xticks(np.arange(lon_target-lon_window-central_lon, lon_target+lon_window+1-central_lon,  step = lon_window/2),\
                            [])
                # plt.xlim(360-lon_target-lon_window, 360-lon_target+lon_window)
            else:
                plt.xticks(np.arange(lon_target-lon_window-central_lon, lon_target+lon_window+1-central_lon,  step = lon_window/2),\
                            list([int(x) for x in np.arange(central_lon-lon_target+lon_window,central_lon-lon_target-lon_window-1,-lon_window/2)]))
                # ax.xlim(180-lon_target-lon_window, 180-lon_target+lon_window)
        else:
            if inight < 2:
                plt.xticks(np.arange(360-lon_target-lon_window, 360-lon_target+lon_window+1,  step = lon_window/2),
                            [])
                plt.xlim(360-lon_target-lon_window, 360-lon_target+lon_window)
            else: 
                plt.xticks(np.arange(360-lon_target-lon_window, 360-lon_target+lon_window+1,  step = lon_window/2),\
                            list([int(x) for x in np.arange(lon_target+lon_window,lon_target-lon_window-1,-lon_window/2)]))
                plt.xlim(360-lon_target-lon_window, 360-lon_target+lon_window)
        plt.yticks(np.arange(lat_target-lat_window, lat_target+lat_window+1, step = 10))

    plt.axes([0.1, 0.1, 0.8, 0.8], frameon=False) 
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel('System III West Longitude', size=20)
    plt.ylabel('Planetocentric Latitude', size=20)
    cax = plt.axes([0.95, 0.2, 0.06, 0.6])
    cbar = plt.colorbar(im, cax=cax, format="%.0f", extend='both', fraction=0.015, pad=0.15)
    cbar.ax.tick_params(labelsize=20)
    cbar.locator = ticker.MaxNLocator(nbins=10)
    cbar.update_ticks()
    cbar.ax.set_title(r" T$_{B}^{'}$ [K]", size=20, pad=30)
    
    _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=0)
    plt.savefig(f"{dir}calib_{wavnb}_zonalpert_maps_lon{lon_target}_lat{lat_target}_all_nights.png", dpi=150, bbox_inches='tight')
    # plt.savefig(f"{dir}calib_{wavnb}_zonalpert_maps_all_nights.eps", dpi=150, bbox_inches='tight')
    # Clear figure to avoid overlapping between plotting subroutines
    plt.close()
    # Write global maps to np.array
    # np.save(f"{dir}calib_{wavnb}_zonalpert_maps_lon{lon_target}_lat{lat_target}_all_nights", zonalpert[ifilt, :, :])