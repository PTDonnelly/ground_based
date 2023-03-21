from ast import Global
import os
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib import colors
import cartopy.crs as ccrs
import operator
import datetime
import Globals
from Tools.CorrectMaps import GetCylandMuMaps, PolynomialAdjust, ApplyPolynom, BlackLineRemoving, MuNormalization
from Tools.SetWave import SetWaveReduced

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
    if dataset == '2018May':
        cmaps, mumaps, wavenumber, adj_location = PolynomialAdjust(dir, files, spectrals)
        mumin = [0.02, 0.02, 0.1, 0.08, 0.01, 0.05, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01]
        Nfilters = Globals.nfilters
    elif dataset == '2022July':
        cmaps, mumaps, wavenumber = BlackLineRemoving(dir, files, cblack=-60, mu_scaling=True)
        mumin = np.empty(13)
        mumin.fill(0.01)
        Nfilters = 10
    else:
        cmaps, mumaps, wavenumber = MuNormalization(files)
        # mumin = [0.02, 0.05, 0.1, 0.08, 0.05, 0.05, 0.0, 0.0, 0.1, 0.08, 0.15, 0.05, 0.05]
        mumin = np.empty(Globals.nfilters)
        mumin.fill(0.01)
        Nfilters = 13 if dataset == '2018May_completed' else Globals.nfilters

    print('Mapping global maps...')

    for ifilt in range(Nfilters):
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
            globalmaps[ifilt, :, :] = np.nanmax(TBmaps[:, :, :], axis=0)
            # for y in range(Globals.ny):
            #     for x in range(Globals.nx):
            #         globalmaps[ifilt, y, x] = np.nanmax(TBmaps[:, y, x])

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
            _, _, wavnb, _, _ = SetWaveReduced(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
            print(ifilt, wavnb)
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
    # cmaps, mumaps, wavenumber = ApplyPolynom(dir, files, spectrals)
    cmaps, mumaps, wavenumber = MuNormalization(files)
    mumin = np.empty(13)
    mumin.fill(0.1)

    list_time = []
    list_ifile = []
    for ifile, fpath in enumerate(files):
        print(fpath)
        # retrieve date, hour of the cylmaps corresponding to 
        # the current file in an array to be sorted
        hour = fpath.split('_20')
        hour  = hour[-1].split('.')
        hour  = hour[0]
        hour = hour.replace('T', '')
        hour = hour.replace('-', '')
        hour = hour.replace(':', '')
        # Append hour and ifile into lists
        list_time.append(hour)
        list_ifile.append(ifile)
    # Store in the tobs_ifile array
    tobs_ifile[:, 0] = list_time
    tobs_ifile[:, 1] = list_ifile
    # Sort the arrays in function of time 
    tobs_ifile = sorted(tobs_ifile, key=operator.itemgetter(1))
    tobs_ifile = np.asarray(tobs_ifile, dtype='float')

    # Set night limits 
    night_limits = [0, 180524120000, 180525120000, 180526120000, 180527120000]
                    
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
                        # Write global maps to txtfiles
                        # np.savetxt(f"{dir}calib_{wavnb}_zonalpert_maps_night_{inight}.txt", zonalpert[ifilt, :, :])