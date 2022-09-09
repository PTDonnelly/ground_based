import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import Globals
from Tools.VisirFilterInfo import Wavenumbers
from matplotlib import ticker

def PlotPolesFromGlobal(dataset):
    """ Plotting pole maps using stored global maps array """

    print('Mapping pole maps...')
    # If subdirectory does not exist, create it
    dir = f'../outputs/{dataset}/pole_maps_figures/'
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
    Nfilters = Globals.nfilters if dataset == '2018May' else 11
    
    #  Subplot figure with both hemisphere
    for ifilt in range(Nfilters):
        if dataset == '2018May':
            # Retrive wavenumber corresponding to ifilt
            filt = Wavenumbers(ifilt)
            adj_location = 'average' if ifilt < 10 else 'southern'
            globalmap[ifilt, :, :] = np.load(f'../outputs/{dataset}/global_maps_figures/calib_{filt}_global_maps_{adj_location}_adj.npy')
        elif dataset == '2022July' or dataset == '2022August':
            if ifilt == 4: 
                filt = Wavenumbers(ifilt+1)
            elif ifilt > 5: 
                filt = Wavenumbers(ifilt+2)
            else:
                filt = Wavenumbers(ifilt)
            globalmap[ifilt, :, :] = np.load(f'../outputs/{dataset}/global_maps_figures/calib_{filt}_global_maps.npy')
        # Set extreme values for mapping
        max = np.nanmax(globalmap[ifilt, :, :]) 
        min = np.nanmin(globalmap[ifilt, :, :])
        if ifilt < 6 or ifilt > 7:
            northkeep = ((lat > 15) & (lat < 75))
            max_north = np.nanmax(globalmap[ifilt, northkeep, :])
            min_north = np.nanmin(globalmap[ifilt, northkeep, :]) 
        southkeep = ((lat < -15) & (lat > -75))
        max_south = np.nanmax(globalmap[ifilt, southkeep, :])
        min_south = np.nanmin(globalmap[ifilt, southkeep, :])


        plt.figure(figsize=(15, 8), dpi=300)
        # Northern pole subplot
        proj = ccrs.AzimuthalEquidistant(central_longitude=central_lon, \
                                        central_latitude=central_lat, globe=None)
        ax1 = plt.subplot2grid((1, 2), (0, 0), projection = proj)
        ax1.imshow(globalmap[ifilt, :, :], \
                        transform=ccrs.PlateCarree(central_longitude=central_lon), \
                        origin='lower', extent=[0, 360, -90, 90], vmin=min, vmax=max, \
                        regrid_shape=1000, cmap='inferno')
        # Define locations of longitude labels and write them
        CustomLongitudeLabels(ax1, central_lat, lat_lim, num_merid)
        # Define locations of latitude labels and write them along lon_to_write array
        CustomLatitudeLabels(ax1, central_lat, lat_lim, num_parra, num_merid, lon_to_write)
        # Set the boundary of the polar projection
        CustomBoundaryLatitude(ax1, proj, lat_lim)
        # Draw the gridlines without the default labels        
        ax1.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), color="grey", y_inline=False, \
                        xlocs=range(-180,180,dmeridian), ylocs=range(-90,91,dparallel), linestyle='--')
        # Southern pole subplot
        proj = ccrs.AzimuthalEquidistant(central_longitude=central_lon, \
                                        central_latitude=-central_lat, globe=None)
        ax2 = plt.subplot2grid((1, 2), (0, 1), projection = proj)
        im = ax2.imshow(globalmap[ifilt, :, :], \
                        transform=ccrs.PlateCarree(central_longitude=central_lon), \
                        origin='lower', extent=[0, 360, -90, 90], vmin=min, vmax=max, \
                        regrid_shape=1000, cmap='inferno')
        # Define locations of longitude labels and write them
        CustomLongitudeLabels(ax2, -central_lat, -lat_lim, num_merid)
        # Define locations of latitude labels and write them along lon_to_write array
        CustomLatitudeLabels(ax2, -central_lat, -lat_lim, num_parra, num_merid, lon_to_write)
        # Set the boundary of the polar projection
        CustomBoundaryLatitude(ax2, proj, -lat_lim)
        # Draw the gridlines without the default labels        
        ax2.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), color="grey", y_inline=False, \
                        xlocs=range(-180,180,dmeridian), ylocs=range(-90,91,dparallel),linestyle='--')
        # Define a colorbar
        cax = plt.axes([0.1, 0.1, 0.8, 0.03])
        cbar = plt.colorbar(im, cax=cax, extend='both', orientation='horizontal')
        #cbar.ax.tick_params(labelsize=15)
        cbar.set_label("Brightness Temperature [K]")
        # Save pole map figure of the current filter
        if dataset== '2018May': 
            plt.savefig(f"{dir}calib_{filt}_pole_maps_{adj_location}_adj.png", dpi=300)
            plt.savefig(f"{dir}calib_{filt}_pole_maps_{adj_location}_adj.eps", dpi=300)
        else:
            plt.savefig(f"{dir}calib_{filt}_pole_maps.png", dpi=300)
            plt.savefig(f"{dir}calib_{filt}_pole_maps.eps", dpi=300)
        # Clear figure to avoid overlapping between plotting subroutines
        plt.close()
    

        plt.figure(figsize=(9, 5), dpi=300)
        if ifilt < 6 or ifilt > 7:
            # Northern pole figure
            PlotOnePole(img=globalmap[ifilt,:,:], vmin=min_north, vmax=max_north, \
                central_longitude=central_lon, central_latitude=central_lat, \
                latitude_limit=lat_lim, number_meridian=num_merid, number_parrallel=num_parra, \
                longitude_to_write=lon_to_write, delta_meridian=dmeridian, delta_parallel=dparallel)
            # Save north pole map figure of the current filter
            if dataset== '2018May':
                plt.savefig(f"{dir}calib_{filt}_north_pole_maps_{adj_location}_adj.png", dpi=300)
                plt.savefig(f"{dir}calib_{filt}_north_pole_maps_{adj_location}_adj.eps", dpi=300)
            else:
                plt.savefig(f"{dir}calib_{filt}_north_pole_maps.png", dpi=300)
                plt.savefig(f"{dir}calib_{filt}_north_pole_maps.eps", dpi=300)
            # Clear figure to avoid overlapping between plotting subroutines
            plt.close()
    
        # Southern pole figure
        plt.figure(figsize=(9, 5), dpi=300)
        PlotOnePole(img=globalmap[ifilt,:,:], vmin=min_south, vmax=max_south, \
            central_longitude=central_lon, central_latitude=-central_lat, \
            latitude_limit=-lat_lim, number_meridian=num_merid, number_parrallel=num_parra, \
            longitude_to_write=lon_to_write, delta_meridian=dmeridian, delta_parallel=dparallel)
        # Save south pole map figure of the current filter 
        if dataset == '2018May':
            plt.savefig(f"{dir}calib_{filt}_south_pole_maps_{adj_location}_adj.png", dpi=300)
            plt.savefig(f"{dir}calib_{filt}_south_pole_maps_{adj_location}_adj.eps", dpi=300)
        else:
            plt.savefig(f"{dir}calib_{filt}_south_pole_maps.png", dpi=300)
            plt.savefig(f"{dir}calib_{filt}_south_pole_maps.eps", dpi=300)
        # Clear figure to avoid overlapping between plotting subroutines
        plt.close()

def PlotOnePole(img, vmin, vmax, central_longitude, central_latitude, latitude_limit, \
                number_meridian, number_parrallel, longitude_to_write, \
                delta_meridian, delta_parallel):
    """ Setting routine to make pretty polar projection for a single pole"""

    projection = ccrs.AzimuthalEquidistant(central_longitude=central_longitude, \
                                            central_latitude=central_latitude, globe=None)
    ax = plt.axes(projection = projection)
    im = ax.imshow(img, transform=ccrs.PlateCarree(central_longitude=central_longitude), \
                    origin='lower', extent=[0, 360, -90, 90], vmin=vmin, vmax=vmax, \
                    regrid_shape=1000, cmap='inferno')
    # Define locations of longitude labels and write them
    CustomLongitudeLabels(axes=ax, clat=central_latitude, lat_lim=latitude_limit, num_merid=number_meridian)
    # Define locations of latitude labels and write them along lon_to_write array
    CustomLatitudeLabels(axes=ax, clat=central_latitude, lat_lim=latitude_limit, num_parra=number_parrallel, 
                            num_merid=number_meridian, lon_to_write=longitude_to_write)
    # Set the boundary of the polar projection
    CustomBoundaryLatitude(axes=ax, proj=projection, lat_lim=latitude_limit)
    # Draw the gridlines without the default labels        
    ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), color="grey", y_inline=False, \
                    xlocs=range(-180,180,delta_meridian), ylocs=range(-90,91,delta_parallel),linestyle='--')
    # Define a colorbar
    cax = plt.axes([0.85, 0.1, 0.03, 0.8])
    cbar = plt.colorbar(im, cax=cax, format="%.0f", extend='both')#, fraction=0.046, pad=0.05)
    cbar.ax.tick_params(labelsize=20)
    cbar.locator = ticker.MaxNLocator(nbins=10)
    cbar.update_ticks()
    cbar.set_label("Brightness Temperature [K]", size=15)

def CustomLongitudeLabels(axes, clat, lat_lim, num_merid):
    """ Small routine to define the longitude labels of the polar projection """

    # Local variable definition
    degree_symbol = u'\u00B0'               # degree symbol in UTF code
    lond = np.linspace(0,360, num_merid)    # array of longitude range dimension
    latd = np.zeros(len(lond))              # array of latitude with longitude range dimension

    for (alon, alat) in zip(lond, latd):
        if clat>0:
            projx, projy = axes.projection.transform_point(alon, lat_lim-2., ccrs.Geodetic())
        if clat<0:
            projx, projy = axes.projection.transform_point(alon, lat_lim+2., ccrs.Geodetic())
        # Define the labeling orientation depending of the longitude value
        if alon>0 and alon<180:
            ha = 'left'
            va = 'center'
        if alon>180 and alon<360:
            ha = 'right'
            va = 'center'
        if np.abs(alon-180)<0.01:
            ha = 'center'
            if clat==90:
                va = 'bottom'
            if clat==-90:
                va = 'top'
        if alon==0.:
            ha = 'center'
            if clat==-90:
                va = 'bottom'
            if clat==90:
                va = 'top'
        # Write the longitude labels 
        if (alon<360. and alon>0):
            txt = f"{int(360-alon)}"+degree_symbol+'W'
            axes.text(projx, projy, txt, va=va, ha=ha, color='black',fontsize = 20)
        if (alon==0):
            txt = f"{int(alon)}"+degree_symbol+'W'
            axes.text(projx, projy, txt, va=va, ha=ha, color='black',fontsize = 20)

def CustomLatitudeLabels(axes, clat, lat_lim, num_parra, num_merid, lon_to_write):
    """ Small routine to define the latitude labels of the polar projection """

    # Local variable definition
    degree_symbol = u'\u00B0'               # degree symbol in UTF code
    lond = np.linspace(0, 360, num_merid)   # array of longitude range dimension...
    lond.fill(lon_to_write)                 #... which is fill with the longitude value on which will be written latitude labels

    # Define latitude array in function of the current hemisphere (through clat value)
    lat = np.linspace(-90, lat_lim, num_parra) if clat<0 else np.linspace(lat_lim, 90, num_parra)
    for (alon, alat) in zip(lond, lat):
        # Southern hemisphere labelisation
        if (clat < 0 and alat <= -20 and alat >- 90):
            projx, projy = axes.projection.transform_point(alon, alat, ccrs.Geodetic())
            txt = f"{int(alat)}"+ degree_symbol
            axes.text(projx, projy, txt, va='center', ha='center', color='white',fontsize = 15) 
        # Northern hemisphere labelisation
        if (clat > 0 and alat >= 20 and alat < 90):
            projx, projy = axes.projection.transform_point(alon, alat, ccrs.Geodetic())
            txt = f"{int(alat)}"+degree_symbol
            axes.text(projx, projy, txt, va='center', ha='center', color='white',fontsize = 15)

def CustomBoundaryLatitude(axes, proj, lat_lim):
    """ Small routine to define the latitude limit of the polar projection """

    # Calculate the y-axis limit in PlateCarree projection corresponding to the polar projection latitude limit 
    _, y_min = proj.transform_point(0, lat_lim, ccrs.PlateCarree())
    r_limit=np.abs(y_min)
    # Extend y_min to 0.01 pourcent to keep lat_lim value mapping
    r_extent = r_limit*1.0001
    # Set x and y limit axis in PlateCarree projection depending to y_min (and to lat_lim by extension)
    axes.set_xlim(-r_extent, r_extent)
    axes.set_ylim(-r_extent, r_extent)
    # Calculation of the circular boundary path in function to lat_lim
    circle_path = mpath.Path.unit_circle()
    circle_path = mpath.Path(circle_path.vertices.copy() * r_limit, circle_path.codes.copy())
    # Set circle boundary
    axes.set_boundary(circle_path)
    # Remove black line contour of the polar projection (cosmetic, could be set to True)
    axes.set_frame_on(False)
