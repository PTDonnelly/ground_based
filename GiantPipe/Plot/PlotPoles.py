import os
from matplotlib import projections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator)
import Globals
from Tools.VisirFilterInfo import Wavenumbers

def PlotPolesFromGlobal(globalmap):
    """ Plotting pole maps using stored global maps array """

    print('Mapping pole maps...')
    # If subdirectory does not exist, create it
    dir = '../outputs/pole_maps_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    # Define local inputs
    nx, ny = 720, 360                                           # Dimensions of an individual global map
    res    = ny / 180.                                          # Resolution of cmaps (i.e. pixels number per degree of longitude/latitude)
    lat = np.arange(-89.75,90,step=0.5)                         # Latitude range from pole-to-pole
    central_lon   = 0.                                          # Central longitude for polar projection
    central_lat   = 90.                                         # Absolute central latitude value for polar projection 
    lat_lim       = 10.                                         # Absolute latitude limit for polar projection 
    dmeridian     = 30                                          # Meridian lines step, interger to please xlocs parameter in gridlines
    dparallel     = 10                                          # Parallel lines step, interger to please ylocs in gridlines
    num_merid     = int(360/dmeridian + 1)                      # Number of meridian lines
    num_parra     = int((90-np.abs(lat_lim)) / dparallel + 1)   # Number of parallel lines per hemisphere
    degree_symbol = u'\u00B0'
    lond = np.linspace(0, 360, num_merid)
    lon_to_write = 45*np.ones(len(lond))                        # Array to set on which longitude will be written latitude labels
    lat_north_labels = np.linspace(lat_lim, central_lat, 10)
    lat_south_labels = np.linspace(-central_lat, -lat_lim, 10)


    #  Subplot figure with both hemisphere
    for ifilt in range(Globals.nfilters):
        # Set extreme values for mapping
        max = np.nanmax(globalmap[ifilt, :, :]) 
        min = np.nanmin(globalmap[ifilt, :, :])
        # Northern pole subplot
        proj = ccrs.AzimuthalEquidistant(central_longitude=central_lon, \
                                        central_latitude=central_lat, globe=None)
        ax1 = plt.subplot2grid((1, 2), (0, 0), projection = proj)
        ax1.imshow(globalmap[ifilt, :, :], \
                        transform=ccrs.PlateCarree(central_longitude=central_lon), \
                        origin='lower', extent=[0, 360, -90, 90], vmin=min, vmax=max, \
                        regrid_shape=1000, cmap='inferno')
        # Define locations of longitude labels and write them
        CustomLongitudeLabels(ax1, central_lat, lat_lim, num_merid, degree_symbol)
        # Define locations of latitude labels and write them along lon_to_write array
        CustomLatitudeLabels(ax1, central_lat, lat_lim, num_parra, lon_to_write, degree_symbol)
        # Set the boundary of the polar projection
        CustomBoundaryLatitude(ax1, proj, lat_lim)
        # Draw the gridlines without the default labels        
        ax1.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), color="grey", y_inline=False, \
                        xlocs=range(-180,180,dmeridian), ylocs=range(-90,91,dparallel), linestyle='--')
        # Southern pole subplot
        proj = ccrs.AzimuthalEquidistant(central_longitude=central_lon, \
                                        central_latitude=-central_lat, globe=None)
        ax2 = plt.subplot2grid((1, 2), (0, 1), projection = proj)
        ax2.imshow(globalmap[ifilt, :, :], \
                        transform=ccrs.PlateCarree(central_longitude=central_lon), \
                        origin='lower', extent=[0, 360, -90, 90], vmin=min, vmax=max, \
                        regrid_shape=1000, cmap='inferno')
        # Define locations of longitude labels and write them
        CustomLongitudeLabels(ax2, -central_lat, -lat_lim, num_merid, degree_symbol)
        # Define locations of latitude labels and write them along lon_to_write array
        CustomLatitudeLabels(ax2, -central_lat, -lat_lim, num_parra, lon_to_write, degree_symbol)
        # Set the boundary of the polar projection
        CustomBoundaryLatitude(ax2, proj, -lat_lim)
        # Draw the gridlines without the default labels        
        ax2.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), color="grey", y_inline=False, \
                        xlocs=range(-180,180,dmeridian), ylocs=range(-90,91,dparallel),linestyle='--')
        # Save pole map figure of the current filter 
        filt = Wavenumbers(ifilt)
        plt.savefig(f"{dir}{filt}_pole_maps.png", dpi=900)
        plt.savefig(f"{dir}{filt}_pole_maps.eps", dpi=900)
        # Clear figure to avoid overlapping between plotting subroutines
        plt.clf()
    
        # Northern pole figure
        projection = ccrs.AzimuthalEquidistant(central_longitude=central_lon, \
                                                central_latitude=central_lat, globe=None)
        ax = plt.axes(projection=projection)
        im = ax.imshow(globalmap[ifilt, :, :], \
                        transform=ccrs.PlateCarree(central_longitude=central_lon), \
                        origin='lower', extent=[0, 360, -90, 90], vmin=min, vmax=max, \
                        regrid_shape=1000, cmap='inferno')
        # Define locations of longitude labels and write them
        CustomLongitudeLabels(ax, central_lat, lat_lim, num_merid, degree_symbol)
        # Define locations of latitude labels and write them along lon_to_write array
        CustomLatitudeLabels(ax, central_lat, lat_lim, num_parra, lon_to_write, degree_symbol)
        # Set the boundary of the polar projection
        CustomBoundaryLatitude(ax, projection, lat_lim)
        # Draw the gridlines without the default labels        
        ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), color="grey", y_inline=False, \
                        xlocs=range(-180,180,dmeridian), ylocs=range(-90,91,dparallel), linestyle='--')
        # Define a colorbar
        cbar = plt.colorbar(im, ax=ax, extend='both', fraction=0.046, pad=0.04)
        #cbar.ax.tick_params(labelsize=15)
        cbar.set_label("Brightness Temperature [K]")
        # Save north pole map figure of the current filter 
        filt = Wavenumbers(ifilt)
        plt.savefig(f"{dir}{filt}_north_pole_maps.png", dpi=900)
        plt.savefig(f"{dir}{filt}_north_pole_maps.eps", dpi=900)
        # Clear figure to avoid overlapping between plotting subroutines
        plt.clf()
    
        # Southern pole figure
        projection = ccrs.AzimuthalEquidistant(central_longitude=central_lon, \
                                                central_latitude=-central_lat, globe=None)
        ax = plt.axes(projection = projection)
        im = ax.imshow(globalmap[ifilt, :, :], \
                        transform=ccrs.PlateCarree(central_longitude=central_lon), \
                        origin='lower', extent=[0, 360, -90, 90], vmin=min, vmax=max, \
                        regrid_shape=1000, cmap='inferno')
        # Define locations of longitude labels and write them
        CustomLongitudeLabels(ax, -central_lat, -lat_lim, num_merid, degree_symbol)
        # Define locations of latitude labels and write them along lon_to_write array
        CustomLatitudeLabels(ax, -central_lat, -lat_lim, num_parra, lon_to_write, degree_symbol)
        # Set the boundary of the polar projection
        CustomBoundaryLatitude(ax, projection, -lat_lim)
        # Draw the gridlines without the default labels        
        ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), color="grey", y_inline=False, \
                        xlocs=range(-180,180,dmeridian), ylocs=range(-90,91,dparallel),linestyle='--')
        # Define a colorbar
        cbar = plt.colorbar(im, ax=ax, extend='both', fraction=0.046, pad=0.04)
        #cbar.ax.tick_params(labelsize=15)
        cbar.set_label("Brightness Temperature [K]")
        # Save south pole map figure of the current filter 
        filt = Wavenumbers(ifilt)
        plt.savefig(f"{dir}{filt}_south_pole_maps.png", dpi=900)
        plt.savefig(f"{dir}{filt}_south_pole_maps.eps", dpi=900)
        # Clear figure to avoid overlapping between plotting subroutines
        plt.clf()

def CustomLongitudeLabels(ax, clat, lat_lim, num_merid, degree_symbol):
    # for locations of (meridional/longitude) labels
    lond = np.linspace(0,360, num_merid)
    latd = np.zeros(len(lond))
    for (alon, alat) in zip(lond, latd):
        if clat>0:
            projx1, projy1 = ax.projection.transform_point(alon, lat_lim-2., ccrs.Geodetic())
        if clat<0:
            projx1, projy1 = ax.projection.transform_point(alon, lat_lim+2., ccrs.Geodetic())
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
        if (alon<360. and alon>0):
            txt = f"{int(360-alon)}"+degree_symbol+'W'
            ax.text(projx1, projy1, txt, \
                    va=va, ha=ha, color='black',fontsize = 10)
        if (alon==0):
            txt = f"{int(alon)}"+degree_symbol+'W'
            ax.text(projx1, projy1, txt, \
                    va=va, ha=ha, color='black',fontsize = 10)

def CustomLatitudeLabels(ax, clat, lat_lim, num_parra, lon_to_write, degree_symbol):    
    lat = np.linspace(-90, lat_lim, num_parra) if clat<0 else np.linspace(lat_lim, 90, num_parra)
    for (alon, alat) in zip(lon_to_write, lat):
        if(clat<0 and alat<=-20 and alat>-90):
            projx1, projy1 = ax.projection.transform_point(alon, alat, ccrs.Geodetic())
            txt = f"{int(alat)}"+ degree_symbol
            ax.text(projx1, projy1, \
                       txt, va='center', ha='center', \
                        color='white',fontsize = 10) 
        if(clat>0 and alat>=20 and alat<90):
            projx1, projy1 = ax.projection.transform_point(alon, alat, ccrs.Geodetic())
            txt = f"{int(alat)}"+degree_symbol
            ax.text(projx1, projy1, \
                        txt, va='center', ha='center', \
                        color='white',fontsize = 10)

def CustomBoundaryLatitude(ax, proj, lat_lim):
    # add extra padding to the plot extents
    # These 2 lines of code grab extents in projection coordinates
    lonlatproj = ccrs.PlateCarree()
    _, y_min = proj.transform_point(0, lat_lim, lonlatproj)  #(0.0, -3189068.5)
    r_limit=np.abs(y_min)
    r_extent = r_limit*1.0001
    ax.set_xlim(-r_extent, r_extent)
    ax.set_ylim(-r_extent, r_extent)

    # Calculation of the circular boundary path in function to lat_lim
    circle_path = mpath.Path.unit_circle()
    circle_path = mpath.Path(circle_path.vertices.copy() * r_limit,
                               circle_path.codes.copy())
    # set circle boundary
    ax.set_boundary(circle_path)
    # Remove black line contour of the polar projection (cosmetic, could be set to True)
    ax.set_frame_on(False)
