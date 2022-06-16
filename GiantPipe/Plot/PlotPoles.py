import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import cartopy.crs as ccrs
import Globals
from Tools.VisirFilterInfo import Wavenumbers

def PlotPolesFromGlobal(globalmap):
    print('Mapping pole maps...')
    # If subdirectory does not exist, create it
    dir = '../outputs/pole_maps_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    # Define local inputs
    to_lon = 180. # East postive
    to_lat = 20.

    # Plotting pole map using stored global maps array 
    for ifilt in range(Globals.nfilters):
        # Northern pole subplot
        ax1 = plt.subplot2grid((1, 2), (0, 0))
        proj=ccrs.AzimuthalEquidistant(central_longitude=to_lon, central_latitude=to_lat, globe=None)
        ax1 = plt.axes(projection=proj)
        ax1.imshow(globalmap[ifilt, :, :], transform=ccrs.PlateCarree(central_longitude=to_lon), origin='lower', regrid_shape=1000)

        # Southern pole subplot
        ax2 = plt.subplot2grid((1, 2), (0, 1))
        proj=ccrs.AzimuthalEquidistant(central_longitude=to_lon, central_latitude=-to_lat, globe=None)
        ax2 = plt.axes(projection=proj)
        ax2.imshow(globalmap[ifilt, :, :], transform=ccrs.PlateCarree(central_longitude=to_lon), origin='lower', regrid_shape=1000)

        # Save pole map figure of the current filter 
        filt = Wavenumbers(ifilt)
        plt.savefig(f"{dir}{filt}_pole_maps.png", dpi=900)
        plt.savefig(f"{dir}{filt}_pole_maps.eps", dpi=900)
        # Clear figure to avoid overlapping between plotting subroutines
        plt.clf()

# #def polarlongitudes(clat,clon,limit,dmeridian,dparallel):
#     if clat>0:
#             latlim=limit
#     if clat<0:
#             latlim=-limit
    
#     num_merid = int(360/dmeridian + 1)
#     num_parra = int((90-np.abs(latlim))/dparallel + 1)
#     print(num_merid,num_parra)
    
#     theta = np.linspace(0, 2*np.pi, 120)
#     verts = np.vstack([np.sin(theta), np.cos(theta)]).T
#     center, radius = [0.5, 0.5], 0.5
#     circle = mpath.Path(verts * radius + center)

#     # for label alignment
#     va = 'center' # also bottom, top
#     ha = 'center' # right, left
#     degree_symbol=u'\u00B0'

#     # for locations of (meridional/longitude) labels
#     lond = np.linspace(0,360, num_merid)
#     latd = np.zeros(len(lond))

#     for (alon, alat) in zip(lond, latd):
#         if clat>0:
#             projx1, projy1 = ax.projection.transform_point(alon, latlim-2., ccrs.Geodetic())
#         if clat<0:
#             projx1, projy1 = ax.projection.transform_point(alon, latlim+2., ccrs.Geodetic())
#         if alon>0 and alon<180:
#             ha = 'left'
#             va = 'center'
#         if alon>180 and alon<360:
#             ha = 'right'
#             va = 'center'
#         if np.abs(alon-180)<0.01:
#             ha = 'center'
#             if clat==90:
#                 va = 'bottom'
#             if clat==-90:
#                 va = 'top'
#         if alon==0.:
#             ha = 'center'
#             if clat==-90:
#                 va = 'bottom'
#             if clat==90:
#                 va = 'top'
#         if (alon<360. and alon>0):
#             txt =  '{0}'.format(str(int(360-alon)))+degree_symbol+'W'
#             ax.text(projx1, projy1, txt, \
#                     va=va, ha=ha, color='black',fontsize = 10)
#         if (alon==0):
#             txt =  '{0}'.format(str(int(alon)))+degree_symbol+'W'
#             ax.text(projx1, projy1, txt, \
#                     va=va, ha=ha, color='black',fontsize = 10)


#     # for locations of (meridional/longitude) labels
#     # select longitude: 315 for label positioning
#     lond2 = 45*np.ones(len(lond))
#     if clat<0:
#         latd2 = np.linspace(-90, latlim, num_parra)
#     if clat>0:
#         latd2 = np.linspace(latlim, 90, num_parra)
#     va, ha = 'center', 'center'
#     for (alon, alat) in zip(lond2, latd2):
#         if(clat<0 and alat<=-20 and alat>-90):
#             projx1, projy1 = ax.projection.transform_point(alon, alat, ccrs.Geodetic())
#             txt =  '{0}'.format(str(int(alat)))+degree_symbol
#             ax.text(projx1, projy1, \
#                        txt, va=va, ha=ha, \
#                         color='black',fontsize = 10) 
#         if(clat>0 and alat>=20 and alat<90):
#             projx1, projy1 = ax.projection.transform_point(alon, alat, ccrs.Geodetic())
#             txt =  '{0}'.format(str(int(alat)))+degree_symbol
#             ax.text(projx1, projy1, \
#                         txt, va=va, ha=ha, \
#                         color='black',fontsize = 10)

#     # add extra padding to the plot extents
#     # These 2 lines of code grab extents in projection coordinates
#     lonlatproj = ccrs.PlateCarree()
#     _, y_min = proj.transform_point(0, latlim, lonlatproj)  #(0.0, -3189068.5)
#     x_max, _ = proj.transform_point(90, latlim, lonlatproj) #(3189068.5, 0)
#     r_limit=np.abs(y_min)
#     r_extent = r_limit*1.0001
#     ax.set_xlim(-r_extent, r_extent)
#     ax.set_ylim(-r_extent, r_extent)

#     # Prep circular boundary
#     circle_path = mpath.Path.unit_circle()
#     circle_path = mpath.Path(circle_path.vertices.copy() * r_limit,
#                                circle_path.codes.copy())

#     #set circle boundary
#     ax.set_boundary(circle_path)
#     #hide frame
#     ax.set_frame_on(True)  #hide the rectangle frame