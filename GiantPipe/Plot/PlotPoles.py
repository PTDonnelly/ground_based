import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import matplotlib.image as image
import cartopy.crs as ccrs
import Globals
from Tools.SetWave import SetWave
from matplotlib import ticker

def PlotPolesFromGlobal(dataset, per_night):
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
    if per_night == True:
        Nnight = 4
        globalmap     = np.empty((Nnight, Globals.nfilters, Globals.ny, Globals.nx))
    else:
        globalmap     = np.empty((Globals.nfilters, Globals.ny, Globals.nx))
    Nfilters = Globals.nfilters if dataset == '2018May' or '2018May_completed' else 11
    
    # Load JunoCam amateur images
    Junosubdir = "/Users/deborah/Documents/Research/Observations/archives/JunoCam_PJ13_amateur_processing/"
    JunoCamVisible_north = image.imread(f'{Junosubdir}JunoCam_visible_north_PJ13.png')
    JunoCamVisible_south = image.imread(f'{Junosubdir}JunoCam_visible_south_PJ13.png')#PJ13_79-117_polS-maps-merged-CH4_v2.jpg')#JunoCam_visible_PJ13.png')
    JunoCamnearIR_north = image.imread(f'{Junosubdir}JunoCam_nearIR_north_PJ13.png')
    JunoCamnearIR_south = image.imread(f'{Junosubdir}JunoCam_nearIR_south_PJ13.png')#PJ13_S-hemis-maps-&grid_Labeld.jpg')#JunoCam_nearIR_PJ13.png')

    #  Subplot figure with both hemisphere
    for ifilt in range(Nfilters):
        if ifilt < 6 or ifilt > 7:
            if dataset == '2018May':
                # Retrive wavenumber corresponding to ifilt
                _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
                adj_location = 'average' if ifilt < 10 else 'southern'
                globalmap[ifilt, :, :] = np.load(f'../outputs/{dataset}/global_maps_figures/calib_{wavnb}_global_maps_{adj_location}_adj.npy')
            elif dataset == '2018May_completed' and per_night==False:
                # Retrive wavenumber corresponding to ifilt
                _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
                adj_location = 'average' if ifilt < 10 else 'southern'
                globalmap[ifilt, :, :] = np.load(f'../outputs/{dataset}/global_maps_figures/calib_{wavnb}_global_maps.npy')
            elif dataset == '2018May_completed' and per_night==True:
                # Retrive wavenumber corresponding to ifilt
                for inight in range(Nnight):
                    _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
                    globalmap[inight, ifilt, :, :] = np.load(f'../outputs/{dataset}/global_maps_per_night_figures/calib_{wavnb}_global_maps_night_{inight}.npy')
            elif dataset == '2022July' or dataset == '2022August':
                if ifilt == 4: 
                    _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt+1)
                elif ifilt > 5: 
                    _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt+2)
                else:
                    _, _, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
                globalmap[ifilt, :, :] = np.load(f'../outputs/{dataset}/global_maps_figures/calib_{wavnb}_global_maps.npy')
            
    if not per_night:
        # for ifilt in range(Nfilters):
        #     adj_location = 'average' if ifilt < 10 else 'southern'
        #     if ifilt < 6 or ifilt > 7:
        #         # Set extreme values for mapping
        #         max = np.nanmax(globalmap[ifilt, :, :]) 
        #         min = np.nanmin(globalmap[ifilt, :, :])
            
        #         northkeep = ((lat > 15) & (lat < 75))
        #         max_north = np.nanmax(globalmap[ifilt, northkeep, :])
        #         min_north = np.nanmin(globalmap[ifilt, northkeep, :]) 
        #         southkeep = ((lat < -15) & (lat > -75))
        #         max_south = np.nanmax(globalmap[ifilt, southkeep, :])
        #         min_south = np.nanmin(globalmap[ifilt, southkeep, :])


        #         plt.figure(figsize=(15, 5))
        #         # Northern pole subplot
        #         proj = ccrs.AzimuthalEquidistant(central_longitude=central_lon, \
        #                                         central_latitude=central_lat, globe=None)
        #         ax1 = plt.subplot2grid((1, 2), (0, 0), projection = proj)
        #         ax1.imshow(globalmap[ifilt, :, :], \
        #                         transform=ccrs.PlateCarree(central_longitude=central_lon), \
        #                         origin='lower', extent=[0, 360, -90, 90], vmin=min, vmax=max, \
        #                         regrid_shape=1000, cmap='inferno')
        #         # Define locations of longitude labels and write them
        #         CustomLongitudeLabels(ax1, central_lat, lat_lim, num_merid)
        #         # Define locations of latitude labels and write them along lon_to_write array
        #         CustomLatitudeLabels(ax1, central_lat, lat_lim, num_parra, num_merid, lon_to_write)
        #         # Set the boundary of the polar projection
        #         CustomBoundaryLatitude(ax1, proj, lat_lim)
        #         # Draw the gridlines without the default labels        
        #         ax1.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), color="grey", y_inline=False, \
        #                         xlocs=range(-180,180,dmeridian), ylocs=range(-90,91,dparallel), linestyle='--')
        #         # Southern pole subplot
        #         proj = ccrs.AzimuthalEquidistant(central_longitude=central_lon, \
        #                                         central_latitude=-central_lat, globe=None)
        #         ax2 = plt.subplot2grid((1, 2), (0, 1), projection = proj)
        #         im = ax2.imshow(globalmap[ifilt, :, :], \
        #                         transform=ccrs.PlateCarree(central_longitude=central_lon), \
        #                         origin='lower', extent=[0, 360, -90, 90], vmin=min, vmax=max, \
        #                         regrid_shape=1000, cmap='inferno')
        #         # Define locations of longitude labels and write them
        #         CustomLongitudeLabels(ax2, -central_lat, -lat_lim, num_merid)
        #         # Define locations of latitude labels and write them along lon_to_write array
        #         CustomLatitudeLabels(ax2, -central_lat, -lat_lim, num_parra, num_merid, lon_to_write)
        #         # Set the boundary of the polar projection
        #         CustomBoundaryLatitude(ax2, proj, -lat_lim)
        #         # Draw the gridlines without the default labels        
        #         ax2.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), color="grey", y_inline=False, \
        #                         xlocs=range(-180,180,dmeridian), ylocs=range(-90,91,dparallel),linestyle='--')
        #         # Define a colorbar
        #         cax = plt.axes([0.1, 0.1, 0.8, 0.03])
        #         cbar = plt.colorbar(im, cax=cax, extend='both', orientation='horizontal')
        #         #cbar.ax.tick_params(labelsize=15)
        #         cbar.set_label("Brightness Temperature [K]")
        #         # Save pole map figure of the current filter
        #         _, wavlg, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
        #         if dataset== '2018May': 
        #             plt.savefig(f"{dir}calib_{wavnb}_pole_maps_{adj_location}_adj.png", dpi=150, bbox_inches='tight')
        #             # plt.savefig(f"{dir}calib_{wavnb}_pole_maps_{adj_location}_adj.pdf", dpi=500, bbox_inches='tight')
        #         else:
        #             plt.savefig(f"{dir}calib_{wavnb}_pole_maps.png", dpi=150, bbox_inches='tight')
        #             # plt.savefig(f"{dir}calib_{wavnb}_pole_maps.pdf", dpi=500, bbox_inches='tight')
        #         # Clear figure to avoid overlapping between plotting subroutines
        #         plt.close()
            
        #         # Northern pole figure
        #         PlotOnePole(img=globalmap[ifilt,:,:], filter=wavlg, vmin=min_north, vmax=max_north, \
        #             central_longitude=central_lon, central_latitude=central_lat, \
        #             latitude_limit=lat_lim, number_meridian=num_merid, number_parrallel=num_parra, \
        #             longitude_to_write=lon_to_write, delta_meridian=dmeridian, delta_parallel=dparallel)
        #         # Save north pole map figure of the current filter
        #         if dataset== '2018May':
        #             plt.savefig(f"{dir}calib_{wavnb}_north_pole_maps_{adj_location}_adj.png", dpi=150, bbox_inches='tight')
        #             # plt.savefig(f"{dir}calib_{wavnb}_north_pole_maps_{adj_location}_adj.pdf", dpi=500, bbox_inches='tight')
        #         else:
        #             plt.savefig(f"{dir}calib_{wavnb}_north_pole_maps.png", dpi=150, bbox_inches='tight')
        #             # plt.savefig(f"{dir}calib_{wavnb}_north_pole_maps.pdf", dpi=500, bbox_inches='tight')
        #         # Clear figure to avoid overlapping between plotting subroutines
        #         plt.close()
        
        #         # Southern pole figure
        #         PlotOnePole(img=globalmap[ifilt,:,:], filter=wavlg, vmin=min_south, vmax=max_south, \
        #             central_longitude=central_lon, central_latitude=-central_lat, \
        #             latitude_limit=-lat_lim, number_meridian=num_merid, number_parrallel=num_parra, \
        #             longitude_to_write=lon_to_write, delta_meridian=dmeridian, delta_parallel=dparallel)
        #         # Save south pole map figure of the current filter 
        #         if dataset == '2018May':
        #             plt.savefig(f"{dir}calib_{wavnb}_south_pole_maps_{adj_location}_adj.png", dpi=150, bbox_inches='tight')
        #             # plt.savefig(f"{dir}calib_{wavnb}_south_pole_maps_{adj_location}_adj.pdf", dpi=500, bbox_inches='tight')
        #         else:
        #             plt.savefig(f"{dir}calib_{wavnb}_south_pole_maps.png", dpi=150, bbox_inches='tight')
        #             # plt.savefig(f"{dir}calib_{wavnb}_south_pole_maps.pdf", dpi=500, bbox_inches='tight')
        #         # Clear figure to avoid overlapping between plotting subroutines
        #         plt.close()

        
        # Create a subplots figures with all filters
        fig = plt.figure(figsize=(13, 16)) 
        # fig = plt.figure() 
        projection = ccrs.AzimuthalEquidistant(central_longitude=central_lon, \
                                                central_latitude=central_lat, globe=None)
        # Remove the frame of the empty subplot
        ax = plt.subplot2grid((5, 3), (0,1),  projection = projection)
        ax.set_frame_on(False)
        ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        # Insert JunoCam NearIR image
        ax = plt.subplot2grid((5, 3), (4,1),  projection = projection)
        ax.set_anchor('C')
        ax.set_frame_on(False)
        ax.imshow(JunoCamnearIR_north)
        ax.set_title(f"(l)                  JunoCam 889 nm", fontfamily='sans-serif', loc='left', fontsize=12)
        # Insert JunoCam RGB visible image
        ax = plt.subplot2grid((5, 3), (4,2),  projection = projection)
        ax.set_anchor('C')
        ax.set_frame_on(False)
        ax.imshow(JunoCamVisible_north)
        ax.set_title(f"(m)                  JunoCam Visible RGB", fontfamily='sans-serif', loc='left', fontsize=12)

        iax = 0
        for ifilt in [0,10,11,12,5,4,6,7,8,9,3,2,1]:
            if ifilt < 6 or ifilt > 7:
                _, wavlg, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
                # # Set extreme values for mapping
                max = np.nanmax(globalmap[ifilt, :, :]) 
                min = np.nanmin(globalmap[ifilt, :, :])
            
                northkeep = ((lat > 15) & (lat < 75))
                max_north = np.nanmax(globalmap[ifilt, northkeep, :])
                min_north = np.nanmin(globalmap[ifilt, northkeep, :]) 

                irow = [0,1,1,1,2,2,2,3,3,3,4]
                icol = [0,0,1,2,0,1,2,0,1,2,0]
                ititle = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)']
                
                ax = plt.subplot2grid((5, 3), (irow[iax],icol[iax]),  projection = projection)
                im = ax.imshow(globalmap[ifilt,:,:], transform=ccrs.PlateCarree(central_longitude=central_lon), \
                                origin='lower', extent=[0, 360, -90, 90], vmin=min_north, vmax=max_north, \
                                regrid_shape=1000, cmap='inferno')
                # Define locations of longitude labels and write them
                CustomLongitudeLabels(axes=ax, clat=central_lat, lat_lim=lat_lim, num_merid=num_merid)
                # Define locations of latitude labels and write them along lon_to_write array
                CustomLatitudeLabels(axes=ax, clat=central_lat, lat_lim=lat_lim, num_parra=num_parra, 
                                        num_merid=num_merid, lon_to_write=lon_to_write)
                # Set the boundary of the polar projection
                CustomBoundaryLatitude(axes=ax, proj=projection, lat_lim=lat_lim)
                # Draw the gridlines without the default labels        
                ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), color="grey", y_inline=False, \
                                xlocs=range(-180,180,dmeridian), ylocs=range(-90,91,dparallel),linestyle='--')
                ax.set_title(ititle[iax]+f"                                 {wavlg}"+r" $\mu$m", fontfamily='sans-serif', loc='left', fontsize=12)
                # Define a colorbar
                cbar = plt.colorbar(im, ax=ax, format="%.0f", extend='both', fraction=0.036, pad=0.1)
                cbar.ax.tick_params(labelsize=10)
                cbar.locator = ticker.MaxNLocator(nbins=10)
                cbar.update_ticks()
                cbar.set_label(r" T$_{B}$ [K]", size=13)

                iax +=1
        # Save north pole map figure of the current filter 
        plt.savefig(f"{dir}calib_all_north_pole_maps.png", dpi=150, bbox_inches='tight')
        plt.savefig(f"{dir}calib_all_north_pole_maps.pdf", dpi=500, bbox_inches='tight')
        # Clear figure to avoid overlapping between plotting subroutines
        plt.close()


        # Create a subplots figures with all filters

        circle1=plt.Circle((115, 215), 20, color='red', fill=False)
        circle2=plt.Circle((85, 320), 20, color='red', fill=False)
        circle3=plt.Circle((565, 220), 20, color='red', fill=False)
        circle4=plt.Circle((595, 345), 20, color='red', fill=False)
        circle5=plt.Circle((625, 395), 20, color='red', fill=False)
        circle6=plt.Circle((525, 160), 20, color='red', fill=False)
        circle7=plt.Circle((440, 100), 20, color='red', fill=False)
        circle8=plt.Circle((240, 475), 20, color='red', fill=False)
        arrow = mpatches.FancyArrowPatch((380,380),(290,330), mutation_scale=20, color="yellow")
        
        fig = plt.figure(figsize=(13, 16)) 
        # fig = plt.figure() 
        projection = ccrs.AzimuthalEquidistant(central_longitude=central_lon, \
                                                central_latitude=-central_lat, globe=None)
        # Remove the frame of the empty subplot
        ax = plt.subplot2grid((5, 3), (0,1),  projection = projection)
        ax.set_frame_on(False)
        ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        # Insert JunoCam NearIR image
        ax = plt.subplot2grid((5, 3), (4,1),  projection = projection)
        ax.set_anchor('C')
        ax.set_frame_on(False)
        ax.imshow(JunoCamnearIR_south)
        ax.set_title(f"(l)                  JunoCam 889 nm", fontfamily='sans-serif', loc='left', fontsize=12)
        # Insert JunoCam RGB visible image
        ax = plt.subplot2grid((5, 3), (4,2),  projection = projection)
        ax.set_anchor('C')
        ax.set_frame_on(False)
        ax.imshow(JunoCamVisible_south)
        ax.add_patch(circle1)
        ax.add_patch(circle2)
        ax.add_patch(circle3)
        ax.add_patch(circle4)
        ax.add_patch(circle5)
        ax.add_patch(circle6)
        ax.add_patch(circle7)
        ax.add_patch(circle8)
        ax.add_patch(arrow)
        ax.set_title(f"(m)                  JunoCam Visible RGB", fontfamily='sans-serif', loc='left', fontsize=12)

        iax = 0
        for ifilt in [0,10,11,12,5,4,6,7,8,9,3,2,1]:
            if ifilt < 6 or ifilt > 7:
                _, wavlg, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
                # # Set extreme values for mapping
                max = np.nanmax(globalmap[ifilt, :, :]) 
                min = np.nanmin(globalmap[ifilt, :, :])
            
                southkeep = ((lat < -15) & (lat > -75))
                max_south = np.nanmax(globalmap[ifilt, southkeep, :])
                min_south = np.nanmin(globalmap[ifilt, southkeep, :])

                irow = [0,1,1,1,2,2,2,3,3,3,4]
                icol = [0,0,1,2,0,1,2,0,1,2,0]
                ititle = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)']
                
                ax = plt.subplot2grid((5, 3), (irow[iax],icol[iax]),  projection = projection)
                im = ax.imshow(globalmap[ifilt,:,:], transform=ccrs.PlateCarree(central_longitude=central_lon), \
                                origin='lower', extent=[0, 360, -90, 90], vmin=min_south, vmax=max_south, \
                                regrid_shape=1000, cmap='inferno')
                # Define locations of longitude labels and write them
                CustomLongitudeLabels(axes=ax, clat=-central_lat, lat_lim=-lat_lim, num_merid=num_merid)
                # Define locations of latitude labels and write them along lon_to_write array
                CustomLatitudeLabels(axes=ax, clat=-central_lat, lat_lim=-lat_lim, num_parra=num_parra, 
                                        num_merid=num_merid, lon_to_write=lon_to_write)
                # Set the boundary of the polar projection
                CustomBoundaryLatitude(axes=ax, proj=projection, lat_lim=-lat_lim)
                # Draw the gridlines without the default labels        
                ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), color="grey", y_inline=False, \
                                xlocs=range(-180,180,dmeridian), ylocs=range(-90,91,dparallel),linestyle='--')
                ax.set_title(ititle[iax]+f"                                 {wavlg}"+r" $\mu$m", fontfamily='sans-serif', loc='left', fontsize=12)
                # Define a colorbar
                cbar = plt.colorbar(im, ax=ax, format="%.0f", extend='both', fraction=0.036, pad=0.1)
                cbar.ax.tick_params(labelsize=10)
                cbar.locator = ticker.MaxNLocator(nbins=10)
                cbar.update_ticks()
                cbar.set_label(r" T$_{B}$ [K]", size=13)

                iax +=1
        # Save south pole map figure of the current filter 
        plt.savefig(f"{dir}calib_all_south_pole_maps.png", dpi=150, bbox_inches='tight')
        plt.savefig(f"{dir}calib_all_south_pole_maps.pdf", dpi=500, bbox_inches='tight')
        # Clear figure to avoid overlapping between plotting subroutines
        plt.close()


    if per_night:
        for inight in range(Nnight):
            for ifilt in range(Nfilters):
                if ifilt < 6 or ifilt > 7:
                    # Set extreme values for mapping
                    max = np.nanmax(globalmap[:, ifilt, :, :]) 
                    min = np.nanmin(globalmap[:, ifilt, :, :])
                
                    northkeep = ((lat > 15) & (lat < 75))
                    max_north = np.nanmax(globalmap[:, ifilt, northkeep, :])
                    min_north = np.nanmin(globalmap[:, ifilt, northkeep, :]) 
                    southkeep = ((lat < -15) & (lat > -75))
                    max_south = np.nanmax(globalmap[:, ifilt, southkeep, :])
                    min_south = np.nanmin(globalmap[:, ifilt, southkeep, :])

                    plt.figure(figsize=(15, 5))
                    # Northern pole subplot
                    proj = ccrs.AzimuthalEquidistant(central_longitude=central_lon, \
                                                    central_latitude=central_lat, globe=None)
                    ax1 = plt.subplot2grid((1, 2), (0, 0), projection = proj)
                    ax1.imshow(globalmap[inight, ifilt, :, :], \
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
                    im = ax2.imshow(globalmap[inight, ifilt, :, :], \
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
                    _, wavlg, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
                    if dataset== '2018May': 
                        plt.savefig(f"{dir}calib_{wavnb}_pole_maps_night_{inight}_{adj_location}_adj.png", dpi=150, bbox_inches='tight')
                        # plt.savefig(f"{dir}calib_{wavnb}_pole_maps_{adj_location}_adj.pdf", dpi=500, bbox_inches='tight')
                    else:
                        plt.savefig(f"{dir}calib_{wavnb}_pole_maps_night_{inight}.png", dpi=150, bbox_inches='tight')
                        # plt.savefig(f"{dir}calib_{wavnb}_pole_maps.pdf", dpi=500, bbox_inches='tight')
                    # Clear figure to avoid overlapping between plotting subroutines
                    plt.close()
                
                    # Northern pole figure
                    PlotOnePole(img=globalmap[inight, ifilt,:,:], filter=wavlg, vmin=min_north, vmax=max_north, \
                        central_longitude=central_lon, central_latitude=central_lat, \
                        latitude_limit=lat_lim, number_meridian=num_merid, number_parrallel=num_parra, \
                        longitude_to_write=lon_to_write, delta_meridian=dmeridian, delta_parallel=dparallel)
                    # Save north pole map figure of the current filter
                    if dataset== '2018May':
                        plt.savefig(f"{dir}calib_{wavnb}_north_pole_maps_night_{inight}_{adj_location}_adj.png", dpi=150, bbox_inches='tight')
                        # plt.savefig(f"{dir}calib_{wavnb}_north_pole_maps_{adj_location}_adj.pdf", dpi=500, bbox_inches='tight')
                    else:
                        plt.savefig(f"{dir}calib_{wavnb}_north_pole_maps_night_{inight}.png", dpi=150, bbox_inches='tight')
                        # plt.savefig(f"{dir}calib_{wavnb}_north_pole_maps.pdf", dpi=500, bbox_inches='tight')
                    # Clear figure to avoid overlapping between plotting subroutines
                    plt.close()
            
                    # Southern pole figure
                    PlotOnePole(img=globalmap[inight, ifilt,:,:], filter=wavlg, vmin=min_south, vmax=max_south, \
                        central_longitude=central_lon, central_latitude=-central_lat, \
                        latitude_limit=-lat_lim, number_meridian=num_merid, number_parrallel=num_parra, \
                        longitude_to_write=lon_to_write, delta_meridian=dmeridian, delta_parallel=dparallel)
                    # Save south pole map figure of the current filter 
                    if dataset == '2018May':
                        plt.savefig(f"{dir}calib_{wavnb}_south_pole_maps_night_{inight}_{adj_location}_adj.png", dpi=150, bbox_inches='tight')
                        # plt.savefig(f"{dir}calib_{wavnb}_south_pole_maps_{adj_location}_adj.pdf", dpi=500, bbox_inches='tight')
                    else:
                        plt.savefig(f"{dir}calib_{wavnb}_south_pole_maps_night_{inight}.png", dpi=150, bbox_inches='tight')
                        # plt.savefig(f"{dir}calib_{wavnb}_south_pole_maps.pdf", dpi=500, bbox_inches='tight')
                    # Clear figure to avoid overlapping between plotting subroutines
                    plt.close()

            fig = plt.figure(figsize=(10, 18)) 
            # fig = plt.figure() 
            projection = ccrs.AzimuthalEquidistant(central_longitude=central_lon, \
                                                    central_latitude=-central_lat, globe=None)
            
            # Remove the frame of the empty subplot
            ax = plt.subplot2grid((6, 2), (0,1),  projection = projection)
            ax.set_frame_on(False)
            ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

            iax = 0
            for ifilt in [0,10,11,12,5,4,6,7,8,9,3,2,1]:
                if ifilt < 6 or ifilt > 7:
                    _, wavlg, wavnb, _, _ = SetWave(filename=None, wavelength=False, wavenumber=False, ifilt=ifilt)
                    # # Set extreme values for mapping
                    max = np.nanmax(globalmap[:, ifilt, :, :]) 
                    min = np.nanmin(globalmap[:, ifilt, :, :])
                
                    northkeep = ((lat > 15) & (lat < 75))
                    max_north = np.nanmax(globalmap[:, ifilt, northkeep, :])
                    min_north = np.nanmin(globalmap[:, ifilt, northkeep, :]) 
                    southkeep = ((lat < -15) & (lat > -75))
                    max_south = np.nanmax(globalmap[:, ifilt, southkeep, :])
                    min_south = np.nanmin(globalmap[:, ifilt, southkeep, :])

                    

                    irow = [0,1,1,2,2,3,3,4,4,5,5]
                    icol = [0,0,1,0,1,0,1,0,1,0,1]
                    ititle = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)']
                    
                    ax = plt.subplot2grid((6, 2), (irow[iax],icol[iax]),  projection = projection)
                    im = ax.imshow(globalmap[inight, ifilt,:,:], transform=ccrs.PlateCarree(central_longitude=central_lon), \
                                    origin='lower', extent=[0, 360, -90, 90], vmin=min_south, vmax=max_south, \
                                    regrid_shape=1000, cmap='inferno')
                    # Define locations of longitude labels and write them
                    CustomLongitudeLabels(axes=ax, clat=-central_lat, lat_lim=-lat_lim, num_merid=num_merid)
                    # Define locations of latitude labels and write them along lon_to_write array
                    CustomLatitudeLabels(axes=ax, clat=-central_lat, lat_lim=-lat_lim, num_parra=num_parra, 
                                            num_merid=num_merid, lon_to_write=lon_to_write)
                    # Set the boundary of the polar projection
                    CustomBoundaryLatitude(axes=ax, proj=projection, lat_lim=-lat_lim)
                    # Draw the gridlines without the default labels        
                    ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), color="grey", y_inline=False, \
                                    xlocs=range(-180,180,dmeridian), ylocs=range(-90,91,dparallel),linestyle='--')
                    ax.set_title(ititle[iax]+f"                                 {wavlg}"+r" $\mu$m", fontfamily='sans-serif', loc='left', fontsize=12)
                    # Define a colorbar
                    # cax = plt.axes([0.75, 0.1, 0.02, 0.8])
                    cbar = plt.colorbar(im, ax=ax, format="%.0f", extend='both', fraction=0.046, pad=0.15)
                    # cbar.ax.tick_params(labelsize=12)
                    # cbar.locator = ticker.MaxNLocator(nbins=10)
                    # cbar.update_ticks()
                    # cbar.set_label(r" T$_{B}$ [K]", size=15)

                    # ax.PlotOnePole(img=globalmap[ifilt,:,:], filter=wavnb, vmin=min_south, vmax=max_south, \
                    # central_longitude=central_lon, central_latitude=-central_lat, \
                    # latitude_limit=-lat_lim, number_meridian=num_merid, number_parrallel=num_parra, \
                    # longitude_to_write=lon_to_write, delta_meridian=dmeridian, delta_parallel=dparallel)
                    iax +=1
            # Save south pole map figure of the current filter 
            if dataset == '2018May':
                plt.savefig(f"{dir}calib_all_south_pole_maps_night_{inight}.png", dpi=150, bbox_inches='tight')
                # plt.savefig(f"{dir}calib_all_south_pole_maps.pdf", dpi=500, bbox_inches='tight')
            else:
                plt.savefig(f"{dir}calib_all_south_pole_maps_night_{inight}.png", dpi=150, bbox_inches='tight')
                # plt.savefig(f"{dir}calib_all_south_pole_maps.pdf", dpi=500, bbox_inches='tight')
                # Clear figure to avoid overlapping between plotting subroutines
                plt.close()


def PlotOnePole(img, filter, vmin, vmax, central_longitude, central_latitude, latitude_limit, \
                number_meridian, number_parrallel, longitude_to_write, \
                delta_meridian, delta_parallel):
    """ Setting routine to make pretty polar projection for a single pole"""
    plt.figure(figsize=(8, 3))
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
    # Adding filter wavelength info on the top right corner,
    # for this, it is needed to project (again) on the Geodetic coordinates
    if central_latitude>0:
        filterx, filtery = ax.projection.transform_point(140, latitude_limit-25., ccrs.Geodetic())
    if central_latitude<0:
        filterx, filtery = ax.projection.transform_point(400, latitude_limit+25., ccrs.Geodetic())
    ax.text(x=filterx, y=filtery, s=f"{filter}"+r" $\mu$m", size=12, fontfamily='sans-serif')
    # Define a colorbar
    # cax = plt.axes([0.72, 0.1, 0.02, 0.8])
    # cbar = plt.colorbar(im, cax=cax, format="%.0f", extend='both', fraction=0.046, pad=0.15)
    # cbar.ax.tick_params(labelsize=12)
    # cbar.locator = ticker.MaxNLocator(nbins=10)
    # cbar.update_ticks()
    # cbar.ax.set_title("[K]", size=12, pad=15)

def CustomLongitudeLabels(axes, clat, lat_lim, num_merid):
    """ Small routine to define the longitude labels of the polar projection """

    # Local variable definition
    degree_symbol = u'\u00B0'               # degree symbol in UTF code
    lond = np.linspace(0,360, num_merid)    # array of longitude range dimension
    latd = np.zeros(len(lond))              # array of latitude with longitude range dimension

    for (alon, alat) in zip(lond, latd):
        if clat>0:
            projx, projy = axes.projection.transform_point(alon, lat_lim-3., ccrs.Geodetic())
        if clat<0:
            projx, projy = axes.projection.transform_point(alon, lat_lim+3., ccrs.Geodetic())
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
            txt = f"{int(360-alon)}"+degree_symbol#+'W'
            axes.text(projx, projy, txt, va=va, ha=ha, color='black',fontsize = 8)
        if (alon==0):
            txt = f"{int(alon)}"+degree_symbol#+'W'
            axes.text(projx, projy, txt, va=va, ha=ha, color='black',fontsize = 8)

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
            axes.text(projx, projy, txt, va='center', ha='center', color='white',fontsize = 6) 
        # Northern hemisphere labelisation
        if (clat > 0 and alat >= 20 and alat < 90):
            projx, projy = axes.projection.transform_point(alon, alat, ccrs.Geodetic())
            txt = f"{int(alat)}"+degree_symbol
            axes.text(projx, projy, txt, va='center', ha='center', color='white',fontsize = 6)

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
