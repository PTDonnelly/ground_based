import os
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import Globals
from Tools.CorrectMaps import PolynomialAdjust, ApplyPolynom, BlackLineRemoving
from Tools.SetWave import SetWave
from Tools.VisirFilterInfo import Wavenumbers

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
        mumin = [0.02, 0.02, 0.1, 0.1, 0.01, 0.05, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01]
        Nfilters = Globals.nfilters
    else:
        cmaps, mumaps, wavenumber = BlackLineRemoving(dir, files, cblack=-60)
        mumin = np.empty(13)
        mumin.fill(0.01)
        Nfilters = 10

    print('Mapping global maps...')

    for ifilt in range(Nfilters):
        # Empty local TBmaps array to avoid overlapping between filter
        TBmaps[:, :, :] = np.nan
        # Get filter index for spectral profiles
        waves = spectrals[:, ifilt, 5]
        if waves[(waves > 0)] != []:
            wave  = waves[(waves > 0)][0]
            _, _, _, ifilt = SetWave(wavelength=False, wavenumber=wave)
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
            im = plt.imshow(globalmaps[ifilt, :, :], origin='lower', vmin=min, vmax=max, cmap='inferno')
            plt.xticks(np.arange(0, Globals.nx+1,  step = 60), list(np.arange(360,-1,-30)))
            plt.yticks(np.arange(0, Globals.ny+1, step = 60), list(np.arange(-90,91,30)))
            plt.xlabel('System III West Longitude')
            plt.ylabel('Planetocentric Latitude')
            #plt.tick_params(labelsize=15)
            cbar = plt.colorbar(im, extend='both', fraction=0.046, pad=0.04)
            #cbar.ax.tick_params(labelsize=15)
            cbar.set_label("Brightness Temperature [K]")

            # Save global map figure of the current filter 
            filt = Wavenumbers(ifilt)
            if dataset == '2018May':
                plt.savefig(f"{dir}calib_{filt}_global_maps_{adj_location}_adj.png", dpi=300)
                plt.savefig(f"{dir}calib_{filt}_global_maps_{adj_location}_adj.eps", dpi=300)
                # Clear figure to avoid overlapping between plotting subroutines
                plt.clf()
                # Write global maps to np.array
                np.save(f"{dir}calib_{filt}_global_maps_{adj_location}_adj", globalmaps[ifilt, :, :])
                # Write global maps to txtfiles
                np.savetxt(f"{dir}calib_{filt}_global_maps_{adj_location}_adj.txt", globalmaps[ifilt, :, :])
                # Write global maps to NetCDF files
                #GlobalMapsNetCDF(dir, filt, globalmaps=globalmaps[ifilt, :, :])
            else:
                plt.savefig(f"{dir}calib_{filt}_global_maps.png", dpi=300)
                plt.savefig(f"{dir}calib_{filt}_global_maps.eps", dpi=300)
                # Clear figure to avoid overlapping between plotting subroutines
                plt.clf()
                # Write global maps to np.array
                np.save(f"{dir}calib_{filt}_global_maps", globalmaps[ifilt, :, :])
                # Write global maps to txtfiles
                np.savetxt(f"{dir}calib_{filt}_global_maps.txt", globalmaps[ifilt, :, :])
                # Write global maps to NetCDF files
                #GlobalMapsNetCDF(dir, filt, globalmaps=globalmaps[ifilt, :, :])
                

def GlobalMapsNetCDF(dir, filt, globalmaps):
    import netCDF4 as nc
    import numpy as np


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


