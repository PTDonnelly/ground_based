import numpy as np
import bottleneck as bn
import warnings
import Globals
from Tools.SetWave import SetWave

def BinRegional(nfiles, spectrum, LCMIII):
    """ Step 4: Create 2D aera average for each observation
        Step 5: Create 2D aera average for each wavelength"""
    
    print('Calculating 2D spectral maps...')

    def singles(nfiles, spectrum, LCMIII):
        """Create 2D aera average for each observation"""

        # Create np.array for all individual mean profiles (one per file)
        single_regions = np.zeros((Globals.nlatbins, Globals.nlonbins, nfiles, 7))
        # Loop over latitudes and create individual mean profiles
        print('Binning singles...')
        # Select lat-lon region around 2D aera to calculate average
        lat_target1 = Globals.lat_target - Globals.para_width
        lat_target2 = Globals.lat_target + Globals.para_width
        lon_target1 = (Globals.lon_target - Globals.merid_width)
        lon_target2 = (Globals.lon_target + Globals.merid_width)
        for ilat, _ in enumerate(Globals.latgrid):
            # print(ilat)
            # Define centre and edges of latitude bin
            clat = Globals.latrange[0] + (Globals.latstep)*ilat + (Globals.latstep/2)
            lat1 = Globals.latrange[0] + (Globals.latstep)*ilat
            lat2 = Globals.latrange[0] + (Globals.latstep)*(ilat+1)
            # Check if the current ilat binning is contained in the target boxe
            if (lat1 >= lat_target1) & (lat2 <= lat_target2):
                for ilon, _ in enumerate(Globals.longrid):
                    # Define centre and edges of longitude bin
                    clon = Globals.lonrange[0] - (Globals.lonstep)*ilon - (Globals.lonstep/2)
                    lon2 = Globals.lonrange[0] - (Globals.lonstep)*ilon
                    lon1 = Globals.lonrange[0] - (Globals.lonstep)*(ilon+1)
                    # Check if the current ilon binning is contained in the target boxe
                    if (lon1 >= lon_target1) & (lon2 <= lon_target2):
                    # Loop over the spectrum array of each input file if ilat,ilon pixel is in the target boxe
                        for ifile in range(nfiles):
                            lats = spectrum[:, :, ifile, 0]
                            lons = spectrum[:, :, ifile, 1]
                            keep = (lats >= lat1) & (lats < lat2) & (lats >= lat_target1) & (lats < lat_target2) \
                                 & (lons > lon1) & (lons <= lon2) & (lons > lon_target1) & (lons <= lon_target2)
                            spx = spectrum[keep, ifile, :]
                            # Expect to see RuntimeWarnings in this block
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore", category=RuntimeWarning)
                                # Throw away hemisphere with negative beam
                                view = np.mean(spx[:, 6])
                                if np.any(spx):
                                    # Pull out variables
                                    mu       = bn.nanmin(spx[:, 4])
                                    rad      = bn.nanmean(spx[:, 5])
                                    rad_err  = bn.nanmean(spx[:, 6])
                                    wavenum  = spx[:, 7][0]
                                    view     = spx[:, 8][0]
                                    # Store individual paraional profiles
                                    single_regions[ilat, ilon, ifile, 0] = clat
                                    single_regions[ilat, ilon, ifile, 1] = clon
                                    single_regions[ilat, ilon, ifile, 2] = mu
                                    single_regions[ilat, ilon, ifile, 3] = rad
                                    single_regions[ilat, ilon, ifile, 4] = rad_err
                                    single_regions[ilat, ilon, ifile, 5] = wavenum
                                    # print(ilon, ifile, wavenum)
                                    single_regions[ilat, ilon, ifile, 6] = view
        # Throw away zeros
        single_regions[single_regions == 0] = np.nan

        return single_regions

    def spectrals(nfiles, spectrum, single_regions):
        """Create 2D aera average for each wavelength"""

        # Create np.array for all spectral mean profiles (one per filter)
        spectral_regions = np.zeros((Globals.nlatbins, Globals.nlonbins, Globals.nfilters, 6))

        print('Binning spectrals...')
        # Loop over filters and create mean spectral profiles
        for ifilt in range(Globals.nfilters):
            _, _, wave, _, _ = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=ifilt)
            # Loop over latitudes and create individual mean profiles
            for ilat, _ in enumerate(Globals.latgrid):
                # Define centre and edges of latitude bin
                clat = Globals.latrange[0] + (Globals.latstep)*ilat + (Globals.latstep/2)
                # Loop over longitudes and create individual mean profiles
                for ilon, _ in enumerate(Globals.longrid):
                    # Define centre and edges of longitude bin
                    clon = Globals.lonrange[0] - (Globals.lonstep)*ilon - (Globals.lonstep/2)
                    # Select a filter to calculate average
                    filters = single_regions[ilat, ilon, :, 5]
                    keep = (filters == wave)
                    spx = single_regions[ilat, ilon, keep, :]
                    if np.any(spx):
                        # Pull out variables
                        mu       = bn.nanmin(spx[:, 2])
                        rad      = bn.nanmax(spx[:, 3])
                        rad_err  = bn.nanmean(spx[:, 4])
                        wavenum  = spx[:, 5][0]
                        # Store spectral paraional profiles
                        spectral_regions[ilat, ilon, ifilt, 0] = clat
                        spectral_regions[ilat, ilon, ifilt, 1] = clon
                        spectral_regions[ilat, ilon, ifilt, 2] = mu
                        spectral_regions[ilat, ilon, ifilt, 3] = rad
                        spectral_regions[ilat, ilon, ifilt, 4] = rad_err
                        spectral_regions[ilat, ilon, ifilt, 5] = wavenum
                        # print(spectral_regions[ilat, ilon, ifilt, 3])
        # Throw away zeros
        spectral_regions[spectral_regions == 0] = np.nan

        return spectral_regions

    singles = singles(nfiles, spectrum, LCMIII)
    spectrals = spectrals(nfiles, spectrum, singles)

    return singles, spectrals