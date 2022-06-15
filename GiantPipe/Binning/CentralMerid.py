import numpy as np
import Globals
from Tools.VisirFilterInfo import Wavenumbers

def BinCentralMerid(Nfiles, spectrum, LCMIII):
    """ Step 4: Create central meridian average for each observation
        Step 5: Create central meridian average for each wavelength"""
    
    print('Calculating meridional profiles...')

    def singles(Nfiles, spectrum, LCMIII):
        """Create central meridian average for each observation"""

        # Create np.array for all individual mean profiles (one per file)
        single_merids = np.zeros((Globals.nlatbins, Nfiles, 7))

        # Loop over latitudes and create individual mean profiles
        print('Binning singles...')
        for ilat, _ in enumerate(Globals.latgrid):
            # Define centre and edges of latitude bin
            clat = Globals.latrange[0] + (Globals.latstep)*ilat + (Globals.latstep/2)
            lat1 = Globals.latrange[0] + (Globals.latstep)*ilat
            lat2 = Globals.latrange[0] + (Globals.latstep)*(ilat+1)
            # Loop over the spectrum array of each input file
            for ifile in range(Nfiles):
                clon = LCMIII[ifile]
                lon1 = LCMIII[ifile] + Globals.merid_width
                lon2 = LCMIII[ifile] - Globals.merid_width
                # Select lat-lon region around central meridian to calculate average
                lats = spectrum[:, :, ifile, 0]
                lons = spectrum[:, :, ifile, 1]
                keep = (lats >= lat1) & (lats < lat2) & (lons < lon1) & (lons > lon2)
                spx = spectrum[keep, ifile, :]
                # Throw away hemisphere with negative beam
                view = np.mean(spx[:, 6])
                if (view == 1) and (lat1 >=-5) or (view == -1) and (lat1 <= 5):
                    if np.any(spx):
                        # Pull out variables
                        LCM      = np.nanmean(spx[:, 1])
                        mu       = np.nanmin(spx[:, 2])
                        rad      = np.nanmean(spx[:, 3])
                        rad_err  = np.nanmean(spx[:, 4])
                        wavenum  = spx[:, 5][0]
                        view     = spx[:, 6][0]
                        # Store individual meridional profiles
                        single_merids[ilat, ifile, 0] = clat
                        single_merids[ilat, ifile, 1] = LCM
                        single_merids[ilat, ifile, 2] = mu
                        single_merids[ilat, ifile, 3] = rad
                        single_merids[ilat, ifile, 4] = rad_err
                        single_merids[ilat, ifile, 5] = wavenum
                        single_merids[ilat, ifile, 6] = view
        # Throw away zeros
        single_merids[single_merids == 0] = np.nan

        return single_merids

    def spectrals(Nfiles, spectrum, LCMIII, single_merids):
        """Create central meridian average for each wavelength"""

        # Create np.array for all spectral mean profiles (one per filter)
        spectral_merids = np.zeros((Globals.nlatbins, Globals.nfilters, 6))

        print('Binning spectrals...')
        # Loop over filters and create mean spectral profiles
        for ifilt in range(Globals.nfilters):
            # Loop over latitudes and create individual mean profiles
            for ilat, _ in enumerate(Globals.latgrid):
                # Define centre and edges of latitude bin
                clat = Globals.latrange[0] + (Globals.latstep)*ilat + (Globals.latstep/2)
                lat1 = Globals.latrange[0] + (Globals.latstep)*ilat
                lat2 = Globals.latrange[0] + (Globals.latstep)*(ilat+1)
                # Select a filter to calculate average
                wave = VisirWavenumbers(ifilt)
                filters = single_merids[ilat, :, 5]
                keep = (filters == wave)
                spx = single_merids[ilat, keep, :]
                if np.any(spx):
                    # Pull out variables
                    LCM      = np.nanmean(spx[:, 1])
                    mu       = np.nanmin(spx[:, 2])
                    rad      = np.nanmean(spx[:, 3])
                    rad_err  = np.nanmean(spx[:, 4])
                    wavenum  = spx[:, 5][0]
                    # Store spectral meridional profiles
                    spectral_merids[ilat, ifilt, 0] = clat
                    spectral_merids[ilat, ifilt, 1] = LCM
                    spectral_merids[ilat, ifilt, 2] = mu
                    spectral_merids[ilat, ifilt, 3] = rad
                    spectral_merids[ilat, ifilt, 4] = rad_err
                    spectral_merids[ilat, ifilt, 5] = wavenum
        # Throw away zeros
        spectral_merids[spectral_merids == 0] = np.nan

        return spectral_merids

    singles = singles(Nfiles, spectrum, LCMIII)
    spectrals = spectrals(Nfiles, spectrum, LCMIII, singles)

    # # Clear spectrum array from local variables
    # del locals()['spectrum']

    return singles, spectrals