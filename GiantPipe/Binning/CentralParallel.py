import numpy as np
import bottleneck as bn
import warnings
import Globals
from Tools.SetWave import SetWave

def BinCentralPara(nfiles, spectrum, LCMIII):
    """ Step 4: Create central parallel average for each observation
        Step 5: Create central parallel average for each wavelength"""
    
    print('Calculating parallel profiles...')

    def singles(nfiles, spectrum, LCMIII):
        """Create central parallel average for each observation"""

        # Create np.array for all individual mean profiles (one per file)
        single_paras = np.zeros((Globals.nlonbins, nfiles, 7))
        # Loop over latitudes and create individual mean profiles
        print('Binning singles...')
        for ilon, _ in enumerate(Globals.longrid):
            # Define centre and edges of latitude bin
            clon = Globals.lonrange[0] - (Globals.lonstep)*ilon - (Globals.lonstep/2)
            lon2 = Globals.lonrange[0] - (Globals.lonstep)*ilon
            lon1 = Globals.lonrange[0] - (Globals.lonstep)*(ilon+1)
            # Loop over the spectrum array of each input file
            for ifile in range(nfiles):
                clat = Globals.LCP
                lat1 = Globals.LCP - Globals.para_width
                lat2 = Globals.LCP + Globals.para_width
                # Select lat-lon region around central parallel to calculate average
                LCM1 = (LCMIII[ifile] - Globals.merid_width)
                LCM2 = (LCMIII[ifile] + Globals.merid_width)
                lats = spectrum[:, :, ifile, 0]
                lons = spectrum[:, :, ifile, 1]
                keep = (lats >= lat1) & (lats < lat2) & (lons > lon1) & (lons <= lon2) & (lons > LCM1) & (lons <= LCM2)
                spx = spectrum[keep, ifile, :]
                # Expect to see RuntimeWarnings in this block
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    # Throw away hemisphere with negative beam
                    view = np.mean(spx[:, 6])
                    if np.any(spx):
                        # Pull out variables
                        clon      = bn.nanmean(spx[:, 1])
                        mu       = bn.nanmin(spx[:, 4])
                        rad      = bn.nanmean(spx[:, 5])
                        rad_err  = bn.nanmean(spx[:, 6])
                        wavenum  = spx[:, 7][0]
                        view     = spx[:, 8][0]
                        # Store individual paraional profiles
                        single_paras[ilon, ifile, 0] = clat
                        single_paras[ilon, ifile, 1] = clon
                        single_paras[ilon, ifile, 2] = mu
                        single_paras[ilon, ifile, 3] = rad
                        single_paras[ilon, ifile, 4] = rad_err
                        single_paras[ilon, ifile, 5] = wavenum
                        # print(ilon, ifile, wavenum)
                        single_paras[ilon, ifile, 6] = view
        # Throw away zeros
        single_paras[single_paras == 0] = np.nan

        return single_paras

    def spectrals(nfiles, spectrum, single_paras):
        """Create central parallel average for each wavelength"""

        # Create np.array for all spectral mean profiles (one per filter)
        spectral_paras = np.zeros((Globals.nlonbins, Globals.nfilters, 6))

        print('Binning spectrals...')
        # Loop over filters and create mean spectral profiles
        for ifilt in range(Globals.nfilters):
            _, _, wave, _, _ = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=ifilt)
            # Loop over latitudes and create individual mean profiles
            for ilon, _ in enumerate(Globals.longrid):
                # Define centre and edges of latitude bin
                clon = Globals.lonrange[0] - (Globals.lonstep)*ilon - (Globals.lonstep/2)
                # Select a filter to calculate average
                filters = single_paras[ilon, :, 5]
                keep = (filters == wave)
                spx = single_paras[ilon, keep, :]
                if np.any(spx):
                    # Pull out variables
                    LCP      = bn.nanmean(spx[:, 0])
                    mu       = bn.nanmin(spx[:, 2])
                    rad      = bn.nanmax(spx[:, 3])
                    rad_err  = bn.nanmean(spx[:, 4])
                    wavenum  = spx[:, 5][0]
                    # Store spectral paraional profiles
                    spectral_paras[ilon, ifilt, 0] = LCP
                    spectral_paras[ilon, ifilt, 1] = clon
                    spectral_paras[ilon, ifilt, 2] = mu
                    spectral_paras[ilon, ifilt, 3] = rad
                    spectral_paras[ilon, ifilt, 4] = rad_err
                    spectral_paras[ilon, ifilt, 5] = wavenum
        # Throw away zeros
        spectral_paras[spectral_paras == 0] = np.nan

        return spectral_paras

    singles = singles(nfiles, spectrum, LCMIII)
    spectrals = spectrals(nfiles, spectrum, singles)

    return singles, spectrals