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
        print('Binning singles...')
        # Loop over the spectrum array of each input file
        for ifile in range(nfiles):
            # Condition to avoid to take into account the correct hemisphere 
            if np.sign(Globals.LCP) == np.sign(np.nanmean(spectrum[:, :, ifile, 8])):
                # # Store the lon-lat coordinates of the current file in intermediate (and simpler) variables
                lats = spectrum[:, :, ifile, 0]
                lons = spectrum[:, :, ifile, 1]
                # Select lat-lon region around central parallel to calculate average
                clat = Globals.LCP
                lat1 = Globals.LCP - Globals.para_width 
                lat2 = Globals.LCP + Globals.para_width 
                # Set the window limits
                LCM1 = (LCMIII[ifile] - Globals.merid_width)
                LCM2 = (LCMIII[ifile] + Globals.merid_width)
                # Loop over longitudes and create individual mean profiles
                for ilon, _ in enumerate(Globals.longrid):
                    # Define centre and edges of longitude bin
                    clon = Globals.lonrange[0] - (Globals.lonstep)*ilon - (Globals.lonstep/2)
                    lon2 = Globals.lonrange[0] - (Globals.lonstep)*ilon
                    lon1 = Globals.lonrange[0] - (Globals.lonstep)*(ilon+1)
                    # Define the base conditions for lats and lons
                    box_edges = (lats >= lat1) & (lats < lat2) & (lons > lon1) & (lons <= lon2)
                    # When the cmaps is splitted in two parts, we have to ensure to take into account these both parts
                    if LCM2 >= 360:
                        adjusted_LCM2 = LCM2 - 360 if ilon >= 180 else LCM2
                        lon_keep = (lons <= adjusted_LCM2) if ilon >= 180 else (lons >= LCM1)
                        print('+360', LCMIII[ifile], LCM1, LCM2, adjusted_LCM2, lon1, lon2, ilon, lons[box_edges & lon_keep])
                    elif LCM1 <= 0:
                        adjusted_LCM1 = LCM1 + 360 if ilon < 180 else LCM1
                        lon_keep = (lons <= LCM2) if ilon >= 180 else (lons >= adjusted_LCM1)
                        print('-0', LCMIII[ifile], LCM1, adjusted_LCM1, LCM2, lon1, lon2, ilon, lons[box_edges & lon_keep])   
                    else:
                        lon_keep = (lons > LCM1) & (lons <= LCM2)
                    # Filter the spectrum array using the determined conditions
                    spx = spectrum[box_edges & lon_keep, ifile, :]
                    # Save the existing values
                    if np.any(spx):
                        # Pull out variables
                        clon     = bn.nanmean(spx[:, 1])
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