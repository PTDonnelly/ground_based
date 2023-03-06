import numpy as np
import bottleneck as bn
import warnings
import matplotlib.pyplot as plt
import Globals
from Tools.SetWave import SetWaveReduced
from collections import defaultdict

def BinCentreToLimb(mode, nfiles, spectrum, LCMIII):
    
    print('Calculating centre-to-limb profiles...')

    def find_filters(filters):
        # Define a defaultdict that returns an empty list instead of a KeyError if key is absent.
        count = defaultdict(list)
        # Store filter name as dictionary key and index as item
        for i, item in enumerate(filters):
            count[item].append(i)
        # Store filter indices only if filter is present
        return ((key, item) for key, item in count.items() if ~np.isnan(key))

    def singles(nfiles, spectrum, LCMIII):
        """Create central meridian average for each observation"""

        single_ctls = np.zeros((Globals.nlatbins, Globals.nmubins, nfiles, 7))

        # Loop over latitudes and create CTL profiles
        print('Binning CTL profiles...')
        
        for ilat, clat in enumerate(Globals.latgrid):
            # Define centre and edges of latitude bin
            lat1 = Globals.latrange[0] + (Globals.latstep)*ilat
            lat2 = Globals.latrange[0] + (Globals.latstep)*(ilat+1)
            print(f"LAT: {clat}")
            # Loop over emission angles and create CTL profiles
            for imu, cmu in enumerate(Globals.mugrid):
                # Define centre and edges of emission angle bin
                mu1 = Globals.murange[0] + (Globals.mustep)*imu
                mu2 = Globals.murange[0] + (Globals.mustep)*(imu+1)    
                # Loop over the spectrum array of each input file
                for ifile in range(nfiles):
                    # Select lat-lon region around central meridian to calculate average
                    lats = spectrum[:, :, ifile, 0]
                    mus = spectrum[:, :, ifile, 4]
                    rads = spectrum[:, :, ifile, 5]
                    keep = (lats >= lat1) & (lats < lat2) & (mus > mu1) & (mus < mu2)
                    spx = spectrum[keep, ifile, :]
                    # Expect to see RuntimeWarnings in this block
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        # Throw away hemisphere with negative beam
                        view = np.mean(spx[:, 8])
                        if (view == 1) and (lat1 >=-5) or (view == -1) and (lat2 <= 5):
                            # if np.any(spx):
                            # Pull out variables
                            LCM      = bn.nanmean(spx[:, 1])
                            mu       = cmu
                            rad      = bn.nanmean(spx[:, 5])
                            rad_err  = bn.nanmean(spx[:, 6])
                            wavenum  = spx[:, 7][0]
                            # print(clat, cmu, ifile, wavenum, view)
                            view     = spx[:, 8][0]
                            # Store individual CTL profiles
                            single_ctls[ilat, imu, ifile, 0] = clat
                            single_ctls[ilat, imu, ifile, 1] = LCM
                            single_ctls[ilat, imu, ifile, 2] = mu
                            single_ctls[ilat, imu, ifile, 3] = rad
                            single_ctls[ilat, imu, ifile, 4] = rad_err
                            single_ctls[ilat, imu, ifile, 5] = wavenum
                            single_ctls[ilat, imu, ifile, 6] = view

        # Throw away zeros
        single_ctls[single_ctls == 0] = np.nan

        return single_ctls

    def spectrals(nfiles, spectrum, LCMIII, single_ctls):
        """Create central meridian average for each wavelength"""

        # Create np.array for all spectral mean profiles (one per filter)
        spectral_ctls = np.zeros((Globals.nlatbins, Globals.nmubins, Globals.nfilters, 6))

        print('Binning spectrals...')
        # Loop over latitude-emission angle grid and create individual mean profiles for each filter
        for ifilt in range(Globals.nfilters):
            _, _, wave, _, _ = SetWaveReduced(filename=None, wavelength=None, wavenumber=None, ifilt=ifilt)
            for ilat, clat in enumerate(Globals.latgrid):
                for imu, cmu in enumerate(Globals.mugrid):
                    # print(f"LAT: {clat}   MU: {cmu}")
                    filters = single_ctls[ilat, imu, :, 5]
                    keep = (filters == wave)
                    spx = single_ctls[ilat, imu, keep, :]
                    if np.any(spx):
                        # Pull out variables
                        LCM      = bn.nanmean(spx[:, 1])
                        mu       = cmu
                        rad      = bn.nanmean(spx[:, 3])
                        rad_err  = bn.nanmean(spx[:, 4])
                        wavenum  = spx[:, 5][0]
                        # Store spectral CTL profiles
                        spectral_ctls[ilat, imu, ifilt, 0] = clat
                        spectral_ctls[ilat, imu, ifilt, 1] = LCM
                        spectral_ctls[ilat, imu, ifilt, 2] = mu
                        spectral_ctls[ilat, imu, ifilt, 3] = rad
                        spectral_ctls[ilat, imu, ifilt, 4] = rad_err
                        spectral_ctls[ilat, imu, ifilt, 5] = wavenum
        # Throw away zeros
        spectral_ctls[spectral_ctls == 0] = np.nan

        return spectral_ctls

    singles = singles(nfiles, spectrum, LCMIII)
    spectrals = spectrals(nfiles, spectrum, LCMIII, singles)

    return singles, spectrals