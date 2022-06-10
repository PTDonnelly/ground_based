import numpy as np

from BinningInputs import BinningInputs
from ReadCal import ReadCal
from SetWave import SetWave
# from VisirWavenumbers import VisirWavenumbers

def CalibrateMeridProfiles(Nfiles, single_merids, spectral_merids, wavenumber):
    """ Step 6: Calibrate spectral_merids to spacecraft data
                (i.e. create calib_spectral_merids)
        Step 7: Calibrate single_merids to calib_spectral_merids
                (i.e. create calib_single_merids)"""

    # Create arrays to store calibration coefficients
    calib_coeff_single   = np.ones((Nfiles, 2))
    calib_coeff_spectral = np.ones((BinningInputs.nfilters, 2))
    
    # Read in Voyager and Cassini data into arrays
    calfile = "../visir.jup.filtered-iris-cirs.10-12-15.data.v3"
    iris, cirs = ReadCal(calfile)

    # Calculate calibration coefficients for the spectral merid profiles
    print('Calibrating spectrals:')
    for iwave in range(BinningInputs.nfilters):
        # Get filter index for calibration file
        waves = spectral_merids[:, iwave, 5]
        wave  = waves[(waves > 0)][0]
        _, _, ifilt_sc, ifilt_v = SetWave(wavelength=False, wavenumber=wave)
        # Calculate averages for calibration
        if ifilt_sc < 12:
            # Establish shared latitudes for accurate averaging
            lmin_visir, lmax_visir = np.nanmin(spectral_merids[:, ifilt_v, 0]), np.nanmax(spectral_merids[:, ifilt_v, 0])
            lmin_calib, lmax_calib = np.nanmin(cirs[:, ifilt_sc, 0]), np.nanmax(cirs[:, ifilt_sc, 0])
            latmin, latmax         = np.max((lmin_visir, lmin_calib, -70)), np.min((lmax_visir, lmax_calib, 70))
            visirkeep              = (spectral_merids[:, ifilt_v, 0] >= latmin) & (spectral_merids[:, ifilt_v, 0] <= latmax)            
            visirdata              = spectral_merids[visirkeep, ifilt_v, 3]
            visirmean              = np.nanmean(spectral_merids[visirkeep, ifilt_v, 3])
            # Use CIRS for N-Band
            calibkeep  = (cirs[:, ifilt_sc, 0] >= latmin) & (cirs[:, ifilt_sc, 0] <= latmax)
            calib      = cirs[:, ifilt_sc, 1]
            calibdata  = cirs[calibkeep, ifilt_sc, 1]
            calibmean  = np.nanmean(calibdata)
        else:
            # Establish shared latitudes for accurate averaging
            lmin_visir, lmax_visir = np.nanmin(spectral_merids[:, ifilt_v, 0]), np.nanmax(spectral_merids[:, ifilt_v, 0])
            lmin_calib, lmax_calib = np.nanmin(iris[:, ifilt_sc, 0]), np.nanmax(iris[:, ifilt_sc, 0])
            latmin, latmax         = np.max((lmin_visir, lmin_calib)), np.min((lmax_visir, lmax_calib))
            visirkeep              = (spectral_merids[:, ifilt_v, 0] >= latmin) & (spectral_merids[:, ifilt_v, 0] <= latmax)            
            visirdata              = spectral_merids[visirkeep, ifilt_v, 3]
            visirmean              = np.nanmean(spectral_merids[visirkeep, ifilt_v, 3])
            # Use IRIS for Q-Band
            calibkeep  = (iris[:, ifilt_sc, 0] >= latmin) & (iris[:, ifilt_sc, 0] <= latmax)
            calib      = iris[:, ifilt_sc, 1]
            calibdata  = iris[calibkeep, ifilt_sc, 1]
            calibmean  = np.nanmean(calibdata)
        # Do calibration
        calib_coeff_spectral[iwave, 0] = wave
        calib_coeff_spectral[iwave, 1] = visirmean / calibmean
        # print(ifilt_sc, visirmean, calibmean, calib_coeff_spectral[iwave, 1])

    # Calculate calibration coefficients for the single merid profiles
    print('Calibrating singles:')
    for ifile, wave in enumerate(wavenumber):
        # Get filter index for spectral profiles
        _, _, ifilt_sc, ifilt_v = SetWave(wavelength=False, wavenumber=wave)
        # Establish shared latitudes for accurate averaging
        lmin_single, lmax_single       = np.nanmin(single_merids[:, ifile, 0]), np.nanmax(single_merids[:, ifile, 0])
        lmin_spectral, lmax_spectral   = np.nanmin(spectral_merids[:, ifilt_v, 0]), np.nanmax(spectral_merids[:, ifilt_v, 0])
        latmin, latmax                 = np.max((lmin_single, lmin_spectral)), np.min((lmax_single, lmax_spectral))
        singlekeep                     = (single_merids[:, ifile, 0] >= latmin) & (single_merids[:, ifile, 0] <= latmax)            
        singledata                     = single_merids[singlekeep, ifile, 3]
        singlemean                     = np.nanmean(singledata)
        spectralkeep                   = (spectral_merids[:, ifilt_v, 0] >= latmin) & (spectral_merids[:, ifilt_v, 0] <= latmax)
        spectraldata                   = spectral_merids[spectralkeep, ifilt_v, 3]
        spectralmean                   = np.nanmean(spectraldata)
        calib_coeff_single[ifile, 0]   = ifile
        calib_coeff_single[ifile, 1]   = singlemean / spectralmean

        # print(ifile, singlemean, spectralmean, calib_coeff_single[ifile, 1])

    # Save calibration
    for ifile in range(Nfiles):
        # Calibrate individual merid profiles using individual calibration coefficients
        calib_single_merids = single_merids
        calib_single_merids[:, ifile, 3] /= calib_coeff_single[ifile, 1]
        calib_single_merids[:, ifile, 4] /= calib_coeff_single[ifile, 1]
    for ifilt in range(BinningInputs.nfilters):
        # Calibrate spectral merid profiles using spectral calibration coefficients
        calib_spectral_merids = spectral_merids
        calib_spectral_merids[:, ifilt, 3] /= calib_coeff_spectral[ifilt, 1]
        calib_spectral_merids[:, ifilt, 4] /= calib_coeff_spectral[ifilt, 1]

    # # Clear single_merids and spectral_merids arrays from from local variables
    # del locals()['single_merids']
    # del locals()['spectral_merids']

    return calib_single_merids, calib_spectral_merids, calib_coeff_single, calib_coeff_spectral