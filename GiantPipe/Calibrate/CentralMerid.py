import numpy as np
import Globals
from Read.ReadCal import ReadCal
from Tools.SetWave import SetWave

def CalCentralMerid(mode, files, singles, spectrals):
    """ Step 6: Calibrate spectrals to spacecraft data
                (i.e. create calib_spectrals)
        Step 7: Calibrate singles to calib_spectrals
                (i.e. create calib_singles)"""

    print('Calibrating meridional profiles...')
    
    # Create arrays to store calibration coefficients
    nfiles = len(files)
    calib_coeff_single   = np.ones((nfiles, 2))
    calib_coeff_spectral = np.ones((Globals.nfilters, 2))
    
    # Read in Voyager and Cassini data into arrays
    calfile = "../inputs/visir.jup.filtered-iris-cirs.10-12-15.data.v3"
    iris, cirs = ReadCal(calfile)

    # Calculate calibration coefficients for the spectral merid profiles
    print('Calibrating spectrals...')
    for iwave in range(Globals.nfilters):
        waves = spectrals[:, iwave, 5]
        if waves[(waves > 0)] != []:
            # Get filter index for calibration file
            filter_name, _, wave, ifilt_sc, ifilt_v = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=iwave)

            # Calculate averages for calibration
            if ifilt_sc < 12:
                # Establish shared latitudes for accurate averaging
                lmin_visir, lmax_visir = np.nanmin(spectrals[:, ifilt_v, 0]), np.nanmax(spectrals[:, ifilt_v, 0])
                if filter_name == 'J7.9':
                    lmin_visir, lmax_visir = -5, 5

                lmin_calib, lmax_calib = np.nanmin(cirs[:, ifilt_sc, 0]), np.nanmax(cirs[:, ifilt_sc, 0])
                latmin, latmax         = np.max((lmin_visir, lmin_calib, -70)), np.min((lmax_visir, lmax_calib, 70))
                visirkeep              = (spectrals[:, ifilt_v, 0] >= latmin) & (spectrals[:, ifilt_v, 0] <= latmax)            
                visirdata              = spectrals[visirkeep, ifilt_v, 3]
                visirmean              = np.nanmean(spectrals[visirkeep, ifilt_v, 3])
                # Use CIRS for N-Band
                calibkeep  = (cirs[:, ifilt_sc, 0] >= latmin) & (cirs[:, ifilt_sc, 0] <= latmax)
                calib      = cirs[:, ifilt_sc, 1]
                calibdata  = cirs[calibkeep, ifilt_sc, 1]
                calibmean  = np.nanmean(calibdata)
            else:
                # Establish shared latitudes for accurate averaging
                lmin_visir, lmax_visir = np.nanmin(spectrals[:, ifilt_v, 0]), np.nanmax(spectrals[:, ifilt_v, 0])
                lmin_calib, lmax_calib = np.nanmin(iris[:, ifilt_sc, 0]), np.nanmax(iris[:, ifilt_sc, 0])
                latmin, latmax         = np.max((lmin_visir, lmin_calib, -70)), np.min((lmax_visir, lmax_calib, 70))

                visirkeep              = (spectrals[:, ifilt_v, 0] >= latmin) & (spectrals[:, ifilt_v, 0] <= latmax)            
                visirdata              = spectrals[visirkeep, ifilt_v, 3]
                visirmean              = np.nanmean(spectrals[visirkeep, ifilt_v, 3])
                # Use IRIS for Q-Band
                calibkeep  = (iris[:, ifilt_sc, 0] >= latmin) & (iris[:, ifilt_sc, 0] <= latmax)
                calib      = iris[:, ifilt_sc, 1]
                calibdata  = iris[calibkeep, ifilt_sc, 1]
                calibmean  = np.nanmean(calibdata)
            # Do calibration
            calib_coeff_spectral[iwave, 0] = wave
            calib_coeff_spectral[iwave, 1] = visirmean / calibmean
            print(visirmean,calibmean)
            print(calib_coeff_spectral[iwave, 1])

    # Calculate calibration coefficients for the single merid profiles
    print('Calibrating singles...')
    for ifile, fname in enumerate(files):
        # Get filter index for spectral profiles
        _, _, wave, _, ifilt_v = SetWave(filename=fname, wavelength=None, wavenumber=None, ifilt=None)
        # Establish shared latitudes for accurate averaging
        lmin_single, lmax_single       = np.nanmin(singles[:, ifile, 0]), np.nanmax(singles[:, ifile, 0])
        lmin_spectral, lmax_spectral   = np.nanmin(spectrals[:, ifilt_v, 0]), np.nanmax(spectrals[:, ifilt_v, 0])
        latmin, latmax                 = np.max((lmin_single, lmin_spectral)), np.min((lmax_single, lmax_spectral))
        singlekeep                     = (singles[:, ifile, 0] >= latmin) & (singles[:, ifile, 0] <= latmax)            
        singledata                     = singles[singlekeep, ifile, 3]
        singlemean                     = np.nanmean(singledata)
        spectralkeep                   = (spectrals[:, ifilt_v, 0] >= latmin) & (spectrals[:, ifilt_v, 0] <= latmax)
        spectraldata                   = spectrals[spectralkeep, ifilt_v, 3]
        spectralmean                   = np.nanmean(spectraldata)
        calib_coeff_single[ifile, 0]   = ifile
        calib_coeff_single[ifile, 1]   = singlemean / spectralmean
        
    # Save calibration
    for ifile in range(nfiles):
        # Calibrate individual merid profiles using individual calibration coefficients
        calib_singles = singles
        calib_singles[:, ifile, 3] /= calib_coeff_single[ifile, 1]
        calib_singles[:, ifile, 4] /= calib_coeff_single[ifile, 1]
    for ifilt in range(Globals.nfilters):
        # Calibrate spectral merid profiles using spectral calibration coefficients
        calib_spectrals = spectrals
        calib_spectrals[:, ifilt, 3] /= calib_coeff_spectral[ifilt, 1]
        calib_spectrals[:, ifilt, 4] /= calib_coeff_spectral[ifilt, 1]

    return calib_singles, calib_spectrals, calib_coeff_single, calib_coeff_spectral