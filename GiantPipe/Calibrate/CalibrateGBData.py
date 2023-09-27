from FindFiles import FindFiles
from RegisterMaps import RegisterMaps
from Binning.CentralMerid import BinCentralMerid
from Calibrate.CentralMerid import CalCentralMerid
from Calibrate.CylindricalMaps import CalCylindricalMaps
from Plot.PlotProfiles import PlotMeridProfiles
import Globals
import numpy as np
from Tools import SetWave
import os


def CalibrateGBData(dataset, mode):


    # Point to location of observations
    files = FindFiles(dataset=dataset, mode=mode)
    nfiles = len(files)

    # Steps 1-3: Generate arrays containing spatial and spectral information of each cylindrical map
    spectrum, wavelength, wavenumber, LCMIII, DATE = RegisterMaps(dataset=dataset, files=files, binning="bin_cmerid")

    # Steps 4-5: Generate average meridional profiles for each observation and each filter
    rawsingles, rawspectrals = BinCentralMerid(nfiles=nfiles, spectrum=spectrum, LCMIII=LCMIII)
    
    # Steps 6-7: Generate calibrated versions of the profiles from Steps 4 and 5
    singles, spectrals, ksingles, kspectrals = CalCentralMerid(mode=mode, files=files, singles=rawsingles, spectrals=rawspectrals)

    # Step 8: Calibrate cylindrical maps using calculated calibration coefficients
    CalCylindricalMaps(files=files, ksingles=ksingles, kspectrals=kspectrals)

    print("Saving calibration coefficients ...")
    # If subdirectory does not exist, create it
    dir = f'../outputs/{dataset}/spectral_coeff_profiles/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    for ifilt in range(Globals.nfilters):
        # Write spectral mean profiles and calibrated coefficient to np.array
        _, _, wavenb, _, _ = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=ifilt)
       
        np.save(f"{dir}{wavenb}_calib_coeff", kspectrals[ifilt, :])
        
        np.savetxt(f"{dir}{wavenb}_calib_coeff.txt", kspectrals[ifilt, :],
                    fmt=['%8.5f'],
                    header='NU      CALIB_COEFF')