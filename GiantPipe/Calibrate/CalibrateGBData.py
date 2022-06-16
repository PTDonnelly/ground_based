from FindFiles import FindFiles
from RegisterMaps import RegisterMaps
from Binning.CentralMerid import BinCentralMerid
from Calibrate.CentralMerid import CalCentralMerid
from Calibrate.CylindricalMaps import CalCylindricalMaps

def CalibrateGBData(mode=mode):

    # Point to location of observations
    files = FindFiles(mode=mode)
    nfiles = len(files)

    # Steps 1-3: Generate arrays containing spatial and spectral information of each cylindrical map
    spectrum, wavelength, wavenumber, LCMIII = RegisterMaps(mode=mode, files=files)

    # Steps 4-5: Generate average meridional profiles for each observation and each filter
    rawsingles, rawspectrals = BinCentralMerid(mode=mode, nfiles=nfiles, spectrum=spectrum, LCMIII=LCMIII)
    
    # Steps 6-7: Generate calibrated versions of the profiles from Steps 4 and 5
    singles, spectrals, ksingles, kspectrals = CalCentralMerid(mode=mode, nfiles=nfiles, singles=rawsingles, spectrals=rawspectrals, wavenumber=wavenumber)

    # Step 8: Calibrate cylindrical maps using calculated calibration coefficients
    CalCylindricalMaps(files, ksingles)