from FindFiles import FindFiles
from RegisterMaps import RegisterMaps
from Binning.CentralMerid import BinCentralMerid
from Calibrate.CentralMerid import CalCentralMerid
from Calibrate.CylindricalMaps import CalCylindricalMaps
from Plot.PlotProfiles import PlotMeridProfiles

def CalibrateGBData(dataset, mode):

    # Point to location of observations
    files = FindFiles(dataset=dataset, mode=mode)
    nfiles = len(files)

    # Steps 1-3: Generate arrays containing spatial and spectral information of each cylindrical map
    spectrum, wavelength, wavenumber, LCMIII = RegisterMaps(files=files)

    # Steps 4-5: Generate average meridional profiles for each observation and each filter
    rawsingles, rawspectrals = BinCentralMerid(nfiles=nfiles, spectrum=spectrum, LCMIII=LCMIII)
    
    # Steps 6-7: Generate calibrated versions of the profiles from Steps 4 and 5
    singles, spectrals, ksingles, kspectrals = CalCentralMerid(mode=mode, files=files, singles=rawsingles, spectrals=rawspectrals)

    # Step 8: Calibrate cylindrical maps using calculated calibration coefficients
    CalCylindricalMaps(files=files, ksingles=ksingles, kspectrals=kspectrals)