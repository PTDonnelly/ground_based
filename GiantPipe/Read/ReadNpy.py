import numpy as np
from FindFiles import FindFiles

def ReadCentralMeridNpy(dataset, mode, return_singles, return_spectrals):

    """Read in pre-calculated profiles and/or coefficients from local .npy files"""

    if return_singles:
        # Point to stored meridional profiles and calibration coefficients
        profiles  = FindFiles(dataset=dataset, mode=mode+'_singles')

        # Load .npy files
        singles    = np.asarray([np.load(p) for p in profiles])
        # Fix shape (numpy changes array shape when storing)
        singles = np.flip(np.rollaxis(singles, 1), 1)
    else:
        singles = None
    
    if return_spectrals:
        # Point to stored meridional profiles and calibration coefficients
        profiles  = FindFiles(dataset=dataset, mode=mode+'_spectrals')

        # Load .npy files
        spectrals    = np.asarray([np.load(p) for p in profiles])
        # Fix shape (numpy changes array shape when storing)
        spectrals = np.flip(np.rollaxis(spectrals, 1), 1)
    else:
        spectrals = None

    return singles, spectrals

def ReadCentralParallelNpy(dataset, mode, return_singles, return_spectrals):
    """Read in pre-calculated profiles and/or coefficients from local .npy files"""

    if return_singles:
        # Point to stored parallel profiles and calibration coefficients
        profiles  = FindFiles(dataset=dataset, mode=mode+'_singles')
        # Load .npy files
        singles    = np.asarray([np.load(p) for p in profiles])
        # Fix shape (numpy changes array shape when storing)
        singles = np.flip(np.rollaxis(singles, 1), 1)
    else:
        singles = None
    
    if return_spectrals:
        # Point to stored parallel profiles and calibration coefficients
        profiles  = FindFiles(dataset=dataset, mode=mode+'_spectrals')
        # Load .npy files
        spectrals    = np.asarray([np.load(p) for p in profiles])
        # Fix shape (numpy changes array shape when storing)
        spectrals = np.flip(np.rollaxis(spectrals, 1), 1)
    else:
        spectrals = None

    return singles, spectrals

def ReadCentreToLimbNpy(mode, return_singles, return_spectrals):
    a=1

def ReadRegionalNpy(dataset, mode, return_singles, return_spectrals):
    """Read in pre-calculated maps and/or coefficients from local .npy files"""

    if return_singles:
        # Point to stored maps and calibration coefficients
        profiles  = FindFiles(dataset=dataset, mode=mode+'_singles')
        # Load .npy files
        singles    = np.asarray([np.load(p) for p in profiles])
        # Fix shape (numpy changes array shape when storing)
        singles = np.flip(np.rollaxis(singles, 1), 1)
    else:
        singles = None
    
    if return_spectrals:
        # Point to stored maps and calibration coefficients
        profiles  = FindFiles(dataset=dataset, mode=mode+'_spectrals')
        # Load .npy files
        spectrals    = np.asarray([np.load(p) for p in profiles])
        # Fix shape (numpy changes array shape when storing)
        spectrals = np.flip(np.rollaxis(spectrals, 1), 1)
    else:
        spectrals = None

    return singles, spectrals

def ReadRegionalAverageNpy(dataset, mode, return_singles, return_spectrals):
    a=1
