import numpy as np
from FindFiles import FindFiles

def ReadNpy(return_singles, return_spectrals, return_ksingles, return_kspectrals):
    """Read in pre-calculated profiles and/or coefficients from local .npy files"""

    if return_singles:
        # Point to stored meridional profiles and calibration coefficients
        profiles  = FindFiles(mode='singles')
        # Load .npy files
        singles    = np.asarray([np.load(p) for p in profiles])
        # Fix shape (numpy changes array shape when storing)
        singles = np.flip(np.rollaxis(singles, 1), 1)
    else:
        singles = None
    
    if return_spectrals:
        # Point to stored meridional profiles and calibration coefficients
        profiles  = FindFiles(mode='spectrals')
        # Load .npy files
        spectrals    = np.asarray([np.load(p) for p in profiles])
        # Fix shape (numpy changes array shape when storing)
        spectrals = np.flip(np.rollaxis(spectrals, 1), 1)
    else:
        spectrals = None
    
    if return_ksingles:
        # Point to stored meridional profiles and calibration coefficients
        coeffs  = FindFiles(mode='ksingles')
        # Load .npy files (no need to fix this 2-d array)
        ksingles    = np.asarray([np.load(p) for p in coeffs])
    else:
        ksingles = None
    
    if return_kspectrals:
        # Point to stored meridional profiles and calibration coefficients
        coeffs  = FindFiles(mode='kspectrals')
        # Load .npy files (no need to fix this 2-d array)
        kspectrals    = np.asarray([np.load(c) for c in coeffs])
    else:
        kspectrals = None

    return singles, spectrals, ksingles, kspectrals