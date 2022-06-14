import numpy as np
from FindFiles import FindFiles

def ReadNpy():
    """Read in pre-calculated profiles and/or coefficients from local .npy files"""

    # Point to stored meridional profiles and calibration coefficients
    profiles1  = FindFiles(mode='singles')
    profiles2  = FindFiles(mode='spectrals')
    coeffs1    = FindFiles(mode='ksingles')
    coeffs2    = FindFiles(mode='kspectrals')

    # Load .npy files
    singles    = [np.load(p) for p in profiles1]
    spectrals  = [np.load(p) for p in profiles2]
    ksingles   = [np.load(c) for c in coeffs1]
    kspectrals = [np.load(c) for c in coeffs1]
    
    # Fix shape (numpy changes array shape when storing)
    layers, rows, cols = np.shape(singles)
    singles = np.reshape(singles, (rows, layers, cols))
    layers, rows, cols = np.shape(spectrals)
    spectrals = np.reshape(spectrals, (rows, layers, cols))
    layers, rows, cols = np.shape(ksingles)
    ksingles = np.reshape(ksingles, (rows, layers, cols))
    layers, rows, cols = np.shape(kspectrals)
    kspectrals = np.reshape(kspectrals, (rows, layers, cols))

    return singles, spectrals, ksingles, kspectrals