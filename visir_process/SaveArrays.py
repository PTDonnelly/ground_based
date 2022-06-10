import os 
import numpy as np 

from BinningInputs import BinningInputs
from VisirWavenumbers import VisirWavenumbers

def WriteProfiles(files, singles, spectrals, ksingles, kspectrals):
    """Save calibrated profiles as numpy arrays and textfiles"""
    
    # If subdirectory does not exist, create it
    dir = 'single_merid_profiles/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Save individual meridional profiles
    for ifile, fname in enumerate(files):
        # Extract filename
        name = fname.split('.fits.gz')
        name = name[0].split('/')
        # Write individual mean profiles to np.array
        np.save(f"{dir}{name[-1]}_merid_profile", singles[:, ifile, :])
        # Write individual mean profiles to textfile
        np.savetxt(f"{dir}{name[-1]}_merid_profile.txt", singles[:, ifile, :],
                    fmt=['%4.2f', '%5.2f', '%4.2f', '%8.5e', '%8.5e', '%8.5f', '%s'],
                    header='LAT    LCM    MU    RAD    ERROR    NU    VIEW')
    
    # If subdirectory does not exist, create it
    dir = 'spectral_merid_profiles/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Save spectral meridional profiles
    for ifilt in range(BinningInputs.nfilters):
        # Write individual mean profiles to np.array
        filt = VisirWavenumbers(ifilt)
        np.save(f"{dir}{filt}_merid_profile", spectrals[:, ifilt, :])
        # Write individual mean profiles to textfile
        np.savetxt(f"{dir}{filt}_merid_profile.txt", spectrals[:, ifilt, :],
                    fmt=['%4.2f', '%5.2f', '%4.2f', '%8.5e', '%8.5e', '%8.5f'],
                    header='LAT    LCM    MU    RAD    ERROR    NU')