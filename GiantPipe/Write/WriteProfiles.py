import os 
import numpy as np 
import Globals
from Tools.SetWave import SetWave

def WriteMeridProfiles(dataset, files, singles, spectrals):
    """Save calibrated profiles (and optionally coefficients) as
    numpy arrays and textfiles"""

    print('Saving profiles...')
    
    if np.any(singles):
        # If subdirectory does not exist, create it
        dir = f'../outputs/{dataset}/single_merid_profiles/'
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
        
    if np.any(spectrals):
        # If subdirectory does not exist, create it
        dir = f'../outputs/{dataset}/spectral_merid_profiles/'
        if not os.path.exists(dir):
            os.makedirs(dir)
        # Save spectral meridional profiles
        for ifilt in range(Globals.nfilters):
            # Write spectral mean profiles to np.array
            _, _, wave, _, _ = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=ifilt)
            np.save(f"{dir}{wave}_merid_profile", spectrals[:, ifilt, :])
            # Write spectral mean profiles to textfiles
            np.savetxt(f"{dir}{wave}_merid_profile.txt", spectrals[:, ifilt, :],
                        fmt=['%4.2f', '%5.2f', '%4.2f', '%8.5e', '%8.5e', '%8.5f'],
                        header='LAT    LCM    MU    RAD    ERROR    NU')

def WriteCentreToLimbProfiles(mode, files, singles, spectrals):
            """Save calibrated profiles (and optionally coefficients) as
            numpy arrays and textfiles"""

            print('Saving profiles...')
            
            a = 1