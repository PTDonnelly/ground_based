import os 
import numpy as np 
import Globals
from Tools.VisirFilterInfo import Wavenumbers

def WriteMeridProfiles(dataset, files, singles, spectrals, ksingles, kspectrals):
    """Save calibrated profiles (and optionally coefficients) as
    numpy arrays and textfiles"""
    
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
            if np.any(ksingles):
                    # Write individual calibration coefficients to np.array
                    np.save(f"{dir}{name[-1]}_calib_coeff", ksingles[ifile, :])
                    # Write individual calibration coefficients to textfile
                    np.savetxt(f"{dir}{name[-1]}_calib_coeff.txt", ksingles[ifile, :],
                                fmt=['%8.5f'], header='FILE INDEX        CALIB_COEFF')
        
    if np.any(spectrals):
        # If subdirectory does not exist, create it
        dir = f'../outputs/{dataset}/spectral_merid_profiles/'
        if not os.path.exists(dir):
            os.makedirs(dir)
        # Save spectral meridional profiles
        for ifilt in range(Globals.nfilters):
            # Write spectral mean profiles to np.array
            filt = Wavenumbers(ifilt)
            np.save(f"{dir}{filt}_merid_profile", spectrals[:, ifilt, :])
            # Write spectral mean profiles to textfiles
            np.savetxt(f"{dir}{filt}_merid_profile.txt", spectrals[:, ifilt, :],
                        fmt=['%4.2f', '%5.2f', '%4.2f', '%8.5e', '%8.5e', '%8.5f'],
                        header='LAT    LCM    MU    RAD    ERROR    NU')
            if np.any(kspectrals):
                # Write spectral calibration coefficients to np.array
                np.save(f"{dir}{filt}_calib_coeff", kspectrals[ifilt, :])
                # Write spectral calibration coefficients to textfiles
                np.savetxt(f"{dir}{filt}_calib_coeff.txt", kspectrals[ifilt, :],
                            fmt=['%8.5f'], header='NU      CALIB_COEFF')

def WriteCTLProfiles():
            """Save calibrated profiles (and optionally coefficients) as
            numpy arrays and textfiles"""
            
            a = 1