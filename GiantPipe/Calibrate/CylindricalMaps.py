from Read.ReadFits import ReadFits
from Tools.SetWave import SetWave
from Write.WriteFits import WriteFits

def CalCylindricalMaps(files, ksingles, kspectrals):
    """Calibrate (or re-calibrate) cylindrical maps using calculated calibration coefficients.
       Used for creating global maps and different binning schemes."""

    # Loop over files and calibration coefficients
    for ifile, fpath in enumerate(files):
        
        # Read in uncalibrated observations (images and cmaps) from .fits files
        imghead, imgdata, cylhead, cyldata, muhead, mudata = ReadFits(filepath=f"{fpath}")

        # Get filter index for spectral profiles
        _, _, wave, _, ifilt_v = SetWave(filename=fpath, wavelength=None, wavenumber=None, ifilt=None)

        # Do calibration: divide raw radiances by calibration coefficient
        imgdata /= kspectrals[ifilt_v, 1]
        imgdata /= ksingles[ifile, 1]
        cyldata /= kspectrals[ifilt_v, 1]
        cyldata /= ksingles[ifile, 1]

        # Write calibrated observations (images and cmaps) to .fits files
        WriteFits(filepath=f"{fpath}", imghead=imghead, imgdata=imgdata, cylhead=cylhead, cyldata=cyldata, muhead=muhead, mudata=mudata)