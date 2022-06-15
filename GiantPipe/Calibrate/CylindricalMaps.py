from Read.ReadFits import ReadFits
from Write.WriteFits import WriteFits

def CalCylindricalMaps(files, ksingles):
    """Calibrate (or re-calibrate) cylindrical maps using calculated calibration coefficients.
       Used for creating global maps and different binning schemes."""

    # Pull out the calibration coefficients
    kcoeffs = ksingles[:, 1]

    # Loop over files and calibration coefficients
    for fpath, kcoeff in zip(files, kcoeffs):
        # Read in uncalibrated observations (images and cmaps) from .fits files
        imghead, imgdata, cylhead, cyldata, muhead, mudata = ReadFits(filepath=f"{fpath}")

        # Do calibration: divide raw radiances by calibration coefficient
        imgdata /= kcoeff
        cyldata /= kcoeff

        # Write calibrated observations (images and cmaps) to .fits files
        WriteFits(filepath=f"{fpath}", imghead=imghead, imgdata=imgdata, cylhead=cylhead, cyldata=cyldata, muhead=muhead, mudata=mudata)







