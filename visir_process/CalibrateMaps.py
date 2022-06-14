import numpy as np
from math import acos, cos, radians, pi
from ReadFits import ReadFits

def CalibrateMaps(files, ksingles):
    """Calibrate (or re-calibrate) cylindrical maps using calculated calibration coefficients.
       Used for creating global maps and different binning schemes."""

    # Pull out the calibration coefficients
    kcoeffs = ksingles[:, 1]

    # Loop over files and calibration coefficients
    for fname, kcoeff in zip(files, kcoeffs):
        # Read in uncalibrated observations (images and cmaps) from .fits files
        imghead, imgdata, cylhead, cyldata, _, _ = ReadFits(filename=f"{fname}")

        # Do calibration: divide raw radiances by calibration coefficient
        imgdata /= kcoeff
        cyldata /= kcoeff

        # # Write calibrated observations (images and cmaps) to .fits files
        # imghead, imgdata, cylhead, cyldata, muhead, mudata = WriteFits(imghead, imgdata, cylhead, cyldata)







