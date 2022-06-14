import numpy as np
from math import acos, cos, radians, pi
from ReadFits import ReadFits

def CalibrateMaps(files, ksingles):
    """Calibrate (or re-calibrate) cylindrical maps using calculated calibration coefficients.
       Used for creating global maps and different binning schemes."""


    print(np.shape(files), np.shape(ksingles))
    # kcoeffs = ksingles[1, :]

    for i in range(90)

    # Loop over files and calibration coefficients
    # for (ifile, fname), (ik, ksingle) in zip(files, ksingles):
    for fname, ksingle in zip(files, kcoeffs):
        print(fname, ksingle)

        # _, imgdata, _, cyldata, _, _ = ReadFits(filename=f"{fname}")

        # print(np.shape(imgdata), np.shape(cyldata))

        exit()




