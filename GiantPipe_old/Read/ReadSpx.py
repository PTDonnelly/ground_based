import numpy as np
import Globals

def ReadSpx(files):
    spectrum = np.zeros((Globals.nfilters, len(files), 3))
    for ifile, filename in enumerate(files):
        # Open file
        with open(filename) as f:
            # Read contents
            lines = f.readlines()
            tmp = lines[0].split()
            lat = tmp[1]
            igeom = 0
            for iline, line in enumerate(lines):
                if (iline > 0) and (iline % 4) == 0:
                    tmp = line.split()
                    spectrum[igeom, ifile, 0] = float(tmp[0])
                    spectrum[igeom, ifile, 1] = float(tmp[1])
                    spectrum[igeom, ifile, 2] = float(tmp[2])
                    igeom += 1