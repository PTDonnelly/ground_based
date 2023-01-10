import numpy as np
import Globals

def ReadSpxFiles(filepath):
    """ Read .spx files over each latitude """

    latgrid     = np.arange(-89.5, 87.5, 1)
    nlatbins = len(latgrid)

    radiance = np.empty((Globals.nfilters, nlatbins))
    rad_err  = np.empty((Globals.nfilters, nlatbins))
    wavenumb = np.empty((Globals.nfilters))
    radiance.fill(np.nan)
    rad_err.fill(np.nan)
    wavenumb.fill(np.nan)

    for ifilt in range(Globals.nfilters):
        for ilat in range(nlatbins):
            with open(f"{filepath}lat_{latgrid[ilat]}.txt") as f:
                # Read file
                lines = f.readlines()
                # Get dimensions
                param = lines[0].split()
                lat_check = param[1]
                # Save file's data
                newline, filedata = [], []
                for iline, line in enumerate(lines[4::4]):
                    l = line.split()
                    [newline.append(il) for il in l]
                    # Store data 
                    filedata.append(newline)
                    # Reset temporary variables
                    newline = []
                data_arr = np.asarray(filedata, dtype='float')
                if ((latgrid[ilat] < 6)):
                    wavenumb[:] = data_arr[:, 0]
                    radiance[:, ilat] = data_arr[:, 1]
                    rad_err[:, ilat]  = data_arr[:, 2]
                elif (latgrid[ilat] > 6):
                    wavenb  = data_arr[:, 0]
                    rad     = data_arr[:, 1]
                    err     = data_arr[:, 2]
                    for ifilt in range(Globals.nfilters-2):
                        if ifilt <= 5:
                            wavenumb[ifilt] = wavenb[ifilt]
                            radiance[ifilt, ilat] = rad[ifilt]
                            rad_err[ifilt, ilat]  = err[ifilt]
                        else:
                            wavenumb[ifilt+2] = wavenb[ifilt]
                            radiance[ifilt+2, ilat] = rad[ifilt]
                            rad_err[ifilt+2, ilat]  = err[ifilt]
    
    return radiance, rad_err, wavenumb, latgrid
