import numpy as np
import operator
import Globals

def RetrieveLatitudeFromCoreNumber(fpath):

    # Initialize local variables
    ncore = int(176)                     # number of core directories (177-1), 
                                    # which is also equivalent to the number of latitude points 
    lat_core = np.empty((ncore, 2)) # 2D array containing latitude and core directory number
    # Read all .prf files through all core directories
    for ifile in range(ncore):
        filename = f"{fpath}_{ifile+1}/nemesis.prf"
        with open(filename) as f:
            # Read header contents
            lines = f.readlines()
            # Save header information
            prior_param = lines[1].split()
            lat_core[ifile, 0] = int(ifile+1)                    # Store core number 
            lat_core[ifile, 1] = float(prior_param[1])    # Store corresponding latitude value
            nlevel             = int(prior_param[2])
            ngas               = int(prior_param[3])
    # Sorted on latitude values
    lat_core = sorted(lat_core, key=operator.itemgetter(1))
    lat_core = np.asarray(lat_core, dtype='float')

    return lat_core, ncore, nlevel, ngas

def ReadLogFiles(filepath):
    """ Read chisq/n from retrieval log files """

    # Retrieve latitude-core_number correspondance
    lat_core, nlat, nlevel, ngas = RetrieveLatitudeFromCoreNumber(f"{filepath}/core")


def ReadmreFiles(filepath):
    """ Read radiance retrieval outputs for all .mre files """

    # Retrieve latitude-core_number correspondance
    lat_core, nlat, nlevel, ngas = RetrieveLatitudeFromCoreNumber(f"{filepath}/core")
    
    radiance = np.empty((Globals.nfilters, nlat))
    rad_err = np.empty((Globals.nfilters, nlat))
    rad_fit = np.empty((Globals.nfilters, nlat))
    wavenumb = np.empty((Globals.nfilters, nlat))
    radiance.fill(np.nan)
    rad_err.fill(np.nan)
    rad_fit.fill(np.nan)
    wavenumb.fill(np.nan) 
    # Create a latitude array for plotting
    latitude = np.asarray(lat_core[:,1], dtype='float')
    # Read and save retrieved profiles
    for ilat in range(nlat):
        ifile = int(lat_core[ilat, 0])
        with open(f"{filepath}/core_{ifile}/nemesis.mre") as f:
            # Read file
            lines = f.readlines()
            # Get dimensions
            param = lines[1].split()
            ispec = int(param[0])
            ngeom = int(param[1])
            nx    = int(param[3])
            # Save file's data
            newline, filedata = [], []
            for iline, line in enumerate(lines[5:ngeom+5]):
                l = line.split()
                [newline.append(il) for il in l]
                # Store data 
                filedata.append(newline)
                # Reset temporary variables
                newline = []
            data_arr = np.asarray(filedata, dtype='float')
            if ((latitude[ilat] < 6)):
                wavenumb[:, ilat] = data_arr[:, 1]
                radiance[:, ilat] = data_arr[:, 2]
                rad_err[:, ilat]  = data_arr[:, 3]
                rad_fit[:, ilat]  = data_arr[:, 5]
            elif (latitude[ilat] > 6):
                wavenb = data_arr[:, 1]
                rad = data_arr[:, 2]
                err  = data_arr[:, 3]
                fit  = data_arr[:, 5]
                for ifilt in range(Globals.nfilters-2):
                    if ifilt <= 5:
                        wavenumb[ifilt, ilat] = wavenb[ifilt]
                        radiance[ifilt, ilat] = rad[ifilt]
                        rad_err[ifilt, ilat]  = err[ifilt]
                        rad_fit[ifilt, ilat]  = fit[ifilt]
                    else:
                        wavenumb[ifilt+2, ilat] = wavenb[ifilt]
                        radiance[ifilt+2, ilat] = rad[ifilt]
                        rad_err[ifilt+2, ilat]  = err[ifilt]
                        rad_fit[ifilt+2, ilat]  = fit[ifilt]

    return radiance, wavenumb, rad_err, rad_fit, latitude, nlat

def ReadprfFiles(filepath):
    """ Read retrieved temperature and gases output profiles for all .prf files """

    # Retrieve latitude-core_number correspondance
    lat_core, nlat, nlevel, ngas = RetrieveLatitudeFromCoreNumber(f"{filepath}/core")
    # Initialize output arrays
    height      = np.empty((nlevel, nlat))          # height array which varies with latitude
    pressure    = np.empty((nlevel))                # pressure array 
    temperature = np.empty((nlevel, nlat))          # temperature array with profiles for all latitude + pressure profiles ([:,0])
    gases       = np.empty((nlevel, nlat, ngas))    # gases array with profiles for all latitude + pressure profiles ([:,0])
    # Create a latitude array for plotting
    latitude = np.asarray(lat_core[:,1], dtype='float')
    # Read and save retrieved profiles
    for ilat in range(nlat):
        ifile = int(lat_core[ilat, 0])
        with open(f"{filepath}/core_{ifile}/nemesis.prf") as f:
            # Read file
            lines = f.readlines()
            # Save file's data
            newline, filedata = [], []
            for iline, line in enumerate(lines[ngas+3::]):
                l = line.split()
                [newline.append(il) for il in l]
                # Store data 
                filedata.append(newline)
                # Reset temporary variables
                newline = []
            data_arr = np.asarray(filedata, dtype='float')
            # Split temperature and gases profiles in separate (dedicated) arrays
            height[:, ilat]         = data_arr[:, 0]            # height profile
            pressure[:]             = data_arr[:, 1] * 1013.25  # pressure profile converted in mbar
            temperature[:, ilat]    = data_arr[:, 2]            # temperature profile
            for igas in range(ngas):
                gases[:, ilat, igas] = data_arr[:,igas+3]
    
    return temperature, gases, latitude, height, pressure, nlat, nlevel, ngas

def ReadaerFiles(filepath):
    """ Read retrieved aerosol output profiles for all .aer files """

    # Retrieve latitude-core_number correspondance
    lat_core, nlat, nlevel, _ = RetrieveLatitudeFromCoreNumber(fpath=f"{filepath}/core")
    # Initialize output arrays
    aerosol = np.empty((nlevel, nlat)) # aerosol array with profiles for all latitude
    height  = np.empty((nlevel, nlat)) # height array with profiles for all latitude 
    # Create a latitude array for plotting
    latitude = lat_core[:,1]
    # Read and save retrieved profiles
    for ilat in range(nlat):
        ifile = int(lat_core[ilat, 0])
        with open(f"{filepath}/core_{ifile}/aerosol.prf") as f:
            lines = f.readlines()
            # Read and store aerosol profiles
            newline, data = [], []
            for iline, line in enumerate(lines[2::]):
                l = line.split()
                [newline.append(il) for il in l]
                # Store data 
                data.append(newline)
                # Reset temporary variables
                newline = []
            aer = np.asarray(data, dtype='float')
            # Split altitude and aerosol profiles in separate (dedicated) arrays
            height[:, ilat] = aer[:,0]
            aerosol[:, ilat]  = aer[:,1]

    return aerosol, height, latitude, nlevel, nlat