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

def RetrieveLongitudeFromCoreNumber(fpath):
    """ Function to retrieve longitude value from the 
        .mre file of each core subdirectory """
    # Initialize local variables
    ncore = int(359)                     # number of core directories (360-1), 
                                    # which is also equivalent to the number of longitude points 
    lon_core = np.empty((ncore, 2)) # 2D array containing longitude and core directory number
    # Read all .prf files through all core directories
    for ifile in range(ncore):
        filename = f"{fpath}_{ifile+1}/nemesis.prf"
        with open(filename) as f:
            # Read header contents
            lines = f.readlines()
            # Save header information
            prior_param = lines[1].split()
            nlevel             = int(prior_param[2])
            ngas               = int(prior_param[3])
        filename = f"{fpath}_{ifile+1}/nemesis.mre"
        with open(filename) as f:
            # Read header contents
            lines = f.readlines()
            # Save header information
            prior_param = lines[2].split()
            lon_core[ifile, 0] = int(ifile+1)                    # Store core number 
            lon_core[ifile, 1] = float(prior_param[1])    # Store corresponding longitude value
    # Sorted on longitude values
    lon_core = sorted(lon_core, key=operator.itemgetter(1))
    lon_core = np.asarray(lon_core, dtype='float')

    return lon_core, ncore, nlevel, ngas

def RetrieveLongitudeLatitudeFromCoreNumber(fpath):
    """ Function to retrieve longitude and latitude
        value from the .mre file of each core
        subdirectory in case of 2D retrieval """
    # Initialize local variables
    ncore = int(800)                # number of core directories (800-1), 
                                    # which is also equivalent to the number of longitude points 
    lat_lon_core = np.empty((ncore, 3)) # 3D array containing latitude, longitude and core directory number
    # Read all .prf files through all core directories
    for ifile in range(ncore):
        filename = f"{fpath}_{ifile+1}/nemesis.prf"
        with open(filename) as f:
            # Read header contents
            lines = f.readlines()
            # Save header information
            prior_param = lines[1].split()
            nlevel             = int(prior_param[2])
            ngas               = int(prior_param[3])
        filename = f"{fpath}_{ifile+1}/nemesis.mre"
        with open(filename) as f:
            # Read header contents
            lines = f.readlines()
            # Save header information
            prior_param = lines[2].split()
            lat_lon_core[ifile, 0] = int(ifile+1)               # Store core number
            lat_lon_core[ifile, 1] = float(prior_param[0])      # Store corresponding latitude value
            lat_lon_core[ifile, 2] = float(prior_param[1])      # Store corresponding longitude value
    # Sorted on latitude and longitude values
    lat_lon_core = sorted(lat_lon_core, key=operator.itemgetter(2), reverse=True) # Sorted descending longitude values to fit the System III W longitude
    lat_lon_core = sorted(lat_lon_core, key=operator.itemgetter(1)) # Sorted ascending latitude values
    lat_lon_core = np.asarray(lat_lon_core, dtype='float')
    # Create longitude and latitude output arrays
    latitude = [] 
    [latitude.append(x) for x in lat_lon_core[:,1] if x not in latitude]
    latitude = np.asarray(latitude, dtype='float')
    longitude = [] 
    [longitude.append(x) for x in lat_lon_core[:,2] if x not in longitude]
    longitude = np.asarray(longitude, dtype='float')

    return lat_lon_core, latitude, longitude, ncore, nlevel, ngas

def ReadLogFiles(filepath, over_axis):
    """ Read chisq/n from retrieval log_** files """

    if over_axis == "latitude":
        # Retrieve latitude-core_number correspondance
        coor_core, ncoor, _, _ = RetrieveLatitudeFromCoreNumber(f"{filepath}/core")
    elif over_axis =="longitude":
        # Retrieve longitude-core_number correspondance
        coor_core, ncoor, _, _ = RetrieveLongitudeFromCoreNumber(f"{filepath}/core")
    elif over_axis=="2D":
        # Retrieve latitude-longitude-core_number correspondance
        coor_core, latitude, longitude, ncoor, _, _ = RetrieveLongitudeLatitudeFromCoreNumber(f"{filepath}/core")
    
    # Create a latitude or longitude array for plotting
    coor_axis = np.asarray(coor_core[:,1], dtype='float')
    # Create a chisquare array
    chisquare = np.empty((ncoor))
    # Read and save retrieved profiles
    for icoor in range(ncoor):
        ifile = int(coor_core[icoor, 0])
        with open(f"{filepath}/core_{ifile}/log_{ifile}") as f:
            # Read file
            lines = f.readlines()
            for iline, line in enumerate(lines):
                l = line.split()
                # Identifying the last line of chisq/ny in the current log file
                if 'chisq/ny' and 'equal' in l:
                    tmp = line.split(':')
                    chisq = tmp[-1]
                    chisquare[icoor] = chisq
    if over_axis=="2D":
        # Create suitable dimensions for output arrays
        nlat = len(latitude)
        nlon = len(longitude)
        # Reshape the output arrays in case of 2D retrieval
        chisquare = np.reshape(chisquare, (nlat, nlon))

    if over_axis=="2D": return chisquare, latitude, nlat, longitude, nlon
    if over_axis=="latitude" or "longitude": return chisquare, coor_axis, ncoor

def ReadmreFiles(filepath, over_axis):
    """ Read radiance retrieval outputs for all .mre files """

    if over_axis=="latitude":
        # Retrieve latitude-core_number correspondance
        coor_core, ncoor, _, _ = RetrieveLatitudeFromCoreNumber(f"{filepath}/core") 
    elif over_axis=="longitude":
        # Retrieve longitude-core_number correspondance
        coor_core, ncoor, _, _ = RetrieveLongitudeFromCoreNumber(f"{filepath}/core")
    elif over_axis=="2D":
        # Retrieve latitude-longitude-core_number correspondance
        coor_core, latitude, longitude, ncoor, _, _ = RetrieveLongitudeLatitudeFromCoreNumber(f"{filepath}/core")

    # Create outputs arrays with suitable dimensions
    radiance = np.empty((Globals.nfilters, ncoor))
    rad_err = np.empty((Globals.nfilters, ncoor))
    rad_fit = np.empty((Globals.nfilters, ncoor))
    wavenumb = np.empty((Globals.nfilters, ncoor))
    radiance.fill(np.nan)
    rad_err.fill(np.nan)
    rad_fit.fill(np.nan)
    wavenumb.fill(np.nan) 
    # Create a latitude or longitude array for plotting
    coor_axis = np.asarray(coor_core[:,1], dtype='float')
    # Read and save retrieved profiles
    for icoor in range(ncoor):
        ifile = int(coor_core[icoor, 0])
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
            if Globals.nfilters > 11:
                if ((coor_axis[icoor] < 6)):
                    wavenumb[:, icoor] = data_arr[:, 1]
                    radiance[:, icoor] = data_arr[:, 2]
                    rad_err[:, icoor]  = data_arr[:, 3]
                    rad_fit[:, icoor]  = data_arr[:, 5]
                elif (coor_axis[icoor] > 6):
                    wavenb = data_arr[:, 1]
                    rad = data_arr[:, 2]
                    err  = data_arr[:, 3]
                    fit  = data_arr[:, 5]
                    if Globals.nfilters ==12:
                        for ifilt in range(Globals.nfilters-1):
                            if ifilt <= 5:
                                wavenumb[ifilt, icoor] = wavenb[ifilt]
                                radiance[ifilt, icoor] = rad[ifilt]
                                rad_err[ifilt, icoor]  = err[ifilt]
                                rad_fit[ifilt, icoor]  = fit[ifilt]
                            else:
                                wavenumb[ifilt+1, icoor] = wavenb[ifilt]
                                radiance[ifilt+1, icoor] = rad[ifilt]
                                rad_err[ifilt+1, icoor]  = err[ifilt]
                                rad_fit[ifilt+1, icoor]  = fit[ifilt]
                    elif Globals.nfilters ==13:
                        for ifilt in range(Globals.nfilters-2):
                            if ifilt <= 5:
                                wavenumb[ifilt, icoor] = wavenb[ifilt]
                                radiance[ifilt, icoor] = rad[ifilt]
                                rad_err[ifilt, icoor]  = err[ifilt]
                                rad_fit[ifilt, icoor]  = fit[ifilt]
                            else:
                                wavenumb[ifilt+2, icoor] = wavenb[ifilt]
                                radiance[ifilt+2, icoor] = rad[ifilt]
                                rad_err[ifilt+2, icoor]  = err[ifilt]
                                rad_fit[ifilt+2, icoor]  = fit[ifilt]
            else:
                wavenumb[:, icoor] = data_arr[:, 1]
                radiance[:, icoor] = data_arr[:, 2]
                rad_err[:, icoor]  = data_arr[:, 3]
                rad_fit[:, icoor]  = data_arr[:, 5]
    
    if over_axis=="2D":
        # Create suitable dimensions for output arrays
        nlat = len(latitude)
        nlon = len(longitude)
        # Reshape the output arrays in case of 2D retrieval
        radiance = np.reshape(radiance, (Globals.nfilters, nlat, nlon))
        rad_err  = np.reshape(rad_err, (Globals.nfilters, nlat, nlon))
        rad_fit  = np.reshape(rad_fit, (Globals.nfilters, nlat, nlon))
        wavenumb = np.reshape(wavenumb, (Globals.nfilters, nlat, nlon))

    if over_axis=="2D": return radiance, wavenumb, rad_err, rad_fit, latitude, nlat, longitude, nlon
    if over_axis=="latitude" or "longitude": return radiance, wavenumb, rad_err, rad_fit, coor_axis, ncoor 

def ReadprfFiles(filepath, over_axis):
    """ Read retrieved temperature and gases output profiles for all .prf files """

    if over_axis=="latitude":
        # Retrieve latitude-core_number correspondance
        coor_core, ncoor, nlevel, ngas = RetrieveLatitudeFromCoreNumber(f"{filepath}/core")
    elif over_axis=="longitude":
        # Retrieve longitude-core_number correspondance
        coor_core, ncoor, nlevel, ngas = RetrieveLongitudeFromCoreNumber(f"{filepath}/core")
    elif over_axis=="2D":
        # Retrieve latitude-longitude-core_number correspondance
        coor_core, latitude, longitude, ncoor, nlevel, ngas = RetrieveLongitudeLatitudeFromCoreNumber(f"{filepath}/core")
    
    # Initialize output arrays
    height      = np.empty((nlevel, ncoor))          # height array which varies with latitude
    pressure    = np.empty((nlevel))                # pressure array 
    temperature = np.empty((nlevel, ncoor))          # temperature array with profiles for all latitude + pressure profiles ([:,0])
    gases       = np.empty((nlevel, ncoor, ngas))    # gases array with profiles for all latitude + pressure profiles ([:,0])
    gases_id    = np.empty((ngas))
    # Create a latitude or longitude array for plotting
    coor_axis = np.asarray(coor_core[:,1], dtype='float')
    # Read and save retrieved profiles
    for icoor in range(ncoor):
        ifile = int(coor_core[icoor, 0])
        with open(f"{filepath}/core_{ifile}/nemesis.prf") as f:
            # Read file
            lines = f.readlines()
            # Store gases id
            for igas in range(ngas):
                id = lines[igas+2].split()
                gases_id[igas] = id[0]
                gases_id = np.asarray(gases_id, dtype=int)
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
            height[:, icoor]         = data_arr[:, 0]            # height profile
            pressure[:]             = data_arr[:, 1] * 1013.25  # pressure profile converted in mbar
            temperature[:, icoor]    = data_arr[:, 2]            # temperature profile
            for igas in range(ngas):
                gases[:, icoor, igas] = data_arr[:,igas+3]
    
    if over_axis=="2D":
        # Create suitable dimensions for output arrays
        nlat = len(latitude)
        nlon = len(longitude)
        # Reshape the output arrays in case of 2D retrieval
        height      = np.reshape(height, (nlevel, nlat, nlon))
        temperature = np.reshape(temperature, (nlevel, nlat, nlon))
        gases       = np.reshape(gases, (nlevel, nlat, nlon, ngas))
    
    if over_axis=="2D": return temperature, gases, latitude, longitude, height, pressure, ncoor, nlevel, nlat, nlon, ngas, gases_id
    if over_axis=="latitude" or "longitude": return temperature, gases, coor_axis, height, pressure, ncoor, nlevel, ngas, gases_id

def ReadaerFiles(filepath, over_axis):
    """ Read retrieved aerosol output profiles for all .aer files """

    if over_axis=="latitude":
        # Retrieve latitude-core_number correspondance
        coor_core, ncoor, nlevel, _ = RetrieveLatitudeFromCoreNumber(f"{filepath}/core")
    elif over_axis=="longitude":
        # Retrieve longitude-core_number correspondance
        coor_core, ncoor, nlevel, _ = RetrieveLongitudeFromCoreNumber(f"{filepath}/core")
    elif over_axis=="2D":
        # Retrieve latitude-longitude-core_number correspondance
        coor_core, latitude, longitude, ncoor, nlevel, _ = RetrieveLongitudeLatitudeFromCoreNumber(f"{filepath}/core")

    # Initialize output arrays
    aerosol = np.empty((nlevel, ncoor)) # aerosol array with profiles for all latitude
    height  = np.empty((nlevel, ncoor)) # height array with profiles for all latitude 
    # Create a latitude or longitude array for plotting
    coor_axis = coor_core[:,1]
    # Read and save retrieved profiles
    for icoor in range(ncoor):
        ifile = int(coor_core[icoor, 0])
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
            height[:, icoor] = aer[:,0]
            aerosol[:, icoor]  = aer[:,1]

    if over_axis=="2D":
        # Create suitable dimensions for output arrays
        nlat = len(latitude)
        nlon = len(longitude)
        # Reshape the output arrays in case of 2D retrieval
        height  = np.reshape(height, (nlevel, nlat, nlon))
        aerosol = np.reshape(aerosol, (nlevel, nlat, nlon))
        
    if over_axis=="2D": return aerosol, height, nlevel, latitude, nlat, longitude, nlon, ncoor
    if over_axis=="latitude" or "longitude": return aerosol, height, coor_axis, nlevel, ncoor

def ReadmreParametricTest(filepath):

    ncores = 1000
    radiance = np.empty((Globals.nfilters, ncores))
    rad_err = np.empty((Globals.nfilters, ncores))
    rad_fit = np.empty((Globals.nfilters, ncores))
    rad_diff = np.empty((Globals.nfilters, ncores))
    wavenumb = np.empty((Globals.nfilters, ncores))
    coeffs = np.empty ((3, ncores))
    radiance.fill(np.nan)
    rad_err.fill(np.nan)
    rad_fit.fill(np.nan)
    rad_diff.fill(np.nan)
    wavenumb.fill(np.nan) 
    coeffs.fill(np.nan)
    # Read and save retrieved profiles
    for icore in range(ncores):
        with open(f"{filepath}core_{icore+1}/nemesis.mre") as f:
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
            # Fill the wavenumber, radiance prior, radiance error and radiance fitted arrays
            wavenumb[:, icore] = data_arr[:, 1]
            radiance[:, icore] = data_arr[:, 2]
            rad_err[:, icore]  = data_arr[:, 3]
            rad_fit[:, icore]  = data_arr[:, 5]
            rad_diff[:, icore]  = data_arr[:, 6]
    # Retrieve coefficients for the three hydrocarbons profiles in each core subdirectories        
    for icore in range(ncores):
        with open(f"{filepath}core_{icore+1}/coeff_hydrocarbons.txt") as f:
            tmp = f.readlines()
            tmp = tmp[0].split()
            coeffs[0, icore] = tmp[0]
            coeffs[1, icore] = tmp[1]
            coeffs[2, icore] = tmp[2]

    return radiance, wavenumb, rad_err, rad_fit, rad_diff, coeffs, ncores