import numpy as np
import operator

def RetrieveLatitudeFromCoreNumber(fpath):

    # Initialize local variables
    ncore = int(176)                     # number of core directories (177-1), 
                                    # which is also equivalent to the number of latitude points 
    lat_core = np.empty((ncore, 2)) # 2D array containing latitude and core directory number
    # Read all .prf files through all core directories
    for ifile in range(ncore):
        filename = f"{fpath}_{ifile+1}.prf"
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
    print(lat_core)
    return lat_core, ncore, nlevel, ngas

def ReadmreFiles(filepath):
    """ Read radiance retrieval outputs for all .mre files """

    # Retrieve latitude-core_number correspondance
    lat_core, nlat, nlevel, ngas = RetrieveLatitudeFromCoreNumber(f"{filepath}/prffiles/visir_2018May")
    

def ReadprfFiles(filepath):
    """ Read retrieved temperature and gases output profiles for all .prf files """

    # Retrieve latitude-core_number correspondance
    lat_core, nlat, nlevel, ngas = RetrieveLatitudeFromCoreNumber(f"{filepath}/prffiles/visir_2018May")
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
        print(ifile, ilat)
        with open(f"{filepath}/prffiles/visir_2018May_{ifile+1}.prf") as f:
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
            height[:, ilat]         = data_arr[:, 0] # height profile
            pressure[:]             = data_arr[:, 1] # pressure profile
            temperature[:, ilat]    = data_arr[:, 2] # temperature profile
            for igas in range(ngas):
                gases[:, ilat, igas] = data_arr[:,igas+3]
    
    return temperature, gases, latitude, height, pressure, nlat, nlevel, ngas

def ReadaerFiles(filepath):
    """ Read retrieved aerosol output profiles for all .aer files """

    # Retrieve latitude-core_number correspondance
    lat_core, nlat, nlevel, _ = RetrieveLatitudeFromCoreNumber(fpath=f"{filepath}/prffiles/visir_2018May")
    # Initialize output arrays
    aerosol = np.empty((nlevel, nlat)) # aerosol array with profiles for all latitude
    height  = np.empty((nlevel, nlat)) # height array with profiles for all latitude 
    # Create a latitude array for plotting
    latitude = lat_core[:,1]
    # Read and save retrieved profiles
    for ilat in range(nlat):
        ifile = int(lat_core[ilat, 0])
        with open(f"{filepath}/aerfiles/visir_2018May_{ifile+1}.aer") as f:
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