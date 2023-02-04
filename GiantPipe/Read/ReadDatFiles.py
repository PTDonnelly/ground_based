import numpy as np

def ReadDatFiles(filepath):

    with open(f"{filepath}") as f:
        # Read file
        lines = f.readlines()
        # Get dimensions
        newline, filedata = [], []
        for iline, line in enumerate(lines):
            l = line.split()
            [newline.append(il) for il in l]
            # Store data 
            filedata.append(newline)
            # Reset temporary variables
            newline = []
        # Transform the list on array
        data_arr = np.asarray(filedata, dtype='float')
        # Separate each column to get each parameter 
        lat_dupli = data_arr[:, 0]
        lon_dupli = data_arr[:, 1]
        c2h2      = data_arr[:, 2]
        c2h2_err  = data_arr[:, 3]
        c2h6      = data_arr[:, 4]
        c2h6_err  = data_arr[:, 5]
    # Remove duplicates in both dimension arrays
    latitude = []
    longitude = []
    [latitude.append(item) for item in lat_dupli if item not in latitude]
    [longitude.append(item) for item in lon_dupli if item not in longitude]
    # Calculated the number of latitude and longitude points
    nlat = len(latitude)
    nlon = len(longitude)
    # Reshape de hydrocarbon arrays 
    c2h2     = np.reshape(c2h2, (nlat, nlon))
    c2h2_err = np.reshape(c2h2_err, (nlat, nlon))
    c2h6     = np.reshape(c2h6, (nlat, nlon))
    c2h6_err = np.reshape(c2h6_err, (nlat, nlon))

    return latitude, longitude, c2h2, c2h2_err, c2h6, c2h6_err