import numpy as np

def ReadSolarWindPredi(filepath):

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
        year              = data_arr[:, 0]
        month             = data_arr[:, 1]
        day               = data_arr[:, 2]
        hour              = data_arr[:, 3]
        minute            = data_arr[:, 4]
        second            = data_arr[:, 5]
        SW_dyna_press     = data_arr[:, 11]
        heliocentric_long = data_arr[:, 12]

    return year, month, day, hour, minute, second, SW_dyna_press, heliocentric_long
