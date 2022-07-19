import numpy as np

def ReadTemperaturePriorProfile(filename):
    """Read temperature and gases prior profiles file and return arrays of profiles"""

    with open(filename) as f:
        # Read header contents
        lines = f.readlines()
        # Save header information
        prior_param = lines[1].split()
        nlevel = int(prior_param[2])
        ngas = int(prior_param[3])
        # Store NEMESIS gases codes 
        newline, gasdata = [], []
        for iline, line in enumerate(lines[2:ngas+2]):
            l = line.split()
            [newline.append(il) for il in l]
            # Store codes
            gasdata.append(newline)
            # Reset temporary variables
            newline = []
        gasname = np.asarray(gasdata, dtype='int')

        # Read and save prior file's data
        priordata = []
        for iline, line in enumerate(lines[ngas+3::]):
            l = line.split()
            [newline.append(il) for il in l]
            # Store data 
            priordata.append(newline)
            # Reset temporary variables
            newline = []
        prior = np.asarray(priordata, dtype='float')
        # Split altitude, pressure, temperature and gases profiles in separate (dedicated) arrays
        altitude = prior[:, 0]
        pressure = prior[:, 1]
        temperature = prior[:, 2]
        gas = prior[:,3:ngas+3]

        return altitude, pressure, temperature, gas, gasname, nlevel, ngas

def ReadAerosolPriorProfile(filename):
    """Read aerosols prior profiles file and return array of profile"""

    with open(filename) as f:
        lines = f.readlines()
        # Get dimensions
        param = lines[1].split()
        nlevel = int(param[0])
        ncloud = int(param[1])
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
        
        altitude = aer[:,0]
        aerosol  = aer[:,1]

        return aerosol, altitude, ncloud, nlevel

