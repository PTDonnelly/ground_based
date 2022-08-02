import numpy as np

def ReadTemperaturePriorProfile(filepath):
    """Read temperature and gases prior profiles file and return arrays of profiles"""
    filename = f"{filepath}nemesis.ref"
    with open(filename) as f:
        # Read header contents
        lines = f.readlines()
        # Save header information
        prior_param = lines[54].split()
        nlevel = int(prior_param[2])
        ngas = int(prior_param[3])
        # Store NEMESIS gases codes 
        newline, gasdata = [], []
        for iline, line in enumerate(lines[55:ngas+55]):
            l = line.split()
            [newline.append(il) for il in l]
            # Store codes
            gasdata.append(newline)
            # Reset temporary variables
            newline = []
        gasname = np.asarray(gasdata, dtype='int')

        # Read and save prior file's data
        priordata = []
        for iline, line in enumerate(lines[ngas+56::]):
            l = line.split()
            [newline.append(il) for il in l]
            # Store data 
            priordata.append(newline)
            # Reset temporary variables
            newline = []
        prior = np.asarray(priordata, dtype='float')
        # Split altitude, pressure, temperature and gases profiles in separate (dedicated) arrays
        altitude = prior[:, 0]
        pressure = prior[:, 1] * 1013.25
        temperature = prior[:, 2]
        gas = prior[:,3:ngas+3]
    
    filename = f"{filepath}tempapr.dat"
    with open(filename) as f:
        # Read header contents
        lines = f.readlines()
        # Read and save prior file's data
        errdata = []
        for iline, line in enumerate(lines[1::]):
            l = line.split()
            [newline.append(il) for il in l]
            # Store data 
            errdata.append(newline)
            # Reset temporary variables
            newline = []
        err = np.asarray(errdata, dtype='float')
        err_tem = err[:,2]


        return altitude, pressure, temperature, err_tem, gas, gasname, nlevel, ngas

def ReadAerosolPriorProfile(filename):
    """Read aerosols prior profiles file and return array of profile"""

    with open(filename) as f:
        lines = f.readlines()
        # Get dimensions
        param = lines[1].split()
        nlevel = int(param[0])
        ncloud = int(param[1])
        aerosol = np.empty((nlevel, ncloud))
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
        if ncloud > 1:
            for icloud in range(ncloud):
                aerosol[:, icloud] = aer[:, icloud+1]
        else:
            aerosol  = aer[:,1]

        return aerosol, altitude, ncloud, nlevel

