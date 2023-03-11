import numpy as np
import os
import operator
from scipy.io import FortranFile
import Globals
from Tools.RetrieveGasesNames import RetrieveGasesNames

def RetrieveLatitudeFromCoreNumber(fpath):

    # Initialize local variables
    dirs = list(os.walk(f"{fpath}"))
    ncore = len(dirs[:-1]) # number of core directories, 
                                    # which is also equivalent to the number of latitude points  
    lat_core = np.empty((ncore, 2)) # 2D array containing latitude and core directory number
    # Read all .prf files through all core directories
    for ifile in range(ncore):
        filename = f"{fpath}/core_{ifile+1}/nemesis.prf"
        with open(filename) as f:
            # print(ifile+1)
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
    dirs = list(os.walk(f"{fpath}"))
    ncore = len(dirs[:-1]) # number of core directories, 
                                    # which is also equivalent to the number of longitude points 
    lon_core = np.empty((ncore, 2)) # 2D array containing longitude and core directory number
    # Read all .prf files through all core directories
    for ifile in range(ncore):
        filename = f"{fpath}/core_{ifile+1}/nemesis.prf"
        with open(filename) as f:
            # Read header contents
            lines = f.readlines()
            # Save header information
            prior_param = lines[1].split()
            nlevel             = int(prior_param[2])
            ngas               = int(prior_param[3])
        filename = f"{fpath}/core__{ifile+1}/nemesis.mre"
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
    dirs = list(os.walk(f"{fpath}"))
    ncore = len(dirs[:-1]) # number of core directories, 
                                    # which is also equivalent to the number of latitude/longitude points 
    lat_lon_core = np.empty((ncore, 3)) # 3D array containing latitude, longitude and core directory number
    # Read all .prf files through all core directories
    for ifile in range(ncore):
        filename = f"{fpath}/core_{ifile+1}/nemesis.prf"
        with open(filename) as f:
            # Read header contents
            lines = f.readlines()
            # Save header information
            prior_param = lines[1].split()
            nlevel             = int(prior_param[2])
            ngas               = int(prior_param[3])
        filename = f"{fpath}/core_{ifile+1}/nemesis.mre"
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

def RetrieveNightFromCoreNumber(fpath):
    """ Function to retrieve night number
        from the .spx file of each core
        subdirectory in case of aurora retrieval """
    # Initialize local variables
    ncore = int(3)                # number of core directories (), 
                                    # which is also equivalent to the number of longitude points 
    night_core = np.empty((ncore, 2)) # 2D array containing night ID and core directory number
    for ifile in range(3):
        # ifile = ifile+1 # To skip the average retrieval core
        filename = f"{fpath}_{ifile+1}/nemesis.prf"
        with open(filename) as f:
            # Read header contents
            lines = f.readlines()
            # Save header information
            prior_param = lines[1].split()
            nlevel             = int(prior_param[2])
            ngas               = int(prior_param[3])
        filename = f"{fpath}_{ifile+1}/logfile"
        with open(filename) as f:
            # Read header contents
            lines = f.readlines()
            # Save header information
            spx_filename = lines[1].split('night')
            spx_filename = spx_filename[-1].split('.')
            night_core[ifile-1, 0] = int(spx_filename[0])
            night_core[ifile-1, 1] = ifile+1
    night_core = sorted(night_core, key=operator.itemgetter(1)) # sorted by night
    night_core = np.asarray(night_core, dtype='float')   

    return night_core, ncore, nlevel, ngas

def ReadLogFiles(filepath, over_axis):
    """ Read chisq/n from retrieval log_** files """

    if over_axis == "latitude":
        # Retrieve latitude-core_number correspondance
        coor_core, ncoor, _, _ = RetrieveLatitudeFromCoreNumber(f"{filepath}")
    elif over_axis =="longitude":
        # Retrieve longitude-core_number correspondance
        coor_core, ncoor, _, _ = RetrieveLongitudeFromCoreNumber(f"{filepath}")
    elif over_axis=="2D":
        # Retrieve latitude-longitude-core_number correspondance
        coor_core, latitude, longitude, ncoor, _, _ = RetrieveLongitudeLatitudeFromCoreNumber(f"{filepath}")
    
    # Create a latitude or longitude array for plotting
    coor_axis = np.asarray(coor_core[:,1], dtype='float')
    # Create a chisquare array
    chisquare = np.empty((ncoor))
    chisquare.fill(np.nan)
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

def ReadmreFiles(filepath, over_axis, gas_name):
    """ Read radiance retrieval outputs for all .mre files """

    if over_axis=="latitude":
        # Retrieve latitude-core_number correspondance
        coor_core, ncoor, nlevel, _ = RetrieveLatitudeFromCoreNumber(f"{filepath}") 
    elif over_axis=="longitude":
        # Retrieve longitude-core_number correspondance
        coor_core, ncoor, nlevel, _ = RetrieveLongitudeFromCoreNumber(f"{filepath}")
    elif over_axis=="2D":
        # Retrieve latitude-longitude-core_number correspondance
        coor_core, latitude, longitude, ncoor, nlevel, _ = RetrieveLongitudeLatitudeFromCoreNumber(f"{filepath}")

    # Create outputs arrays with suitable dimensions
    radiance = np.empty((Globals.nfilters, ncoor))
    rad_err = np.empty((Globals.nfilters, ncoor))
    rad_fit = np.empty((Globals.nfilters, ncoor))
    wavenumb = np.empty((Globals.nfilters, ncoor))
    
    temp_prior_mre = np.empty((nlevel, ncoor))
    temp_errprior_mre = np.empty((nlevel, ncoor))
    temp_fit_mre = np.empty((nlevel, ncoor))
    temp_errfit_mre = np.empty((nlevel, ncoor))

    aer_mre = np.empty((ncoor))
    aer_err = np.empty((ncoor)) 
    aer_fit = np.empty((ncoor))
    fit_err = np.empty((ncoor))

    ngas = len(gas_name)
    if ngas > 1:
        gas_scale = np.empty((ngas, ncoor))
        gas_scaleerr = np.empty((ngas, ncoor)) 
        gas_scalefit = np.empty((ngas, ncoor))
        gas_errscalefit = np.empty((ngas, ncoor))

        gas_vmr = np.empty((ngas, ncoor))
        gas_vmrerr = np.empty((ngas, ncoor)) 
        gas_vmrfit = np.empty((ngas, ncoor))
        gas_errvmrfit = np.empty((ngas, ncoor))

        gas_fsh = np.empty((ngas, ncoor))
        gas_fsherr = np.empty((ngas, ncoor)) 
        gas_fshfit = np.empty((ngas, ncoor))
        gas_errfshfit = np.empty((ngas, ncoor))

        gas_abunerr = np.empty((ngas, nlevel, ncoor))

    else:
        gas_scale = np.empty((ncoor))
        gas_scaleerr = np.empty((ncoor)) 
        gas_scalefit = np.empty((ncoor))
        gas_errscalefit = np.empty((ncoor))

        gas_vmr = np.empty((ncoor))
        gas_vmrerr = np.empty((ncoor)) 
        gas_vmrfit = np.empty((ncoor))
        gas_errvmrfit = np.empty((ncoor))

        gas_fsh = np.empty((ncoor))
        gas_fsherr = np.empty((ncoor)) 
        gas_fshfit = np.empty((ncoor))
        gas_errfshfit = np.empty((ncoor))

        gas_abunerr = np.empty((nlevel, ncoor))
    ppo = np.empty((nlevel))
    # Initialise with NaN values to avoid any suprise in the plot figures 
    radiance.fill(np.nan)
    rad_err.fill(np.nan)
    rad_fit.fill(np.nan)
    wavenumb.fill(np.nan)

    temp_prior_mre.fill(np.nan)
    temp_errprior_mre.fill(np.nan)
    temp_fit_mre.fill(np.nan)
    temp_errfit_mre.fill(np.nan)
     
    aer_mre.fill(np.nan)
    aer_err.fill(np.nan) 
    aer_fit.fill(np.nan)
    fit_err.fill(np.nan)

    gas_scale.fill(np.nan)
    gas_scaleerr.fill(np.nan) 
    gas_scalefit.fill(np.nan)
    gas_errscalefit.fill(np.nan)

    gas_vmr.fill(np.nan)
    gas_vmrerr.fill(np.nan) 
    gas_vmrfit.fill(np.nan)
    gas_errvmrfit.fill(np.nan)

    gas_fsh.fill(np.nan)
    gas_fsherr.fill(np.nan) 
    gas_fshfit.fill(np.nan)
    gas_errfshfit.fill(np.nan)

    gas_abunerr.fill(np.nan)
    
    ppo.fill(np.nan)
    # Read pressure array in the nemesis.prf files
    if over_axis=='latitude' or over_axis=='longitude': 
        _, gases, _, _, pressure, _, _, _, _ = ReadprfFiles(filepath, over_axis)
    elif over_axis=='2D':
        _, gases, _, _, _, pressure, _, _, _, _, _, _ = ReadprfFiles(filepath, over_axis)
    pressure = pressure / 1013.25
    # Create a latitude or longitude array for plotting
    coor_axis = np.asarray(coor_core[:,1], dtype='float')
    # Read and save retrieved profiles
    for icoor in range(ncoor):
        ifile = int(coor_core[icoor, 0])
        with open(f"{filepath}/core_{ifile}/nemesis.mre") as f:
            # Read file
            lines = f.readlines()
            # Get dimensions
            if len(lines) > 1:
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

        # Save temperature data

                newline, filedata = [], []
                for iline, line in enumerate(lines[5+ngeom+6:5+ngeom+6+nlevel]):
                    l = line.split()
                    [newline.append(il) for il in l]
                    # Store data 
                    filedata.append(newline)
                    # Reset temporary variables
                    newline = []
                data_arr = np.asarray(filedata, dtype='float')
                temp_prior_mre[:, icoor] = data_arr[:, 2]
                temp_errprior_mre[:, icoor] = data_arr[:, 3]
                temp_fit_mre[:, icoor] = data_arr[:, 4]
                temp_errfit_mre[:, icoor] = data_arr[:, 5]


                # Save aerosol data
                if '          -1           0           3\n' in lines:
                    aer_ind = lines.index('          -1           0           3\n')
                    aer_param = lines[aer_ind+3].split()
                    aer_mre[icoor] = float(aer_param[2])
                    aer_err[icoor] = aer_param[3]
                    aer_fit[icoor] = aer_param[4]
                    fit_err[icoor] = aer_param[5]
                
                # Save gases data
                if ngas > 1:
                    for igas in range(ngas):
                        _, gas_id = RetrieveGasesNames(gas_name=gas_name[igas], gas_id=False)
                        # Scaling retrieval case:
                        if f"          {gas_id}           0           3\n" in lines:
                            gas_ind = lines.index(f"          {gas_id}           0           3\n")
                            gas_param = lines[gas_ind+3].split()
                            gas_scale[igas, icoor] = float(gas_param[2])
                            gas_scaleerr[igas, icoor] = float(gas_param[3])
                            gas_scalefit[igas, icoor] = float(gas_param[4])
                            gas_errscalefit[igas, icoor] = float(gas_param[5])
                            # print(igas, gas_id, gas_errscalefit[igas, icoor])
                        # Parametric retrieval case:
                        if f"          {gas_id}           0          18\n" in lines:
                            gas_ind = lines.index(f"          {gas_id}           0          18\n")
                            # VMR info
                            gas_param = lines[gas_ind+3].split()
                            gas_vmr[igas, icoor] = float(gas_param[2])
                            gas_vmrerr[igas, icoor] = float(gas_param[3])
                            gas_vmrfit[igas, icoor] = float(gas_param[4])
                            gas_errvmrfit[igas, icoor] = float(gas_param[5])
                            # Fractional scale heigth info
                            gas_param = lines[gas_ind+4].split()
                            gas_fsh[igas, icoor] = float(gas_param[2])
                            gas_fsherr[igas, icoor] = float(gas_param[3])
                            gas_fshfit[igas, icoor] = float(gas_param[4])
                            gas_errfshfit[igas, icoor] = float(gas_param[5])
                            # Retrieve the value of the knee pressure for each parametric retrieval 
                            with open(f"{filepath}/core_1/nemesis.apr") as f:
                                # Read file
                                aprlines = f.readlines()
                                #find the right chemical specie
                                if f"{gas_id} 0 18 - {gas_name[igas]} Parameterised\n" in aprlines:
                                    param_ind = aprlines.index(f"{gas_id} 0 18 - {gas_name[igas]} Parameterised\n")
                                    pkn = float(aprlines[param_ind+1])
                            # Calculate the pressure axis of the parametric retrieved profile
                            pknee = pkn * np.ones(nlevel)
                            ppo = np.divide(pressure, pknee)
                            # Calculate the actual error on the abundance using vmr and fsh parameters.
                            for ilev in range(nlevel):
                                gas_abunerr[igas, ilev, icoor] = ppo[ilev]**((1.-gas_fshfit[igas, icoor])/gas_fshfit[igas, icoor]) * gas_errvmrfit[igas, icoor] \
                                            - ppo[ilev]**(1./gas_fshfit[igas, icoor])*np.log(ppo[ilev]) / (gas_fshfit[igas, icoor] * gas_fshfit[igas, icoor] * ppo[ilev]) \
                                            * gas_vmrfit[igas, icoor] * gas_errfshfit[igas, icoor]

                else:
                    _, gas_id = RetrieveGasesNames(gas_name=gas_name, gas_id=False)
                    # Scaling retrieval case:
                    if f"          {gas_id}           0           3\n" in lines:
                        gas_ind = lines.index(f"          {gas_id}           0           3\n")
                        gas_param = lines[gas_ind+3].split()
                        gas_scale[icoor] = float(gas_param[2])
                        gas_scaleerr[icoor] = gas_param[3]
                        gas_scalefit[icoor] = gas_param[4]
                        gas_errscalefit[icoor] = gas_param[5]
                    # Parametric retrieval case:
                    if f"          {gas_id}           0          18\n" in lines:
                        gas_ind = lines.index(f"          {gas_id}           0          18\n")
                        # VMR info
                        gas_param = lines[gas_ind+3].split()
                        gas_vmr[icoor] = float(gas_param[2])
                        gas_vmrerr[icoor] = gas_param[3]
                        gas_vmrfit[icoor] = gas_param[4]
                        gas_errvmrfit[icoor] = gas_param[5]
                        # Fractional scale heigth info
                        gas_param = lines[gas_ind+4].split()
                        gas_fsh[icoor] = float(gas_param[2])
                        gas_fsherr[icoor] = gas_param[3]
                        gas_fshfit[icoor] = gas_param[4]
                        gas_errfshfit[icoor] = gas_param[5]
                        # Retrieve the value of the knee pressure for each parametric retrieval 
                        with open(f"{filepath}/core_1/nemesis.apr") as f:
                            # Read file
                            aprlines = f.readlines()
                            #find the right chemical specie
                            if f"{gas_id} 0 18 - {gas_name} Parameterised\n" in aprlines:
                                param_ind = aprlines.index(f"{gas_id} 0 18 - {gas_name} Parameterised\n")
                                pknee = aprlines[param_ind+1]
                        # Calculate the pressure axis of the parametric retrieved profile
                        ppo = pressure / pknee
                        # Calculate the actual error on the abundance using vmr and fsh parameters.
                        gas_abunerr[ :, icoor] = ppo**((1.-gas_fshfit[icoor])/gas_fshfit[icoor]) * gas_errvmrfit[icoor] \
                                            - ppo**(1./gas_fshfit[icoor])*np.log10(ppo) / (gas_fshfit[icoor] * gas_fshfit[icoor] * ppo) \
                                            * gas_vmrfit[icoor] * gas_errfshfit[icoor]

    if over_axis=="2D":
        # Create suitable dimensions for output arrays
        nlat = len(latitude)
        nlon = len(longitude)
        # Reshape the output arrays in case of 2D retrieval
        radiance = np.reshape(radiance, (Globals.nfilters, nlat, nlon))
        rad_err  = np.reshape(rad_err, (Globals.nfilters, nlat, nlon))
        rad_fit  = np.reshape(rad_fit, (Globals.nfilters, nlat, nlon))
        wavenumb = np.reshape(wavenumb, (Globals.nfilters, nlat, nlon))

        temp_prior_mre    = np.reshape(temp_prior_mre, (nlevel, nlat, nlon))
        temp_errprior_mre = np.reshape(temp_errprior_mre, (nlevel, nlat, nlon))
        temp_fit_mre      = np.reshape(temp_fit_mre, (nlevel, nlat, nlon))
        temp_errfit_mre   = np.reshape(temp_errfit_mre, (nlevel, nlat, nlon))

        aer_mre = np.reshape(aer_mre, (nlat, nlon))
        aer_err = np.reshape(aer_err, (nlat, nlon)) 
        aer_fit = np.reshape(aer_fit, (nlat, nlon))
        fit_err = np.reshape(fit_err, (nlat, nlon))

        if ngas > 1:
            gas_scale = np.reshape(gas_scale, (ngas, nlat, nlon))
            gas_scaleerr = np.reshape(gas_scaleerr, (ngas, nlat, nlon)) 
            gas_scalefit = np.reshape(gas_scalefit, (ngas, nlat, nlon))
            gas_errscalefit = np.reshape(gas_errscalefit, (ngas, nlat, nlon))

            gas_vmr = np.reshape(gas_vmr, (ngas, nlat, nlon))
            gas_vmrerr = np.reshape(gas_vmrerr, (ngas, nlat, nlon)) 
            gas_vmrfit = np.reshape(gas_vmrfit, (ngas, nlat, nlon))
            gas_errvmrfit = np.reshape(gas_errvmrfit, (ngas, nlat, nlon))

            gas_fsh = np.reshape(gas_fsh, (ngas, nlat, nlon))
            gas_fsherr = np.reshape(gas_fsherr, (ngas, nlat, nlon)) 
            gas_fshfit = np.reshape(gas_fshfit, (ngas, nlat, nlon))
            gas_errfshfit = np.reshape(gas_errfshfit, (ngas, nlat, nlon))

            gas_abunerr = np.reshape(gas_abunerr, (ngas, nlevel, nlat, nlon))
        else:
            gas_scale = np.reshape(gas_scale, (nlat, nlon))
            gas_scaleerr = np.reshape(gas_scaleerr, (nlat, nlon)) 
            gas_scalefit = np.reshape(gas_scalefit, (nlat, nlon))
            gas_errscalefit = np.reshape(gas_errscalefit, (nlat, nlon))

            gas_vmr = np.reshape(gas_vmr, (nlat, nlon))
            gas_vmrerr = np.reshape(gas_vmrerr, (nlat, nlon)) 
            gas_vmrfit = np.reshape(gas_vmrfit, (nlat, nlon))
            gas_errvmrfit = np.reshape(gas_errvmrfit, (nlat, nlon))

            gas_fsh = np.reshape(gas_fsh, (nlat, nlon))
            gas_fsherr = np.reshape(gas_fsherr, (nlat, nlon)) 
            gas_fshfit = np.reshape(gas_fshfit, (nlat, nlon))
            gas_errfshfit = np.reshape(gas_errfshfit, (nlat, nlon))

            gas_abunerr = np.reshape(gas_abunerr, (nlevel, nlat, nlon))

    if over_axis=="2D": return  radiance, wavenumb, rad_err, rad_fit, temp_prior_mre, temp_errprior_mre, \
                                temp_fit_mre, temp_errfit_mre, aer_mre, aer_err,  aer_fit, fit_err, \
                                gas_scale, gas_scaleerr,  gas_scalefit, gas_errscalefit, \
                                gas_vmr, gas_vmrerr,  gas_vmrfit, gas_errvmrfit,\
                                gas_fsh, gas_fsherr,  gas_fshfit, gas_errfshfit,\
                                gas_abunerr,\
                                latitude, nlat, longitude, nlon

    if over_axis=="latitude" or "longitude": return radiance, wavenumb, rad_err, rad_fit, temp_prior_mre, \
                                                    temp_errprior_mre, temp_fit_mre, temp_errfit_mre, \
                                                    aer_mre, aer_err,  aer_fit, fit_err, \
                                                    gas_scale, gas_scaleerr,  gas_scalefit, gas_errscalefit, \
                                                    gas_vmr, gas_vmrerr, gas_vmrfit, gas_errvmrfit,\
                                                    gas_fsh, gas_fsherr, gas_fshfit, gas_errfshfit, \
                                                    gas_abunerr, \
                                                    coor_axis, ncoor 

def ReadprfFiles(filepath, over_axis):
    """ Read retrieved temperature and gases output profiles for all .prf files """

    if over_axis=="latitude":
        # Retrieve latitude-core_number correspondance
        coor_core, ncoor, nlevel, ngas = RetrieveLatitudeFromCoreNumber(f"{filepath}")
    elif over_axis=="longitude":
        # Retrieve longitude-core_number correspondance
        coor_core, ncoor, nlevel, ngas = RetrieveLongitudeFromCoreNumber(f"{filepath}")
    elif over_axis=="2D":
        # Retrieve latitude-longitude-core_number correspondance
        coor_core, latitude, longitude, ncoor, nlevel, ngas = RetrieveLongitudeLatitudeFromCoreNumber(f"{filepath}")
    
    # Create output arrays
    height      = np.empty((nlevel, ncoor))          # height array which varies with latitude
    pressure    = np.empty((nlevel))                # pressure array 
    temperature = np.empty((nlevel, ncoor))          # temperature array with profiles for all latitude + pressure profiles ([:,0])
    gases       = np.empty((nlevel, ncoor, ngas))    # gases array with profiles for all latitude + pressure profiles ([:,0])
    gases_id    = np.empty((ngas))
    # Initialize output arrays
    height.fill(np.nan)          
    pressure.fill(np.nan) 
    temperature.fill(np.nan)
    gases.fill(np.nan)
    gases_id.fill(np.nan)
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
        coor_core, ncoor, nlevel, _ = RetrieveLatitudeFromCoreNumber(f"{filepath}")
    elif over_axis=="longitude":
        # Retrieve longitude-core_number correspondance
        coor_core, ncoor, nlevel, _ = RetrieveLongitudeFromCoreNumber(f"{filepath}")
    elif over_axis=="2D":
        # Retrieve latitude-longitude-core_number correspondance
        coor_core, latitude, longitude, ncoor, nlevel, _ = RetrieveLongitudeLatitudeFromCoreNumber(f"{filepath}")

    # Initialize output arrays
    aerosol = np.empty((nlevel, ncoor)) # aerosol array with profiles for all latitude
    height  = np.empty((nlevel, ncoor)) # height array with profiles for all latitude 
    # Initialize output arrays
    aerosol.fill(np.nan)
    height.fill(np.nan)
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

def ReadAerFromMreFiles(filepath, over_axis):
    """ Read radiance retrieval outputs for all .mre files """

    if over_axis=="latitude":
        # Retrieve latitude-core_number correspondance
        coor_core, ncoor, nlevel, _ = RetrieveLatitudeFromCoreNumber(f"{filepath}") 
    elif over_axis=="longitude":
        # Retrieve longitude-core_number correspondance
        coor_core, ncoor, nlevel, _ = RetrieveLongitudeFromCoreNumber(f"{filepath}")
    elif over_axis=="2D":
        # Retrieve latitude-longitude-core_number correspondance
        coor_core, latitude, longitude, ncoor, nlevel, _ = RetrieveLongitudeLatitudeFromCoreNumber(f"{filepath}")

    # Create output arrays with suitable dimensions
    aerosol = np.empty((ncoor))
    aer_err = np.empty((ncoor)) 
    aer_fit = np.empty((ncoor))
    fit_err = np.empty((ncoor))
    # Initialize output arrays 
    aerosol.fill(np.nan)
    aer_err.fill(np.nan) 
    aer_fit.fill(np.nan)
    fit_err.fill(np.nan)
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
            # Save aerosol's data
            aer_ind = lines.index('          -1           0           3\n')
            aer_param = lines[aer_ind+3].split()
            aerosol[icoor] = float(aer_param[2])
            aer_err[icoor] = aer_param[3]
            aer_fit[icoor] = aer_param[4]
            fit_err[icoor] = aer_param[5]
               
    if over_axis=="2D":
        # Create suitable dimensions for output arrays
        nlat = len(latitude)
        nlon = len(longitude)
        # Reshape the output arrays in case of 2D retrieval
        aerosol = np.reshape(aerosol, (nlat, nlon))
        aer_err  = np.reshape(aer_err, (nlat, nlon))
        aer_fit  = np.reshape(aer_fit, (nlat, nlon))
        fit_err = np.reshape(fit_err, (nlat, nlon))

    if over_axis=="2D": return aerosol, aer_err, aer_fit, fit_err, latitude, nlat, longitude, nlon
    if over_axis=="latitude" or "longitude": return aerosol, aer_err, aer_fit, fit_err, coor_axis, ncoor


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


def ReadContributionFunctions(filepath, over_axis):

    def read_s_matrices(f, nlines):
        lines = []
        for _ in range(nlines):
            line = f.readline()
            split_line = line.split()
            lines.append(split_line)
        return lines

    def kk_matrix(f, nlines):
        lines = []
        for _ in range(nlines):
            line = f.readline()
            split_line = line.split()
            lines.append(split_line)
            if len(lines) == ny:
                return lines

    if over_axis == "latitude":
        # Retrieve latitude-core_number correspondance
        coor_core, ncoor, nlevel, _ = RetrieveLatitudeFromCoreNumber(f"{filepath}")
    elif over_axis =="longitude":
        # Retrieve longitude-core_number correspondance
        coor_core, ncoor, nlevel, _ = RetrieveLongitudeFromCoreNumber(f"{filepath}")
    elif over_axis=="2D":
        # Retrieve latitude-longitude-core_number correspondance
        coor_core, latitude, longitude, ncoor, nlevel, _ = RetrieveLongitudeLatitudeFromCoreNumber(f"{filepath}")
    
    # Create a latitude or longitude array for plotting
    coor_axis = np.asarray(coor_core[:,1], dtype='float')
    weighting_function = np.empty((ncoor, nlevel, 11))
    # Read pressure array in the nemesis.prf files
    if over_axis=='latitude' or over_axis=='longitude': 
        _, _, _, _, pressure, _, _, _, _ = ReadprfFiles(filepath, over_axis)
    elif over_axis=='2D':
        _, _, _, _, _, pressure, _, _, _, _, _, _ = ReadprfFiles(filepath, over_axis)
    # pressure = pressure / 1013.25
    # Read and save contribution function profiles
    for icoor in range(ncoor):
        ifile = int(coor_core[icoor, 0])       
        # Reading the unformatted Fortran file (unformatted... What idea?!) 
        with open(f"{filepath}/core_{ifile}/nemesis.cov", 'r') as f:
            #  Read no. of levels (NPRO) and number of variables (NVAR)
            line = f.readline()
            npro, nvar = line.split()
            npro, nvar = int(npro), int(nvar)

            # Read variable IDs and parameters
            varident = np.empty((nvar, 3))
            varparam = np.empty((nvar, 5))

            for ivar in range(nvar):
                line = f.readline()
                varident[ivar, :] = line.split()
                line = f.readline()
                varparam[ivar, :] = line.split()

            # Read in number of ??? (NX) and number of ??? (NY) 
            line = f.readline()
            nx, ny = line.split()
            nx, ny = int(nx), int(ny)

            # Read covariance matrices
            sa, sm, sn, st = [np.empty((nx, nx)) for _ in range(4)]
            nlines = int(nx / 5) # Look I'm sorry okay, this is not my fault
            for ix in range(nx):       
                lines = read_s_matrices(f=f, nlines=nlines)
                sa[ix, :] = [element for line in lines for element in line]
                lines = read_s_matrices(f=f, nlines=nlines)
                sm[ix, :] = [element for line in lines for element in line]
                lines = read_s_matrices(f=f, nlines=nlines)
                sn[ix, :] = [element for line in lines for element in line]
                lines = read_s_matrices(f=f, nlines=nlines)
                st[ix, :] = [element for line in lines for element in line]
            
            # Read ??? matrices
            aa = np.empty((nx, nx))
            nlines = int(nx / 5) # Look I'm sorry okay, this is not my fault
            for ix in range(nx):       
                lines = read_s_matrices(f=f, nlines=nlines)
                aa[ix, :] = [element for line in lines for element in line]
            
            dd = np.empty((nx, ny))
            for iy in range(ny):  
                lines = read_s_matrices(f=f, nlines=nlines)
                dd[:, iy] = [element for line in lines for element in line]
            
            kk = np.empty((ny, nx))
            nlines = 3 # Look I'm sorry okay, this is not my fault
            for ix in range(nx):
                lines = read_s_matrices(f=f, nlines=nlines)
                kk[:, ix] = [element for line in lines for element in line] 
            
            kt = np.transpose(kk)

            se = np.empty(ny)
            nlines = 3 # Look I'm sorry okay, this is not my fault
            lines = read_s_matrices(f=f, nlines=nlines)
            se[:] = [element for line in lines for element in line]

        # Plot weighting functions
        wf = np.empty((npro, ny))

        for iy in range(ny):
            wf[:npro, iy] = abs(kt[:npro, iy]) / max(kt[:npro, iy])
        
            weighting_function[icoor, :, iy] = wf[:npro, iy]


    return weighting_function, ny, coor_axis, nlevel, pressure

def ReadAllForAuroraOverTime(filepath, gas_name):

    # Retrieve the number of night from the number of core subdirectories
    night_core, nnight, nlevel, ngas = RetrieveNightFromCoreNumber(f"{filepath}/core")


    # Create outputs arrays with suitable dimensions
    # from log_icore files
    chisquare = np.empty(nnight)
    # from .mre files 
    radiance = np.empty((Globals.nfilters, nnight))
    rad_err = np.empty((Globals.nfilters, nnight))
    rad_fit = np.empty((Globals.nfilters, nnight))
    wavenumb = np.empty((Globals.nfilters, nnight))
    temp_prior_mre = np.empty((nlevel, nnight))
    temp_errprior_mre = np.empty((nlevel, nnight))
    temp_fit_mre = np.empty((nlevel, nnight))
    temp_errfit_mre = np.empty((nlevel, nnight))
    aer_mre = np.empty((nnight))
    aer_err = np.empty((nnight)) 
    aer_fit = np.empty((nnight))
    fit_err = np.empty((nnight))
    ngases = len(gas_name)
    if ngases > 1:
        gas_scale = np.empty((ngases, nnight))
        gas_scaleerr = np.empty((ngases, nnight)) 
        gas_scalefit = np.empty((ngases, nnight))
        gas_errscalefit = np.empty((ngases, nnight))

    else:
        gas_scale = np.empty((nnight))
        gas_scaleerr = np.empty((nnight)) 
        gas_scalefit = np.empty((nnight))
        gas_errscalefit = np.empty((nnight))

    # from .prf files
    height      = np.empty((nlevel, nnight))          # height array which varies with latitude
    pressure    = np.empty((nlevel))                  # pressure array 
    temperature = np.empty((nlevel, nnight))          # temperature array with profiles for all latitude + pressure profiles ([:,0])
    gases       = np.empty((nlevel, nnight, ngas))    # gases array with profiles for all latitude + pressure profiles ([:,0])
    gases_id    = np.empty((ngas))
    #from aerosol.prf files 
    h_prf      = np.empty((nlevel, nnight))           # height array which varies with latitude
    aer_prf      = np.empty((nlevel, nnight))         # aerosol profiles

    #Initialize to avoid strange values 
    chisquare.fill(np.nan)
    radiance.fill(np.nan)
    rad_err.fill(np.nan)
    rad_fit.fill(np.nan)
    wavenumb.fill(np.nan)
    temp_prior_mre.fill(np.nan)
    temp_errprior_mre.fill(np.nan)
    temp_fit_mre.fill(np.nan)
    temp_errfit_mre.fill(np.nan)
    aer_mre.fill(np.nan)
    aer_err.fill(np.nan) 
    aer_fit.fill(np.nan)
    fit_err.fill(np.nan)
    gas_scale.fill(np.nan)
    gas_scaleerr.fill(np.nan) 
    gas_scalefit.fill(np.nan)
    gas_errscalefit.fill(np.nan)
    height.fill(np.nan)          
    pressure.fill(np.nan) 
    temperature.fill(np.nan)
    gases.fill(np.nan)
    gases_id.fill(np.nan)
    aer_prf.fill(np.nan)
    h_prf.fill(np.nan)

    # Create a latitude or longitude array for plotting
    time_axis = np.asarray(night_core[:,1], dtype='float')
    # Read and save retrieved profiles
    for inight in range(nnight):
        ifile = int(night_core[inight, 1])
        ## Read ChiSquare data
        with open(f"{filepath}/core_{ifile}/log_{ifile}") as f:
            # Read file
            lines = f.readlines()
            for iline, line in enumerate(lines):
                l = line.split()
                # Identifying the last line of chisq/ny in the current log file
                if 'chisq/ny' and 'equal' in l:
                    tmp = line.split(':')
                    chisq = tmp[-1]
                    chisquare[inight] = chisq

        ## Read radiance data
        with open(f"{filepath}/core_{ifile}/nemesis.mre") as f:
            # Read file
            lines = f.readlines()
            # Get dimensions
            param = lines[1].split()
            ispec = int(param[0])
            ngeom = int(param[1])
            nx    = int(param[3])

            # Save radiance data
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
                if ((time_axis[inight] < 6)):
                    wavenumb[:, inight] = data_arr[:, 1]
                    radiance[:, inight] = data_arr[:, 2]
                    rad_err[:, inight]  = data_arr[:, 3]
                    rad_fit[:, inight]  = data_arr[:, 5]
                elif (time_axis[inight] > 6):
                    wavenb = data_arr[:, 1]
                    rad = data_arr[:, 2]
                    err  = data_arr[:, 3]
                    fit  = data_arr[:, 5]
                    
                    if Globals.nfilters ==12:
                        for ifilt in range(Globals.nfilters-1):
                            if ifilt <= 5:
                                wavenumb[ifilt, inight] = wavenb[ifilt]
                                radiance[ifilt, inight] = rad[ifilt]
                                rad_err[ifilt, inight]  = err[ifilt]
                                rad_fit[ifilt, inight]  = fit[ifilt]
                            else:
                                wavenumb[ifilt+1, inight] = wavenb[ifilt]
                                radiance[ifilt+1, inight] = rad[ifilt]
                                rad_err[ifilt+1, inight]  = err[ifilt]
                                rad_fit[ifilt+1, inight]  = fit[ifilt]
                    elif Globals.nfilters ==13:
                        for ifilt in range(Globals.nfilters-2):
                            if ifilt <= 5:
                                wavenumb[ifilt, inight] = wavenb[ifilt]
                                radiance[ifilt, inight] = rad[ifilt]
                                rad_err[ifilt, inight]  = err[ifilt]
                                rad_fit[ifilt, inight]  = fit[ifilt]
                            else:
                                wavenumb[ifilt+2, inight] = wavenb[ifilt]
                                radiance[ifilt+2, inight] = rad[ifilt]
                                rad_err[ifilt+2, inight]  = err[ifilt]
                                rad_fit[ifilt+2, inight]  = fit[ifilt]
            else:
                wavenb = data_arr[:, 1]
                rad = data_arr[:, 2]
                err  = data_arr[:, 3]
                fit  = data_arr[:, 5]
                if len(wavenb) == Globals.nfilters:
                    wavenumb[:, inight] = data_arr[:, 1]
                    radiance[:, inight] = data_arr[:, 2]
                    rad_err[:, inight]  = data_arr[:, 3]
                    rad_fit[:, inight]  = data_arr[:, 5]
                elif len(wavenb) == 9:
                    for ifilt in range(Globals.nfilters-2):
                        if ifilt < 8:
                            wavenumb[ifilt, inight] = wavenb[ifilt]
                            radiance[ifilt, inight] = rad[ifilt]
                            rad_err[ifilt, inight]  = err[ifilt]
                            rad_fit[ifilt, inight]  = fit[ifilt]
                        else:
                            wavenumb[ifilt+2, inight] = wavenb[ifilt]
                            radiance[ifilt+2, inight] = rad[ifilt]
                            rad_err[ifilt+2, inight]  = err[ifilt]
                            rad_fit[ifilt+2, inight]  = fit[ifilt]
                elif len(wavenb) == 2:
                    idata = 0
                    for ifilt in [1,5]:
                        wavenumb[ifilt, inight] = wavenb[idata]
                        radiance[ifilt, inight] = rad[idata]
                        rad_err[ifilt, inight]  = err[idata]
                        rad_fit[ifilt, inight]  = fit[idata]
                        idata +=1
            
            # Save temperature data

            newline, filedata = [], []
            for iline, line in enumerate(lines[5+ngeom+6:5+ngeom+6+nlevel]):
                l = line.split()
                [newline.append(il) for il in l]
                # Store data 
                filedata.append(newline)
                # Reset temporary variables
                newline = []
            data_arr = np.asarray(filedata, dtype='float')
            temp_prior_mre[:, inight] = data_arr[:, 2]
            temp_errprior_mre[:, inight] = data_arr[:, 3]
            temp_fit_mre[:, inight] = data_arr[:, 4]
            temp_errfit_mre[:, inight] = data_arr[:, 5]


            # Save aerosol data
            if '          -1           0           3\n' in lines:
                aer_ind = lines.index('          -1           0           3\n')
                aer_param = lines[aer_ind+3].split()
                aer_mre[inight] = float(aer_param[2])
                aer_err[inight] = aer_param[3]
                aer_fit[inight] = aer_param[4]
                fit_err[inight] = aer_param[5]
                
            # Save gases data
            if ngases > 1:
                for igas in range(ngases):
                    _, idgas = RetrieveGasesNames(gas_name=gas_name[igas], gas_id=False)
                    # Scaling retrieval case:
                    if f"          {idgas}           0           3\n" in lines:
                        gas_ind = lines.index(f"          {idgas}           0           3\n")
                        gas_param = lines[gas_ind+3].split()
                        gas_scale[igas, inight] = float(gas_param[2])
                        gas_scaleerr[igas, inight] = float(gas_param[3])
                        gas_scalefit[igas, inight] = float(gas_param[4])
                        gas_errscalefit[igas, inight] = float(gas_param[5])
                else:
                    _, idgas = RetrieveGasesNames(gas_name=gas_name, gas_id=False)
                    # Scaling retrieval case:
                    if f"          {idgas}           0           3\n" in lines:
                        gas_ind = lines.index(f"          {idgas}           0           3\n")
                        gas_param = lines[gas_ind+3].split()
                        gas_scale[inight] = float(gas_param[2])
                        gas_scaleerr[inight] = gas_param[3]
                        gas_scalefit[inight] = gas_param[4]
                        gas_errscalefit[inight] = gas_param[5]
          
        ## Read Temperature, Gases data
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
            height[:, inight]       = data_arr[:, 0]            # height profile
            pressure[:]             = data_arr[:, 1] * 1013.25  # pressure profile converted in mbar
            temperature[:, inight]  = data_arr[:, 2]            # temperature profile
            for igas in range(ngas):
                gases[:, inight, igas] = data_arr[:,igas+3]
        
        ## Read aerosol from aerosol.prf files
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
            h_prf[:, inight] = aer[:,0]
            aer_prf[:, inight]  = aer[:,1]
    
    return nnight, time_axis, chisquare, radiance, rad_err, rad_fit, wavenumb, \
            temp_prior_mre, temp_errprior_mre, temp_fit_mre, temp_errfit_mre, \
            aer_mre, aer_err,  aer_fit, fit_err, \
            gas_scale, gas_scaleerr,  gas_scalefit, gas_errscalefit,\
            height, pressure, temperature, \
            gases, gases_id, aer_prf, h_prf



def ReadPreviousWork(fpath, ncore, corestart, namerun, over_axis):

# Initialize local variables
    lat_core = np.empty((ncore, 2)) # 2D array containing latitude and core directory number
    # Read all .prf files through all core directories
    for ifile in range(ncore):
        filename = f"{fpath}/prffiles/{namerun}_{ifile+corestart}.prf"
        with open(filename) as f:
            # Read header contents
            lines = f.readlines()
            # Save header information
            prior_param = lines[1].split()
            lat_core[ifile, 0] = int(ifile+corestart)                    # Store core number 
            lat_core[ifile, 1] = float(prior_param[1])    # Store corresponding latitude value
            nlevel             = int(prior_param[2])
            ngas               = int(prior_param[3])
    # Sorted on latitude values
    lat_core = sorted(lat_core, key=operator.itemgetter(1))
    lat_core = np.asarray(lat_core, dtype='float')



# Create output arrays
    height      = np.empty((nlevel, ncore))          # height array which varies with latitude
    pressure    = np.empty((nlevel))                # pressure array 
    temperature = np.empty((nlevel, ncore))          # temperature array with profiles for all latitude + pressure profiles ([:,0])
    gases       = np.empty((nlevel, ncore, ngas))    # gases array with profiles for all latitude + pressure profiles ([:,0])
    gases_id    = np.empty((ngas))
    # Initialize output arrays
    height.fill(np.nan)          
    pressure.fill(np.nan) 
    temperature.fill(np.nan)
    gases.fill(np.nan)
    gases_id.fill(np.nan)
    # Create a latitude or longitude array for plotting
    coor_axis = np.asarray(lat_core[:,1], dtype='float')
    # Read and save retrieved profiles
    for icoor in range(ncore):
        ifile = int(lat_core[icoor, 0])
        with open(f"{fpath}/prffiles/{namerun}_{ifile}.prf") as f:
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
    
    # if over_axis=="2D":
    #     # Create suitable dimensions for output arrays
    #     nlat = len(latitude)
    #     nlon = len(longitude)
    #     # Reshape the output arrays in case of 2D retrieval
    #     height      = np.reshape(height, (nlevel, nlat, nlon))
    #     temperature = np.reshape(temperature, (nlevel, nlat, nlon))
    #     gases       = np.reshape(gases, (nlevel, nlat, nlon, ngas))
    
    # if over_axis=="2D": return temperature, gases, latitude, longitude, height, pressure, ncore, nlevel, nlat, nlon, ngas, gases_id
    if over_axis=="latitude" or "longitude": return temperature, gases, coor_axis, height, pressure, ncore, nlevel, ngas, gases_id