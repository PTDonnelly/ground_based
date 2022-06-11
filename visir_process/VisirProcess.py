"""
PTD 01/06/22: Code to calibrate VISIR cylindrical maps and
              prepare spectra for input into NEMESIS.
"""

def main():
    import time
    import numpy as np
    start = time.time()
    
    from FindFiles import FindFiles
    from RegisterMaps import RegisterMaps
    from CreateMeridProfiles import CreateMeridProfiles
    from CalibrateMeridProfiles import CalibrateMeridProfiles
    from WriteProfiles import WriteProfiles
    from PlotProfiles import PlotProfiles
    from PlotMaps import PlotMaps
    from WriteSpx import WriteSpx

    
    ##### Define global inputs #####
    files       = FindFiles(mode='images')           # Point to location of all input observations
    Nfiles      = len(files)
    # Flags
    calc        = 0                                   # (0) Calculate meridional profiles, (1) read stored profiles
    save        = 1                                   # (0) Do not save (1) save meridional profiles
    plot        = 1                                   # (0) Do not plot (1) plot meridional profiles
    maps        = 0                                   # (0) Do not plot (1) plot cylindrical maps
    spx         = 0                                   # (0) Do not write (1) do write spxfiles for NEMESIS input
    
    if calc == 0:
        # Steps 1-3: Generate nested dictionaries containing spatial and spectral information of each cylindrical map
        print('Registering maps...')
        spectrum, wavelength, wavenumber, LCMIII = RegisterMaps(files)

        # Steps 4-5: Generate average meridional profiles for each observation and each filter
        print('Calculating meridional profiles...')
        singles, spectrals = CreateMeridProfiles(Nfiles, spectrum, LCMIII)
        
        # Steps 6-7: Generate calibrated versions of the profiles from Steps 4 and 5
        print('Calibrating meridional profiles...')
        calsingles, calspectrals, ksingles, kspectrals = CalibrateMeridProfiles(Nfiles, singles, spectrals, wavenumber)

        # Step 8: Store all cmap profiles and calibration parameters
        if save == 1:
            WriteProfiles(files, calsingles, calspectrals, ksingles, kspectrals)

    ###

    # Plot meridional profiles (optionally read stored numpy arrays from Step 8)
    if plot == 1:
        if calc == 0:
            PlotProfiles(calsingles, calspectrals, ksingles, kspectrals, wavenumber)
        if calc == 1:
            # Point to stored meridional profiles and calibration coefficients
            profiles1 = FindFiles(mode='singles')
            profiles2 = FindFiles(mode='spectrals')
            coeffs1   = FindFiles(mode='ksingles')
            coeffs2   = FindFiles(mode='kspectrals')
            PlotProfiles(singles=profiles1, spectrals=profiles2, ksingles=coeffs1, kspectrals=coeffs2, wavenumber=False)

    # Plot cylindrical maps (optionally read stored numpy arrays from Step 8)
    if maps == 1:
        if calc == 0:
            PlotMaps(files, ksingles, kspectrals)
        if calc == 1:
            # Point to stored calibration coefficients
            coeffs1   = FindFiles(mode='ksingles')
            coeffs2   = FindFiles(mode='kspectrals')
            PlotMaps(filesksingles=coeffs1, kspectrals=coeffs2)
    
    # Generate spectral inputs for NEMESIS (optionally read stored numpy arrays from Step 8)
    if spx == 1:
        if calc == 0:
            WriteSpx(calspectrals)
        if calc == 1:
            # Point to stored individual meridional profiles
            profiles = FindFiles(mode='spectrals')
            WriteSpx(spectrals=profiles)

    end = time.time()
    print(f"Elapsed time: {np.round(end-start, 3)} s")
    print(f"Time per file: {np.round((end-start)/len(files), 3)} s")

if __name__ == '__main__':
    main()