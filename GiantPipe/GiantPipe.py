"""
PTD 01/06/22: Code to calibrate VISIR cylindrical maps and
              prepare spectra for input into NEMESIS.
"""

def main():
    from FindFiles import FindFiles
    from RegisterMaps import RegisterMaps
    # from Binning.CentralMerid import BinCentralMerid
    # from Binning.CentreToLimb import BinCentreToLimb
    # from Calibrate.CentralMerid import CalCentralMerid
    # from Calibrate.CylindricalMaps import CalCylindricalMaps
    # from Plot.PlotProfiles import PlotMeridProfiles
    from Plot.PlotMaps import PlotMaps
    from Read.ReadNpy import ReadNpy
    # from Write.WriteProfiles import WriteMeridProfiles
    # from Write.WriteProfiles import WriteCTLProfiles
    # from Write.WriteSpx import WriteMeridSpx
    # from Write.WriteSpx import WriteCTLSpx
    
    # Define flags to configure pipeline
    source      = 0     # Raw data (to be calibrated) (0), or calibrated data (1), pre-calculated .npy profiles (2)
    # Binning
    bin_cmerid  = 0     # Use central meridian binning scheme (1), or not (0)
    bin_ctl     = 0     # Use centre-to-limb binning scheme (1), or not (0)
    # Output
    save        = 0     # Store calculated profiles to local files (1), or not (0)
    plotting    = 0     # Plot calculated profiles (1), or not (0)
    mapping     = 0     # Plot maps of observations or retrieval (1), or not (0)
    spx         = 0     # Write spxfiles as spectral input for NEMESIS (1), or not (0)


    ############################################################
    # Perform geometric registration and radiometric calibration
    # of cylindrically-mapped data through central meridian binning
    # and comparison to spacecraft data, then output calculated
    # profiles and coefficients. Recommended as a first pass. For 
    # plotting, mapping and spxing use calibrated data and .npy files
    ############################################################

    if source == 0:
        from Binning.CentralMerid import BinCentralMerid
        from Calibrate.CentralMerid import CalCentralMerid
        from Calibrate.CylindricalMaps import CalCylindricalMaps
        from Plot.PlotProfiles import PlotMeridProfiles
        from Write.WriteProfiles import WriteMeridProfiles
        from Write.WriteSpx import WriteMeridSpx
    
        # Point to location of observations
        files       = FindFiles(mode='images')
        nfiles      = len(files)

        # Steps 1-3: Generate arrays containing spatial and spectral information of each cylindrical map
        spectrum, wavelength, wavenumber, LCMIII = RegisterMaps(files=files)

        # Steps 4-5: Generate average meridional profiles for each observation and each filter
        rawsingles, rawspectrals = BinCentralMerid(nfiles=nfiles, spectrum=spectrum, LCMIII=LCMIII)
        
        # Steps 6-7: Generate calibrated versions of the profiles from Steps 4 and 5
        singles, spectrals, ksingles, kspectrals = CalCentralMerid(nfiles=nfiles, singles=rawsingles, spectrals=rawspectrals, wavenumber=wavenumber)

        # Step 8: Calibrate cylindrical maps using calculated calibration coefficients
        CalCylindricalMaps(files, ksingles)

        # Step 9: Store calculated profiles and calibration parameters
        WriteMeridProfiles(files=files, singles=singles, spectrals=spectrals, ksingles=ksingles, kspectrals=kspectrals)


    ############################################################
    # Read in calibrated cylindrical maps of radiance, emission
    # angle (and optionally Doppler velocity) and create profiles
    # depending on the chosen binning scheme.
    ############################################################

    if source == 1:
        from Binning.CentreToLimb import BinCentreToLimb
        from Write.WriteProfiles import WriteCTLProfiles

        # Point to location of observations
        files       = FindFiles(mode='images')
        nfiles      = len(files)

        # Generate arrays containing spatial and spectral information of each cylindrical map
        spectrum, wavelength, wavenumber, LCMIII = RegisterMaps(files=files)

        if bin_cmerid == 1:
            # Execute the central meridian binning scheme
            singles, spectrals = BinCentralMerid(nfiles=nfiles, spectrum=spectrum, LCMIII=LCMIII)

            if save == 1:
                # Store calculated profiles
                WriteMeridProfiles(files=files, singles=singles, spectrals=spectrals, ksingles=False, kspectrals=False)

            if plotting == 1:
                # Create plots
                PlotMeridProfiles(singles=singles, spectrals=spectrals, ksingles=ksingles, kspectrals=kspectrals, wavenumber=wavenumber)
        
        if bin_ctl == 1:
            # Execute the centre-to-limb binning scheme
            output = BinCentreToLimb(nfiles, spectrum)

            if save == 1:
                # Store calculated profiles
                WriteCTLProfiles(files=files)


    ############################################################
    # Read in pre-calculated and locally-stored numpy arrays (.npy)
    # of radiance
    ############################################################

    if source == 2:
        from Plot.PlotProfiles import PlotMeridProfiles
        if bin_cmerid == 1:

            # Read in profiles and coefficients
            singles, spectrals, ksingles, kspectrals = ReadNpy(return_singles=True, return_spectrals=True, return_ksingles=True, return_kspectrals=True)

            if save == 1:
                # Store calculated profiles
                WriteMeridProfiles(files=files, singles=singles, spectrals=spectrals, ksingles=None, kspectrals=None)

            if plotting == 1:
                # Create plots
                PlotMeridProfiles(singles, spectrals, ksingles, kspectrals, wavenumber=False)


    ############################################################
    # Read in calibrated data, calculated profiles or retrieved
    # maps and and create maps (cylindrical, polar, etc.)
    ############################################################

    if mapping == 1:
    ### Plot cylindrical maps
        if bin_cmerid == 0:
            # Create plots
            PlotMaps(files, spectrals)
        if bin_cmerid == 1:
            # Read in individual calibration coefficients
            _, spectrals, _, _ = ReadNpy(return_singles=False, return_spectrals=True, return_ksingles=True, return_kspectrals=False)
            # Create plots
            PlotMaps(files, spectrals)
    
    ############################################################
    # Read in relevant calculated profiles (bin_cmerid or bin_ctl)
    # and generate spectral inputs for NEMESIS
    ############################################################
    if spx == 1:

        if bin_cmerid == 0:
            # Create spectra
            WriteMeridSpx(spectrals)
        if bin_cmerid == 1:
            # Read in profiles
            _, spectrals, _, _ = ReadNpy(return_singles=False, return_spectrals=True, return_ksingles=False, return_kspectrals=False)
            # Create spectra
            WriteMeridSpx(spectrals)


if __name__ == '__main__':
    import numpy as np
    import time
    import cProfile
    import io
    import pstats
    # # Create profiler
    # pr = cProfile.Profile()
    # # Start profiler
    # pr.enable()

    # Start clock
    start = time.time()

    main()

    # Stop clock
    end = time.time()
    # Print elapsed time
    print(f"Elapsed time: {np.round(end-start, 3)} s")
    print(f"Time per file: {np.round((end-start)/90, 3)} s")
    
    # # Stop profiler
    # pr.disable()
    # # Print profiler output to file
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    # ps.strip_dirs()
    # ps.print_stats()

    # with open('../cProfiler_output.txt', 'w+') as f:
    #     f.write(s.getvalue())