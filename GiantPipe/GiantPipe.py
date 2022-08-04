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
    from Plot.PlotMaps import PlotMaps, PlotZoomMaps
    from Plot.PlotPoles import PlotPolesFromGlobal
    from Plot.PlotPseudoWindShear import PlotPseudoWindShear, PlotCompositePseudoWindShear
    from Plot.PlotBrightnessTemperatureProf import PlotCompositeTBprofile
    from Plot.PlotPriorProfiles import PlotTemperaturePriorProfiles, PlotAerosolPriorProfiles
    from Plot.PlotRetrievalOutputs import PlotRetrievedTemperature, PlotRetrievedTemperatureProfile, PlotRetrievedRadiance, PlotRetrievedAerosolProfile, PlotRetrievedRadianceMeridian
    from Read.ReadNpy import ReadNpy
    # from Write.WriteProfiles import WriteMeridProfiles
    # from Write.WriteProfiles import WriteCTLProfiles
    # from Write.WriteSpx import WriteMeridSpx
    # from Write.WriteSpx import WriteCTLSpx
    from Plot.PlotSpx import PlotSpxMerid
    
    # Define flags to configure pipeline
    source      = 3     # Raw data (to be calibrated) (0), or calibrated data (1), pre-calculated .npy profiles (2)
    # Binning
    bin_cmerid  = 1     # Use central meridian binning scheme (1), or not (0)
    bin_ctl     = 0     # Use centre-to-limb binning scheme (1), or not (0)
    # Output
    save        = 1     # Store calculated profiles to local files (1), or not (0)
    plotting    = 1     # Plot calculated profiles (1), or not (0)
    mapping     = 1     # Plot maps of observations or retrieval (1), or not (0)
    spx         = 0     # Write spxfiles as spectral input for NEMESIS (1), or not (0)
    retrieval   = 0     # Plot retrieval outputs

    # ############################################################
    # # Perform geometric registration and radiometric calibration
    # # of cylindrically-mapped data through central meridian binning
    # # and comparison to spacecraft data, then output calculated
    # # profiles and coefficients. Recommended as a first pass. For 
    # # plotting, mapping and spxing use calibrated data and .npy files
    # ############################################################

    if source == 0:
        from Binning.CentralMerid import BinCentralMerid
        from Calibrate.CentralMerid import CalCentralMerid
        from Calibrate.CylindricalMaps import CalCylindricalMaps
        from Plot.PlotProfiles import PlotMeridProfiles
        from Write.WriteProfiles import WriteMeridProfiles
        from Write.WriteSpx import WriteMeridSpx
    
        dataset='2022July'

        # Point to location of observations
        files       = FindFiles(mode='images_raw', dataset=dataset)
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
        WriteMeridProfiles(dataset=dataset, files=files, singles=singles, spectrals=spectrals, ksingles=ksingles, kspectrals=kspectrals)


    # ############################################################
    # # Read in calibrated cylindrical maps of radiance, emission
    # # angle (and optionally Doppler velocity) and create profiles
    # # depending on the chosen binning scheme.
    # ############################################################

    if source == 1:
        from Binning.CentralMerid import BinCentralMerid
        from Binning.CentreToLimb import BinCentreToLimb
        from Calibrate.CentralMerid import CalCentralMerid
        from Write.WriteProfiles import WriteMeridProfiles
        from Write.WriteProfiles import WriteCTLProfiles
        from Plot.PlotProfiles import PlotMeridProfiles

        dataset = '2018May'
        # Point to location of observations
        files       = FindFiles(mode='images_raw', dataset=dataset)
        nfiles      = len(files)

        # Generate arrays containing spatial and spectral information of each cylindrical map
        spectrum, wavelength, wavenumber, LCMIII = RegisterMaps(files=files)

        if bin_cmerid == 1:
            # Execute the central meridian binning scheme
            singles, spectrals = BinCentralMerid(nfiles=nfiles, spectrum=spectrum, LCMIII=LCMIII)

            singles, spectrals, ksingles, kspectrals = CalCentralMerid(nfiles=nfiles, singles=singles, spectrals=spectrals, wavenumber=wavenumber)

            if save == 1:
                # Store calculated profiles
                WriteMeridProfiles(dataset=dataset, files=files, singles=singles, spectrals=spectrals, ksingles=ksingles, kspectrals=kspectrals)

            if plotting == 1:
                # Create plots
                PlotMeridProfiles(dataset=dataset, files=files, singles=singles, spectrals=spectrals)
                PlotSpxMerid(dataset='2018May')

        # if bin_ctl == 1:
        #     # Execute the centre-to-limb binning scheme
        #     output = BinCentreToLimb(nfiles, spectrum)

        #     if save == 1:
        #         # Store calculated profiles
        #         WriteCTLProfiles(files=files)


    # ############################################################
    # # Read in pre-calculated and locally-stored numpy arrays (.npy)
    # # of radiance
    # ############################################################

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
        dataset = '2022July'

    ### Point to location of observations
        files       = FindFiles(dataset=dataset, mode='images_calib')
    ## Plot cylindrical maps
        if bin_cmerid == 0:
            # Create plots and save global maps into npy arrays
            PlotMaps(dataset, files, spectrals)
        if bin_cmerid == 1:
            # Read in individual calibration coefficients
            _, spectrals, _, _ = ReadNpy(dataset=dataset, return_singles=False, return_spectrals=True, return_ksingles=False, return_kspectrals=False)
            # Create plots and save global maps into npy arrays
            PlotMaps(dataset, files, spectrals)
    # Plot pole maps from global maps npy arrays
        PlotZoomMaps(dataset=dataset, central_lon=180, lat_target=-20, lon_target=285, lat_window=15, lon_window=30)
        PlotPolesFromGlobal(dataset=dataset)
        # PlotPseudoWindShear(dataset=dataset)
        # PlotCompositePseudoWindShear(dataset=dataset)
        # PlotCompositeTBprofile(dataset=dataset)
        
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

    ###############################################################
    # Read in retrieved output files from NEMESIS and create
    # meridian and vertical profiles of temperature, aerosols, etc.
    ###############################################################

    if retrieval == 1:
        # PlotTemperaturePriorProfiles()
        # PlotAerosolPriorProfiles()
        # PlotRetrievedTemperature()
        # PlotRetrievedTemperatureProfile()
        # PlotRetrievedRadiance()
        # PlotRetrievedAerosolProfile()
        PlotRetrievedRadianceMeridian()
        


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