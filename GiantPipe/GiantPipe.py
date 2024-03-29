"""
PTD 01/06/22: Code to calibrate VISIR cylindrical maps and
              prepare spectra for input into NEMESIS.
"""

def main():
    from FindFiles import FindFiles
    from RegisterMaps import RegisterMaps
    from Binning.CentralMerid import BinCentralMerid
    from Binning.CentralParallel import BinCentralPara
    from Binning.CentreToLimb import BinCentreToLimb
    from Binning.CentralRegional import BinRegional
    from Binning.CentralRegionalAverage import BinRegionalAverage
    from Calibrate.CalibrateGBData import CalibrateGBData
    from Calibrate.CentralMerid import CalCentralMerid
    # from Calibrate.CylindricalMaps import CalCylindricalMaps
    from Plot.PlotProfiles import PlotMeridProfiles
    from Plot.PlotProfiles import PlotParaProfiles
    from Plot.PlotProfiles import PlotGlobalSpectrals
    from Plot.PlotProfiles import PlotRegionalMaps
    from Plot.PlotProfiles import PlotRegionalPerNight
    from Read.ReadNpy import ReadCentralMeridNpy
    from Read.ReadNpy import ReadCentralParallelNpy
    from Read.ReadNpy import ReadCentreToLimbNpy
    from Read.ReadNpy import ReadRegionalNpy
    from Read.ReadNpy import ReadRegionalAverageNpy
    from Read.ReadSpx import ReadSpx
    from Write.WriteProfiles import WriteMeridProfiles
    from Write.WriteProfiles import WriteParallelProfiles
    from Write.WriteProfiles import WriteCentreToLimbProfiles
    from Write.WriteProfiles import WriteRegional
    from Write.WriteProfiles import WriteRegionalAverage
    from Write.WriteSpx import WriteMeridSpx
    from Write.WriteSpx import WriteParaSpx
    from Write.WriteSpx import WriteCentreToLimbSpx
    from Write.WriteSpx import WriteRegionalSpx
    from Write.WriteSpx import WriteRegionalAverageSpx

    # Define flags to configure pipeline
    calibrate   = True      # Read raw data and calibrate
    source      = 'fits'     # Source of data: local cmaps ('fits') or local numpy arrays ('npy')
    # Binning
    bin_cmerid  = True     # Use central meridian binning scheme
    bin_cpara   = False     # Use central parallel binning scheme
    bin_ctl     = False     # Use centre-to-limb binning scheme
    bin_region  = False     # Use regional binning scheme (for a zoom 2D retrieval)
    bin_av_region = False   # Use averaged regional binning scheme (for a single profile retrieval)
    # Output
    save        = True      # Store calculated profiles to local files
    plotting    = True      # Plot calculated profiles
    mapping     = False      # Plot maps of observations or retrieval
    spx         = True      # Write spxfiles as spectral input for NEMESIS
    retrieval   = False      # Plot NEMESIS outputs 

    ############################################################
    # Perform geometric registration and radiometric calibration
    # of cylindrically-mapped data through central meridian binning
    # and comparison to spacecraft data. Recommended as a first pass. For 
    # plotting, mapping and spxing use calibrated data and .npy files.
    ############################################################

    if calibrate:
        # Define calibration mode
        mode = 'drm'
        dataset = '2018May_completed'


        # Point to observations
        CalibrateGBData(dataset=dataset, mode=mode+'_files')
        exit()

    ############################################################
    # source = 'fits': read in calibrated cylindrical maps of 
    # radiance, emission angle (and optionally Doppler velocity), 
    # or pre-calculated profiles saved to local numpy files, and
    # create profiles depending on the chosen binning scheme.
    ############################################################
    
    # Define calibration mode
    mode   = 'giantpipe'
    dataset = '2018May_completed'
    per_night = True
    # Point to observations
    files  = FindFiles(dataset=dataset, mode=mode+'_files')
    nfiles = len(files)
    
    if 'fits' in source:

        # Generate arrays containing spatial and spectral information of each cylindrical map
        if bin_cmerid:
            spectrum, wavelength, wavenumber, LCMIII, DATE = RegisterMaps(dataset=dataset, files=files, binning='bin_cmerid')
        if bin_cpara:
            spectrum, wavelength, wavenumber, LCMIII, DATE = RegisterMaps(dataset=dataset, files=files, binning='bin_cpara')
        if bin_region:
            spectrum, wavelength, wavenumber, LCMIII, DATE = RegisterMaps(dataset=dataset, files=files, binning='bin_region')
        if bin_av_region:
            spectrum, wavelength, wavenumber, LCMIII, DATE = RegisterMaps(dataset=dataset, files=files, binning='bin_av_region')

    if bin_cmerid:
        # Execute the central meridian binning scheme
        if 'fits' in source:
            singles, spectrals = BinCentralMerid(nfiles=nfiles, spectrum=spectrum, LCMIII=LCMIII)
        if 'npy' in source:
            singles, spectrals = ReadCentralMeridNpy(dataset=dataset, mode=mode, return_singles=True, return_spectrals=True)

        if save:
            # Store calculated profiles
            WriteMeridProfiles(dataset=dataset, files=files, singles=singles, spectrals=spectrals)

        if plotting:
            # Plot mean central meridian profiles
            PlotMeridProfiles(dataset=dataset, mode=mode, files=files, singles=singles, spectrals=spectrals)
            PlotGlobalSpectrals(dataset=dataset, spectrals=spectrals)

        if spx:
            # Write mean central meridian profiles to spxfile
            WriteMeridSpx(dataset=dataset, mode=mode, spectrals=spectrals)
    
    if bin_cpara:
        # Execute the central parallel binning scheme
        if 'fits' in source:
            singles, spectrals = BinCentralPara(nfiles=nfiles, spectrum=spectrum, LCMIII=LCMIII)
        if 'npy' in source:
            singles, spectrals = ReadCentralParallelNpy(dataset=dataset, mode=mode, return_singles=True, return_spectrals=True)

        if save:
            # Store calculated profiles
            WriteParallelProfiles(dataset=dataset, files=files, singles=singles, spectrals=spectrals)

        if plotting:
            # Plot mean central parallel profiles
            PlotParaProfiles(dataset=dataset, mode=mode, files=files, singles=singles, spectrals=spectrals, DATE=DATE)

        if spx:
            # Write mean central parallel profiles to spxfile
            WriteParaSpx(dataset=dataset, mode=mode, spectrals=spectrals)

    if bin_ctl:
        # Execute the central meridian binning scheme
        if 'fits' in source:
            singles, spectrals = BinCentreToLimb(nfiles=nfiles, spectrum=spectrum)
        if 'npy' in source:
            singles, spectrals = ReadCentreToLimbNpy(mode=mode, return_singles=False, return_spectrals=True)

        if save:
            # Store calculated profiles
            WriteCentreToLimbProfiles(dataset=dataset, mode=mode, files=files, singles=singles, spectrals=spectrals)

        # if plotting:
            # Plot mean central meridian profiles
            # PlotCentreToLimbProfiles(dataset=dataset, mode=mode, singles=singles, spectrals=spectrals)

        if spx:
            # Write mean central meridian profiles to spxfile
            WriteCentreToLimbSpx(dataset=dataset, mode=mode, spectrals=spectrals)

    if bin_region: 
        # Execute the bi-dimensional binning scheme 
        if 'fits' in source: 
            singles, spectrals = BinRegional(nfiles=nfiles, spectrum=spectrum, LCMIII=LCMIII, per_night=per_night, Nnight=4)
        if 'npy' in source:
            singles, spectrals = ReadRegionalNpy(dataset=dataset, mode=mode, return_singles=True, return_spectrals=True)

        if save:
            # Store calculated maps
            WriteRegional(dataset=dataset, files=files, singles=singles, spectrals=spectrals, per_night=per_night, Nnight=4)
        if plotting:
            # Plot bi-dimensional maps
            if per_night:
                PlotRegionalPerNight(dataset=dataset, spectrals=spectrals, Nnight=4)
            else:
                PlotRegionalMaps(dataset=dataset, mode=mode, spectrals=spectrals)
        if spx:
            # Write bi-dimensional maps to spxfile
            WriteRegionalSpx(dataset=dataset, mode=mode, spectrals=spectrals, per_night=per_night, Nnight=4)
    
    if bin_av_region: 
        # Execute the bi-dimensional binning scheme 
        if 'fits' in source: 
            singles, spectrals = BinRegionalAverage(nfiles=nfiles, spectrum=spectrum, LCMIII=LCMIII, per_night=per_night, Nnight=4)
        if 'npy' in source:
            singles, spectrals = ReadRegionalAverageNpy(dataset=dataset, mode=mode, return_singles=True, return_spectrals=True)

        if save:
            # Store calculated maps
            WriteRegionalAverage(dataset=dataset, files=files, singles=singles, spectrals=spectrals, per_night=per_night, Nnight=4)

        if spx:
            # Write bi-dimensional maps to spxfile
            WriteRegionalAverageSpx(dataset=dataset, mode=mode, spectrals=spectrals, per_night=per_night, Nnight=4)

    ############################################################
    # Read in calibrated data, calculated profiles or retrieved
    # maps and and create maps (cylindrical, polar, etc.)
    ############################################################
    if mapping:

        from Plot.PlotMaps import PlotMaps, PlotMontageGlobalMaps, PlotZoomMaps, PlotMapsPerNight, PlotSubplotMapsPerNight, PlotSubplotMapsPerNightForJGRPaper
        from Plot.PlotPoles import PlotPolesFromGlobal
        from Plot.PlotPseudoWindShear import PlotPseudoWindShear, PlotCompositePseudoWindShear
        from Plot.PlotBrightnessTemperatureProf import PlotCompositeTBprofile

    #     dataset = '2018May'

    # ### Point to location of observations
    #     files       = FindFiles(dataset=dataset, mode=mode+'_files')
    # # Plot cylindrical maps
        if bin_cmerid:
        # Create plots and save global maps into npy arrays
            PlotMaps(dataset, files, spectrals)
        if not bin_cmerid:
        #     # Read in individual calibration coefficients
            _, spectrals = ReadCentralMeridNpy(dataset=dataset, mode=mode, return_singles=False, return_spectrals=True)
        #     # Create plots and save global maps into npy arrays
            PlotMaps(dataset, files, spectrals)
            
    # Plot pole maps from global maps npy arrays
        # PlotZoomMaps(dataset=dataset, central_lon=180, lat_target=-20, lon_target=285, lat_window=15, lon_window=30)
        PlotPolesFromGlobal(dataset=dataset, per_night=False)
        PlotMontageGlobalMaps(dataset=dataset)
        PlotMapsPerNight(dataset=dataset, files=files, spectrals=spectrals)
        # PlotSubplotMapsPerNight(dataset=dataset)
        PlotSubplotMapsPerNightForJGRPaper(dataset=dataset)
        # PlotPseudoWindShear(dataset=dataset)
        PlotCompositePseudoWindShear(dataset=dataset)
        PlotCompositeTBprofile(dataset=dataset)
        
    ############################################################
    # Read in spectral inputs for NEMESIS and plot
    ############################################################

    # if 'spx' in source:
    #     files  = FindFiles(mode=mode+'_spx')
    #     nfiles = len(files)


    ###############################################################
    # Read in retrieved output files from NEMESIS and create
    # meridian and vertical profiles of temperature, aerosols, etc.
    ###############################################################

    if retrieval:

        from Plot.PlotRetrievalOutputs import PlotContributionFunction
        from Plot.PlotRetrievalOutputs import PlotChiSquareOverNy, PlotChiSquareOverNySuperpose, PlotChiSquareMap
        from Plot.PlotRetrievalOutputs import stat_test
        from Plot.PlotRetrievalOutputs import PlotWindShearFromRetrievedTemperature
        from Plot.PlotRetrievalOutputs import PlotRetrievedTemperature, PlotRetrievedTemperatureProfile, PlotRetrievedTemperatureProfileSuperpose 
        from Plot.PlotRetrievalOutputs import PlotRetrievedTemperatureMaps, PlotRetrievedTemperatureCrossSection
        from Plot.PlotRetrievalOutputs import PlotRetrievedRadiance, PlotRetrievedRadianceMap, PlotRetrievedRadianceMeridian, PlotRetrievedRadianceMeridianSuperpose
        from Plot.PlotRetrievalOutputs import PlotRadianceParametricTest
        from Plot.PlotRetrievalOutputs import PlotRetrievedAerosolProfile, PlotRetrievedAerosolMaps, PlotRetrievedAerosolCrossSection,PlotRetrievedAerosolsMeridianProfiles
        from Plot.PlotRetrievalOutputs import PlotRetrievedGasesProfile, PlotRetrievedGasesProfileSuperpose, PlotRetrievedGasesMaps, PlotRetrievedGasesCrossSection
        from Plot.PlotRetrievalOutputs import PlotRetrievedGasesMeridianProfiles, PlotRetrievedGasesMeridianProfilesSuperposed

        from Plot.PlotRetrievalOutputs import PlotComparisonParametricGasesHydrocarbons, PlotComparisonParametricGasesHydrocarbonsParallel
        from Plot.PlotRetrievalOutputs import PlotAllForAuroraOverTime
        from Plot.PlotRetrievalOutputs import PlotSolarWindActivity


        # # # # PlotTemperaturePriorProfiles()
        # # # # PlotRetrievedTemperature()
        # # # # PlotRetrievedTemperatureProfile()
        # PlotRetrievedTemperatureCrossSection(over_axis="latitude")
        # PlotRetrievedTemperatureMaps()

        # PlotRetrievedRadianceMeridian(over_axis="latitude")
        PlotRetrievedRadianceMeridianSuperpose(over_axis="latitude")
        # PlotRetrievedRadianceMap()
        
        # # # # # # # PlotAerosolPriorProfiles()
        # # # # # # # PlotRetrievedAerosolProfile()
        # # PlotRetrievedAerosolCrossSection(over_axis="latitude")
        # PlotRetrievedAerosolsMeridianProfiles(over_axis="latitude")
        # PlotRetrievedAerosolMaps()
        
        # # # # # # # PlotChiSquareOverNy(over_axis="latitude")
        # PlotChiSquareOverNySuperpose(over_axis="latitude")
        # PlotChiSquareMap()
        
        # PlotWindShearFromRetrievedTemperature(over_axis="latitude")
        
        # PlotRetrievedGasesProfile(over_axis="latitude")
        # # # # PlotRetrievedGasesCrossSection(over_axis="latitude")
        PlotRetrievedGasesMeridianProfiles(over_axis="latitude")
        # PlotRetrievedGasesMeridianProfilesSuperposed(over_axis="latitude")
        # PlotComparisonParametricGasesHydrocarbons()
        # PlotRetrievedGasesMaps()
        
        # stat_test()
        # # PlotCheckPente()
        # PlotAllForAuroraOverTime()
        # PlotSolarWindActivity()

        # PlotContributionFunction()


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
    # print(f"Time per file: {np.round((end-start)/90, 3)} s")
    
    # # Stop profiler
    # pr.disable()
    # # Print profiler output to file
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    # ps.strip_dirs()
    # ps.print_stats()

    # with open('../cProfiler_output.txt', 'w+') as f:
    #     f.write(s.getvalue())