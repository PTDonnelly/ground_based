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
    from Plot.PlotProfiles import PlotCentreToLimbProfiles
    from Plot.PlotProfiles import PlotGlobalSpectrals
    # from Plot.PlotProfiles import PlotRegionalMaps
    from Plot.PlotProfiles import PlotRegionalAverage
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
    from Write.WriteSpx import WriteLimbSpx
    from Write.WriteSpx import WriteLimbAverageSpx
    from Write.WriteSpx import WriteRegionalSpx
    from Write.WriteSpx import WriteRegionalAverageSpx

    # Define flags to configure pipeline
    calibrate   = False      # Read raw data and calibrate
    source      = 'npy'     # Source of data: local cmaps ('fits') or local numpy arrays ('npy')
    # Binning
    bin_cmerid  = False     # Use central meridian binning scheme
    bin_cpara   = False     # Use central parallel binning scheme
    bin_ctl     = True     # Use centre-to-limb binning scheme
    bin_region  = False     # Use regional binning scheme (for a zoom 2D retrieval)
    bin_av_region = False   # Use averaged regional binning scheme (for a single profile retrieval)
    # Output
    save        = False      # Store calculated profiles to local files
    plotting    = False      # Plot calculated profiles
    mapping     = False      # Plot maps of observations or retrieval
    spx         = False      # Write spxfiles as spectral input for NEMESIS
    retrieval   = True      # Plot NEMESIS outputs 

    ############################################################
    # Perform geometric registration and radiometric calibration
    # of cylindrically-mapped data through central meridian binning
    # and comparison to spacecraft data. Recommended as a first pass. For 
    # plotting, mapping and spxing use calibrated data and .npy files.
    ############################################################

    if calibrate:
        # Define calibration mode
        mode = 'raw'
        dataset = '2016Feb'


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
    mode = 'giantpipe'
    dataset = '2016Feb'
    # Point to observations
    files  = FindFiles(dataset=dataset, mode=mode+'_files')
    nfiles = len(files)
    
    if 'fits' in source:

        # Generate arrays containing spatial and spectral information of each cylindrical map
        if bin_cmerid:
            spectrum, wavelength, wavenumber, LCMIII, DATE = RegisterMaps(files=files, binning='bin_cmerid')
        if bin_cpara:
            spectrum, wavelength, wavenumber, LCMIII, DATE = RegisterMaps(files=files, binning='bin_cpara')
        if bin_region:
            spectrum, wavelength, wavenumber, LCMIII, DATE = RegisterMaps(files=files, binning='bin_region')
        if bin_av_region:
            spectrum, wavelength, wavenumber, LCMIII, DATE = RegisterMaps(files=files, binning='bin_av_region')
        if bin_ctl:
            spectrum, wavelength, wavenumber, LCMIII, DATE = RegisterMaps(files=files, binning='bin_cmerid')

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
            # PlotGlobalSpectrals(dataset=dataset, spectrals=spectrals)

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
            PlotParaProfiles(dataset=dataset, mode=mode, files=files, singles=singles, spectrals=spectrals)

        if spx:
            # Write mean central parallel profiles to spxfile
            WriteParaSpx(dataset=dataset, mode=mode, spectrals=spectrals)

    if bin_ctl:
        # Execute the central meridian binning scheme
        if 'fits' in source:
            singles, spectrals = BinCentreToLimb(mode=mode, nfiles=nfiles, spectrum=spectrum, LCMIII=LCMIII)
        if 'npy' in source:
            singles, spectrals = ReadCentreToLimbNpy(dataset=dataset, mode=mode, return_singles=True, return_spectrals=True)

        if save:
            # Store calculated profiles
            WriteCentreToLimbProfiles(dataset=dataset, mode=mode, files=files, singles=singles, spectrals=spectrals)

        if plotting:
            # Plot mean central meridian profiles
            PlotCentreToLimbProfiles(dataset=dataset, mode=mode, files=files, singles=singles, spectrals=spectrals)

        if spx:
            # Write mean central meridian profiles to spxfile
            # WriteCentreToLimbSpx(dataset=dataset, mode=mode, spectrals=spectrals)
            # WriteLimbSpx(dataset=dataset, mode=mode, spectrals=spectrals)
            WriteLimbAverageSpx(dataset=dataset, mode=mode, spectrals=spectrals)

    if bin_region: 
        # Execute the bi-dimensional binning scheme 
        if 'fits' in source: 
            singles, spectrals = BinRegional(nfiles=nfiles, spectrum=spectrum, LCMIII=LCMIII)
        if 'npy' in source:
            singles, spectrals = ReadRegionalNpy(dataset=dataset, mode=mode, return_singles=True, return_spectrals=True)

        if save:
            # Store calculated maps
            WriteRegional(dataset=dataset, files=files, singles=singles, spectrals=spectrals)
        if plotting:
            # Plot bi-dimensional maps
            PlotRegionalMaps(dataset=dataset, mode=mode, spectrals=spectrals)
        if spx:
            # Write bi-dimensional maps to spxfile
            WriteRegionalSpx(dataset=dataset, mode=mode, spectrals=spectrals)
    
    if bin_av_region: 
        # Execute the bi-dimensional binning scheme 
        if 'fits' in source: 
            singles, spectrals = BinRegionalAverage(nfiles=nfiles, spectrum=spectrum, LCMIII=LCMIII, DATE=DATE, per_night=True, Nnight=4)
        if 'npy' in source:
            singles, spectrals = ReadRegionalAverageNpy(dataset=dataset, mode=mode, return_singles=True, return_spectrals=True)

        if save:
            # Store calculated maps
            WriteRegionalAverage(dataset=dataset, files=files, singles=singles, spectrals=spectrals, per_night=True, Nnight=4)
        # if plotting:
        #     # Plot bi-dimensional maps
        #     PlotRegionalAverage(dataset=dataset, mode=mode, spectrals=spectrals)
        if spx:
            # Write bi-dimensional maps to spxfile
            WriteRegionalAverageSpx(dataset=dataset, mode=mode, spectrals=spectrals, per_night=True, Nnight=4)

    ############################################################
    # Read in calibrated data, calculated profiles or retrieved
    # maps and and create maps (cylindrical, polar, etc.)
    ############################################################
    if mapping:

        from Plot.PlotMaps import PlotMaps, PlotZoomMaps, PlotMapsPerNight
        # from Plot.PlotPoles import PlotPolesFromGlobal
        # from Plot.PlotPseudoWindShear import PlotPseudoWindShear, PlotCompositePseudoWindShear
        # from Plot.PlotBrightnessTemperatureProf import PlotCompositeTBprofile

        mode = 'giantpipe'
        dataset = '2016Feb'

        ### Point to location of observations
        files = FindFiles(dataset=dataset, mode=mode+'_files')
    
        ## Plot cylindrical maps
        if bin_cmerid:
            # Create plots and save global maps into npy arrays
            PlotMaps(dataset, files, spectrals)
        if not bin_cmerid:
            # Read in individual calibration coefficients
            _, spectrals = ReadCentralMeridNpy(dataset=dataset, mode=mode, return_singles=False, return_spectrals=True)
            # Create plots and save global maps into npy arrays
            PlotMaps(dataset, files, spectrals)
    
        # Plot pole maps from global maps npy arrays
        # PlotZoomMaps(dataset=dataset, central_lon=180, lat_target=-20, lon_target=285, lat_window=15, lon_window=30)
        # PlotPolesFromGlobal(dataset=dataset, per_night=False)
        # PlotPseudoWindShear(dataset=dataset)
        # PlotCompositePseudoWindShear(dataset=dataset)
        # PlotCompositeTBprofile(dataset=dataset)
        
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
        from Plot.PlotNemesis import PlotNemesis as pn
        
        # pn.plot_results_global()
        pn.plot_contribution_function()


        # from Plot.PlotRetrievalOutputs import PlotChiSquareOverNy, PlotChiSquareOverNySuperpose, PlotRetrievedTemperatureProfileSuperpose
        # # PlotChiSquareOverNy(over_axis='latitude')
        # PlotChiSquareOverNySuperpose(over_axis='latitude')
        # # PlotRetrievedTemperatureProfileSuperpose(over_axis='latitude')


    return

if __name__ == '__main__':
    import numpy as np
    import time

    # Start clock
    start = time.time()

    main()

    # Stop clock
    end = time.time()
    # Print elapsed time
    print(f"Elapsed time: {np.round(end-start, 3)} s")
