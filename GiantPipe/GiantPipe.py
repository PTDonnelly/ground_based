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
    from Plot.PlotProfiles import PlotRegionalAverage
    from Plot.PlotMaps import PlotMaps
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
    calibrate   = False      # Read raw data and calibrate
    source      = 'fits'     # Source of data: local cmaps ('fits') or local numpy arrays ('npy')
    # Binning
    bin_cmerid  = False     # Use central meridian binning scheme
    bin_cpara   = False     # Use central parallel binning scheme
    bin_ctl     = False     # Use centre-to-limb binning scheme
    bin_region  = False     # Use regional binning scheme (for a zoom retrieval)
    bin_av_region = True   # Use averaged regional binning scheme (for a single profile retrieval)
    # Output
    save        = True      # Store calculated profiles to local files
    plotting    = True      # Plot calculated profiles
    mapping     = False      # Plot maps of observations or retrieval
    spx         = True      # Write spxfiles as spectral input for NEMESIS

    ############################################################
    # Perform geometric registration and radiometric calibration
    # of cylindrically-mapped data through central meridian binning
    # and comparison to spacecraft data. Recommended as a first pass. For 
    # plotting, mapping and spxing use calibrated data and .npy files.
    ############################################################

    if calibrate:
        # Define calibration mode
        mode = 'drm'
        dataset = '2018May'

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
            PlotParaProfiles(dataset=dataset, mode=mode, files=files, singles=singles, spectrals=spectrals)

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

    # if mapping == 1:
    # ### Plot cylindrical maps
    #     if bin_cmerid == 0:
    #         # Create plots
    #         PlotMaps(files, spectrals)
    #     if bin_cmerid == 1:
    #         # Read in individual calibration coefficients
    #         _, spectrals, _, _ = ReadNpy(return_singles=False, return_spectrals=True, return_ksingles=True, return_kspectrals=False)
    #         # Create plots
    #         PlotMaps(files, spectrals)
    
    ############################################################
    # Read in spectral inputs for NEMESIS and plot
    ############################################################

    # if 'spx' in source:
    #     files  = FindFiles(mode=mode+'_spx')
    #     nfiles = len(files)

        # if plotting:
        #     PlotSpx(files=files)

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