"""
PTD 01/06/22: Code to calibrate VISIR cylindrical maps and
              prepare spectra for input into NEMESIS.
"""

def main():
    from FindFiles import FindFiles
    from RegisterMaps import RegisterMaps
    from Binning.CentralMerid import BinCentralMerid
    from Binning.CentreToLimb import BinCentreToLimb
    from Calibrate.CalibrateGBData import CalibrateGBData
    from Calibrate.CentralMerid import CalCentralMerid
    # from Calibrate.CylindricalMaps import CalCylindricalMaps
    from Plot.PlotProfiles import PlotMeridProfiles
    from Plot.PlotProfiles import PlotGlobalSpectrals
    from Plot.PlotMaps import PlotMaps
    from Read.ReadNpy import ReadCentralMeridNpy
    from Read.ReadNpy import ReadCentreToLimbNpy
    from Read.ReadSpx import ReadSpx
    from Write.WriteProfiles import WriteMeridProfiles
    from Write.WriteProfiles import WriteCentreToLimbProfiles
    from Write.WriteSpx import WriteMeridSpx
    from Write.WriteSpx import WriteCentreToLimbSpx
    
    # Define flags to configure pipeline
    calibrate   = False      # Read raw data and calibrate
    source      = 'fits'     # Source of data: local cmaps ('fits') or local numpy arrays ('npy')
    # Binning
    bin_cmerid  = True     # Use central meridian binning scheme
    bin_ctl     = False     # Use centre-to-limb binning scheme
    # Output
    save        = False      # Store calculated profiles to local files
    plotting    = True      # Plot calculated profiles
    mapping     = False      # Plot maps of observations or retrieval
    spx         = False      # Write spxfiles as spectral input for NEMESIS

    ############################################################
    # Perform geometric registration and radiometric calibration
    # of cylindrically-mapped data through central meridian binning
    # and comparison to spacecraft data. Recommended as a first pass. For 
    # plotting, mapping and spxing use calibrated data and .npy files.
    ############################################################

    if calibrate:
        # Define calibration mode
        mode = 'drm'
        # Point to observations
        CalibrateGBData(mode=mode+'_files')
        exit()

    ############################################################
    # source = 'fits': read in calibrated cylindrical maps of 
    # radiance, emission angle (and optionally Doppler velocity), 
    # or pre-calculated profiles saved to local numpy files, and
    # create profiles depending on the chosen binning scheme.
    ############################################################
    
    # Define calibration mode
    mode   = 'giantpipe'
    # Point to observations
    files  = FindFiles(mode=mode+'_files')
    nfiles = len(files)
    
    if 'fits' in source:
        # Generate arrays containing spatial and spectral information of each cylindrical map
        spectrum, wavelength, wavenumber, LCMIII = RegisterMaps(files=files)

    if bin_cmerid:

        # Execute the central meridian binning scheme
        if 'fits' in source:
            singles, spectrals = BinCentralMerid(nfiles=nfiles, spectrum=spectrum, LCMIII=LCMIII)
        if 'npy' in source:
            singles, spectrals = ReadCentralMeridNpy(mode=mode, return_singles=True, return_spectrals=True)

        if save:
            # Store calculated profiles
            WriteMeridProfiles(files=files, singles=singles, spectrals=spectrals)

        if plotting:
            # Plot mean central meridian profiles
            # PlotMeridProfiles(mode=mode, files=files, singles=singles, spectrals=spectrals)
            PlotGlobalSpectrals(spectrals=spectrals)

        if spx:
            # Write mean central meridian profiles to spxfile
            WriteMeridSpx(mode=mode, spectrals=spectrals)
    
    if bin_ctl:

        # Execute the central meridian binning scheme
        if 'fits' in source:
            singles, spectrals = BinCentreToLimb(nfiles=nfiles, spectrum=spectrum)
        if 'npy' in source:
            singles, spectrals = ReadCentreToLimbNpy(mode=mode, return_singles=False, return_spectrals=True)

        if save:
            # Store calculated profiles
            WriteCentreToLimbProfiles(mode=mode, files=files, singles=singles, spectrals=spectrals)

        if plotting:
            # Plot mean central meridian profiles
            PlotCentreToLimbProfiles(mode=mode, singles=singles, spectrals=spectrals)

        if spx:
            # Write mean central meridian profiles to spxfile
            WriteCentreToLimbSpx(mode=mode, spectrals=spectrals)

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

    if 'spx' in source:
        files  = FindFiles(mode=mode+'_spx')
        nfiles = len(files)

        if plotting:
            PlotSpx(files=files)

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