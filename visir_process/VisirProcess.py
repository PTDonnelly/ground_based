"""
PTD 01/06/22: Code to calibrate VISIR cylindrical maps and
              prepare spectra for input into NEMESIS.
"""

def main():
    import numpy as np
    import time
    from BinningInputs import BinningInputs
    from FindFiles import FindFiles
    from ReadNpy import ReadNpy
    from RegisterMaps import RegisterMaps
    from CreateMeridProfiles import CreateMeridProfiles
    from CalibrateMeridProfiles import CalibrateMeridProfiles
    from WriteProfiles import WriteProfiles
    from CalibrateMaps import CalibrateMaps
    from PlotProfiles import PlotProfiles
    from PlotMaps import PlotMaps
    from WriteSpx import WriteSpx
    
    # Start clock
    start = time.time()
    
    ##### Define global inputs #####
    files       = FindFiles(mode='images')           # Point to location of all input observations
    # Flags
    calc        = 1                                   # (0) Calculate meridional profiles, (1) read stored profiles
    save        = 0                                   # (0) Do not save (1) save meridional profiles
    recal       = 0                                   # (0) Do not calibrate (1) calibrate cylindrical maps (with ksingles)
    plot        = 0                                   # (0) Do not plot (1) plot meridional profiles
    maps        = 0                                   # (0) Do not plot (1) plot cylindrical maps
    spx         = 1                                   # (0) Do not write (1) do write spxfiles for NEMESIS input
    
    ### Calibrate cylindrical maps and produce meridional profiles
    if calc == 0:
        # Steps 1-3: Generate arrays containing spatial and spectral information of each cylindrical map
        spectrum, wavelength, wavenumber, LCMIII = RegisterMaps(files)

        # Steps 4-5: Generate average meridional profiles for each observation and each filter
        singles, spectrals = CreateMeridProfiles(BinningInputs.nfiles, spectrum, LCMIII)
        
        # Steps 6-7: Generate calibrated versions of the profiles from Steps 4 and 5
        calsingles, calspectrals, ksingles, kspectrals = CalibrateMeridProfiles(BinningInputs.nfiles, singles, spectrals, wavenumber)

        # Step 8: Store all cmap profiles and calibration parameters
        if save == 1:
            WriteProfiles(files, calsingles, calspectrals, ksingles, kspectrals)

        # Step 9: Calibrate (or re-calibrate) cylindrical maps using calculated calibration coefficients
        if recal == 1:
            CalibrateMaps(files, ksingles)
    if calc == 1:
        # Calibrate (or re-calibrate) cylindrical maps using pre-calculated calibration coefficients
        if recal == 1:
            # Read in individual calibration coefficients
            _, _, ksingles, _ = ReadNpy(return_singles=False, return_spectrals=False, return_ksingles=True, return_kspectrals=False)
            CalibrateMaps(files, ksingles)

    ### Plot meridional profiles
    if plot == 1:
        if calc == 0:
            # Create plots
            PlotProfiles(calsingles, calspectrals, ksingles, kspectrals, wavenumber)
        if calc == 1:
            # Read in profiles and coefficients
            singles, spectrals, ksingles, kspectrals = ReadNpy(return_singles=True, return_spectrals=True, return_ksingles=True, return_kspectrals=True)
            # Create plots
            PlotProfiles(singles, spectrals, ksingles, kspectrals, wavenumber=False)

    ### Plot cylindrical maps
    if maps == 1:
        if calc == 0:
            # Create plots
            PlotMaps(files, spectrals, ksingles, wavenumber)
        if calc == 1:
            # Read in individual calibration coefficients
            _, spectrals, ksingles, _ = ReadNpy(return_singles=False, return_spectrals=True, return_ksingles=True, return_kspectrals=False)
            # Create plots
            PlotMaps(files, spectrals, ksingles, wavenumber=False)
    
    ### Generate spectral inputs for NEMESIS
    if spx == 1:
        if calc == 0:
            # Create spectra
            WriteSpx(calspectrals)
        if calc == 1:
            # Read in profiles
            _, spectrals, _, _ = ReadNpy(return_singles=False, return_spectrals=True, return_ksingles=False, return_kspectrals=False)
            # Create spectra
            WriteSpx(spectrals)

    # Stop clock
    end = time.time()
    # Print elapsed time
    print(f"Elapsed time: {np.round(end-start, 3)} s")
    print(f"Time per file: {np.round((end-start)/len(files), 3)} s")

if __name__ == '__main__':
    # import cProfile, io, pstats
    # # Create profiler
    # pr = cProfile.Profile()
    # # Start profiler
    # pr.enable()

    main()
    
    # # Stop profiler
    # pr.disable()
    # # Print profiler output to file
    # s = io.StringIO()
    # ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    # ps.strip_dirs()
    # ps.print_stats()

    # with open('../cProfiler_output.txt', 'w+') as f:
    #     f.write(s.getvalue())