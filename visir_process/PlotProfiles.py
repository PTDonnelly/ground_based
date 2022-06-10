def PlotProfiles(singles, spectrals, ksingles, kspectrals):
    import matplotlib.pyplot as plt
    from ReadCal import ReadCal
    from BinningInputs import BinningInputs
    from VisirWavenumbers import VisirWavenumbers
    from SetWave import SetWave
    
    # Read in Voyager and Cassini data into arrays
    calfile = "../visir.jup.filtered-iris-cirs.10-12-15.data.v3"
    iris, cirs = ReadCal(calfile)

    print('Plotting profiles')
    dir = 'calibration_profiles_figures/'
    for ifilt in range(BinningInputs.nfilters):
        # Get filter index for plotting spacecraft and calibrated data
        waves = spectrals[:, ifilt, 5]
        wave  = waves[(waves > 0)][0]
        _, _, ifilt_sc, ifilt_v = SetWave(wavelength=False, wavenumber=wave)
        # Create a figure per filter
        #plt.figure(dpi=900)
        for ifile, wave in enumerate(waves):
            # Get filter index for spectral profiles
            _, _, ifilt_sc, ifilt_v = SetWave(wavelength=False, wavenumber=wave)
            # subplot showing the averaging of each singles merid profiles (ignoring negative beam)
            ax1 = plt.subplot2grid((2, 1), (1, 0))
            #ax1.plot(singles[:, ifile, 0], singles[:, ifile, 3], lw=0, marker='.', markersize=5, label='single')
            ax1.plot(spectrals[:, ifilt_v, 0], spectrals[:, ifilt_v, 3]*kspectrals[ifilt_v, 1], color='orange', lw=0, marker='o', markersize=3, label='visir_av')
            ax1.set_xlim((-90, 90))
            ax1.set_ylim((0, 20e-8))
            ax1.legend()
            # subplot showing the calibration of the spectral merid profile to spacecraft data
            ax2 = plt.subplot2grid((2, 1), (0, 0))
            if ifilt_sc < 12:
                # Use CIRS for N-Band
                ax2.plot(cirs[:, ifilt_sc, 0], cirs[:, ifilt_sc, 1], color='k', lw=1, label='Cassini/CIRS')
            else:
                # Use IRIS for Q-Band
                ax2.plot(iris[:, ifilt_sc, 0], iris[:, ifilt_sc, 1], color='k', lw=1, label='Voyager/IRIS')
            ax2.plot(spectrals[:, ifilt_v, 0], spectrals[:, ifilt_v, 3], color='skyblue', lw=0, marker='o', markersize=3, label='visir_raw')
            ax2.plot(spectrals[:, ifilt_v, 0], spectrals[:, ifilt_v, 3]/kspectrals[ifilt_v, 1], color='orange', lw=0, marker='o', markersize=3, label='visir_calib')
            ax2.set_xlim((-90, 90))
            ax2.set_ylim((0, 20e-8))
            ax2.legend()
        # Save figure showing calibation method 
        filt = VisirWavenumbers(ifilt)
        plt.savefig(f"{dir}{filt}_calibration_merid_profiles.png", dpi=900)
        plt.savefig(f"{dir}{filt}_calibration_merid_profiles.eps", dpi=900)



