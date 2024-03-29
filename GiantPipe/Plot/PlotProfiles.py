import os 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from copy import copy
from scipy.interpolate import UnivariateSpline, interp1d
import Globals
from Read.ReadCal import ReadCal
from Tools.SetWave import SetWave

# Colormap definition
cmap = get_cmap("magma")

def PlotMeridProfiles(dataset, mode, files, singles, spectrals):
    """ Plot meridian profiles and spacecraft data to illustrate 
            the calibration method """

    print('Plotting profiles...')

    # Read in Voyager and Cassini data into arrays
    calfile = "../inputs/visir.jup.filtered-iris-cirs.10-12-15.data.v3"
    iris, cirs = ReadCal(calfile)

    # If subdirectory does not exist, create it
    dir = f'../outputs/{dataset}/calibration_profiles_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    for ifilt in range(Globals.nfilters):
        # Get filter index for plotting spacecraft and calibrated data
        _, _, wave, ifilt_sc, ifilt_v = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=ifilt)
        
        # Create a figure per filter
        fig = plt.figure(figsize=(8, 3))
        # subplot showing the averaging of each singles merid profiles (ignoring negative beam)
        for ifile, fname in enumerate(files):
            _, _, iwave, _, _ = SetWave(filename=fname, wavelength=None, wavenumber=None, ifilt=None)
            if iwave == wave:
                plt.plot(singles[:, ifile, 0], singles[:, ifile, 3], color='black', lw=0, marker='.', markersize=2)
        # Select the suitable spacecraft meridian profile
        if ifilt_sc < 12:
            # Use CIRS for N-Band
            cirskeep = (cirs[:, ifilt_sc, 0] >= -70) & (cirs[:, ifilt_sc, 0] < 70)
            plt.plot(cirs[cirskeep, ifilt_sc, 0], cirs[cirskeep, ifilt_sc, 1], color='green', lw=2, label='Cassini/CIRS')
        else:
            # Use IRIS for Q-Band
            iriskeep = (iris[:, ifilt_sc, 0] >= -70) & (iris[:, ifilt_sc, 0] < 70)
            plt.plot(iris[iriskeep, ifilt_sc, 0], iris[iriskeep, ifilt_sc, 1], color='green', lw=2, label='Voyager/IRIS')
        # Plot the VLT/VISIR pole-to-pole meridian profile
        plt.plot(spectrals[:, ifilt_v, 0], spectrals[:, ifilt_v, 3], color='orange', lw=0, marker='o', markersize=3, label=f"averaged VLT/VISIR at {int(wave)}"+" cm$^{-1}$")
        plt.xlim((-90, 90))
        plt.tick_params(labelsize=12)
        plt.grid()
        plt.legend(fontsize=12)
        plt.xlabel("Planetocentric Latitude", size=15)
        plt.ylabel("Radiance (W cm$^{-1}$ sr$^{-1}$)", size=15)

        # Save figure showing calibation method 
        # _, _, wave, _, _ = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=ifilt)
        plt.savefig(f"{dir}{wave}_calibration_merid_profiles.png", dpi=150, bbox_inches='tight')
        plt.savefig(f"{dir}{wave}_calibration_merid_profiles.pdf", dpi=500, bbox_inches='tight')
        # Clear figure to avoid overlapping between plotting subroutines
        plt.close()

    fig, ax = plt.subplots(6, 2, figsize=(12, 12), sharex=True, sharey=False)
    iax = 0
    for ifilt in [0,10,11,12,5,4,6,7,8,9,3,2,1]:
        if ifilt < 6 or ifilt > 7:
            irow = [0,1,1,2,2,3,3,4,4,5,5]
            icol = [0,0,1,0,1,0,1,0,1,0,1]
            ititle = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)']
            # Remove the frame of the empty subplot
            ax[0][1].set_frame_on(False)
            ax[0][1].tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            # Get filter index for plotting spacecraft and calibrated data
            _, wavl, wave, ifilt_sc, ifilt_v = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=ifilt)
            
            # subplot showing the averaging of each singles merid profiles (ignoring negative beam)
            for ifile, fname in enumerate(files):
                _, _, iwave, _, _ = SetWave(filename=fname, wavelength=None, wavenumber=None, ifilt=None)
                if iwave == wave:
                    ax[irow[iax]][icol[iax]].plot(singles[:, ifile, 0], singles[:, ifile, 3]*1.e9, lw=0, marker='.', markersize=2, color = 'black')
            # Select the suitable spacecraft meridian profile
            if ifilt_sc < 12:
                # Use CIRS for N-Band
                cirskeep = (cirs[:, ifilt_sc, 0] >= -70) & (cirs[:, ifilt_sc, 0] < 70)
                ax[irow[iax]][icol[iax]].plot(cirs[cirskeep, ifilt_sc, 0], cirs[cirskeep, ifilt_sc, 1]*1.e9, color='green', lw=2, label='Cassini/CIRS')
            else:
                # Use IRIS for Q-Band
                iriskeep = (iris[:, ifilt_sc, 0] >= -70) & (iris[:, ifilt_sc, 0] < 70)
                ax[irow[iax]][icol[iax]].plot(iris[iriskeep, ifilt_sc, 0], iris[iriskeep, ifilt_sc, 1]*1.e9, color='green', lw=2, label='Voyager/IRIS')
            # Plot the VLT/VISIR pole-to-pole meridian profile
            ax[irow[iax]][icol[iax]].plot(spectrals[:, ifilt_v, 0], spectrals[:, ifilt_v, 3]*1.e9, color='orange', lw=0, marker='o', markersize=2, label='VLT/VISIR')# at "+f"{abs(Globals.LCP)}"+"$^{\circ}$S")
            ax[irow[iax]][icol[iax]].set_title(ititle[iax]+f"    {(wavl)}"+" $\mu$m", fontfamily='sans-serif', loc='left', fontsize=12)
            ax[irow[iax]][icol[iax]].legend(fontsize=11)#, loc='lower right')
            ax[irow[iax]][icol[iax]].grid()
            ax[irow[iax]][icol[iax]].set_xlim(-90, 90)
            ax[irow[iax]][icol[iax]].set_xticks([]) if (iax < 9) else ax[irow[iax]][icol[iax]].set_xticks([-90,-80, -60, -40, -20, 0, 20, 40, 60, 80, 90])
            ax[irow[iax]][icol[iax]].tick_params(axis='x', labelrotation=45)
            ax[irow[iax]][icol[iax]].tick_params(labelsize=14)
            iax+=1 
    plt.axes([0.1, 0.09, 0.8, 0.8], frameon=False) 
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Planetocentric Latitude", size=18)
    # if spectrals[0, ifilt_v, 1] >=0 and spectrals[-1, ifilt_v, 1]>=0:
    #     plt.xlim(spectrals[0, ifilt_v, 1], spectrals[-1, ifilt_v, 1]) 
    #     plt.xticks(ticks=np.arange(360,-1,-60), labels=list(np.arange(360,-1,-60)))
    plt.ylabel("Radiance (nW cm$^{-2}$ sr$^{-1}$ [cm$^{-1}$]$^{-1}$)", size=18)
    # Save figure showing calibation method 
    plt.savefig(f"{dir}all_filters_calibration_merid_profiles.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{dir}all_filters_calibration_merid_profiles.pdf", dpi=500, bbox_inches='tight')
    #plt.savefig(f"{dir}{wave}_calibration_merid_profiles.pdf", dpi=900)
    # Clear figure to avoid overlapping between plotting subroutines
    plt.close()

    fig, ax = plt.subplots(6, 2, figsize=(12, 12), sharex=True, sharey=False)
    iax = 0
    for ifilt in [0,10,11,12,5,4,6,7,8,9,3,2,1]:
        if ifilt < 6 or ifilt > 7:
            irow = [0,1,1,2,2,3,3,4,4,5,5]
            icol = [0,0,1,0,1,0,1,0,1,0,1]
            ititle = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)']
            # Remove the frame of the empty subplot
            ax[0][1].set_frame_on(False)
            ax[0][1].tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            # Get filter index for plotting spacecraft and calibrated data
            _, wavl, wave, ifilt_sc, ifilt_v = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=ifilt)

            coeff = np.load(f'/Users/deborah/Documents/Research/code/ground_based/outputs/{dataset}/spectral_coeff_profiles/{wave}_calib_coeff.npy')
            
            # subplot showing the averaging of each singles merid profiles (ignoring negative beam)
            for ifile, fname in enumerate(files):
                _, _, iwave, _, _ = SetWave(filename=fname, wavelength=None, wavenumber=None, ifilt=None)
                if iwave == wave:
                    ax[irow[iax]][icol[iax]].plot(singles[:, ifile, 0], singles[:, ifile, 3]*1.e9, lw=0, marker='.', markersize=2, color = 'black')
            # Select the suitable spacecraft meridian profile
            if ifilt_sc < 12:
                # Use CIRS for N-Band
                cirskeep = (cirs[:, ifilt_sc, 0] >= -70) & (cirs[:, ifilt_sc, 0] < 70)
                ax[irow[iax]][icol[iax]].plot(cirs[cirskeep, ifilt_sc, 0], cirs[cirskeep, ifilt_sc, 1]*1.e9, color='green', lw=2, label='Cassini/CIRS')
            else:
                # Use IRIS for Q-Band
                iriskeep = (iris[:, ifilt_sc, 0] >= -70) & (iris[:, ifilt_sc, 0] < 70)
                ax[irow[iax]][icol[iax]].plot(iris[iriskeep, ifilt_sc, 0], iris[iriskeep, ifilt_sc, 1]*1.e9, color='green', lw=2, label='Voyager/IRIS')
            # Plot the VLT/VISIR pole-to-pole meridian profile
            ax[irow[iax]][icol[iax]].plot(spectrals[:, ifilt_v, 0], spectrals[:, ifilt_v, 3]*1.e9, color='orange', lw=0, marker='o', markersize=2, label='VLT/VISIR')# at "+f"{abs(Globals.LCP)}"+"$^{\circ}$S")
            ax[irow[iax]][icol[iax]].set_title(ititle[iax]+f"    {(wavl)}"+" $\mu$m"+f"                            calib coeff = {coeff[1]:.2f} ", fontfamily='sans-serif', loc='left', fontsize=12)
            ax[irow[iax]][icol[iax]].legend(fontsize=11)#, loc='lower right')
            # ax[irow[iax]][icol[iax]].text(1.0, 1.0, f'{coeff[1]}', fontsize=12)
            ax[irow[iax]][icol[iax]].grid()
            ax[irow[iax]][icol[iax]].set_xlim(-90, 90)
            ax[irow[iax]][icol[iax]].set_xticks([]) if (iax < 9) else ax[irow[iax]][icol[iax]].set_xticks([-90,-80, -60, -40, -20, 0, 20, 40, 60, 80, 90])
            ax[irow[iax]][icol[iax]].tick_params(axis='x', labelrotation=45)
            ax[irow[iax]][icol[iax]].tick_params(labelsize=14)
            iax+=1 
    plt.axes([0.1, 0.09, 0.8, 0.8], frameon=False) 
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("Planetocentric Latitude", size=18)
    # if spectrals[0, ifilt_v, 1] >=0 and spectrals[-1, ifilt_v, 1]>=0:
    #     plt.xlim(spectrals[0, ifilt_v, 1], spectrals[-1, ifilt_v, 1]) 
    #     plt.xticks(ticks=np.arange(360,-1,-60), labels=list(np.arange(360,-1,-60)))
    plt.ylabel("Radiance (nW cm$^{-2}$ sr$^{-1}$ [cm$^{-1}$]$^{-1}$)", size=18)
    # Save figure showing calibation method 
    plt.savefig(f"{dir}all_filters_calibration_merid_profiles_and_coefficients.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{dir}all_filters_calibration_merid_profiles_and_coefficients.pdf", dpi=500, bbox_inches='tight')
    # Clear figure to avoid overlapping between plotting subroutines
    plt.close()

def PlotParaProfiles(dataset, mode, files, singles, spectrals, DATE):
    """ Plot parallel profiles """

    print('Plotting parallel profiles...')

    # If subdirectory does not exist, create it
    dir = f'../outputs/{dataset}/parallel_{Globals.LCP}_profiles_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    for ifilt in range(Globals.nfilters):
        # Get filter index for plotting spacecraft and calibrated data
        _, _, wave, _, ifilt_v = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=ifilt)
        # Create a figure per filter
        fig = plt.subplots(1, 1, figsize=(8, 3), sharex=True, sharey=True)
        # subplot showing the parallel profiles for each filter
        for ifile, fname in enumerate(files):
            _, _, iwave, _, _ = SetWave(filename=fname, wavelength=None, wavenumber=None, ifilt=None)
            if iwave == wave:
                plt.plot(singles[:, ifile, 1], singles[:, ifile, 3]*1.e9, lw=0, marker='.', markersize=2, color = 'black')
        plt.plot(spectrals[:, ifilt_v, 1], spectrals[:, ifilt_v, 3]*1.e9, color='orange', lw=0, marker='o', markersize=2, label=f"{int(wave)}"+" cm$^{-1}$ VLT/VISIR profile at "+f"{Globals.LCP}"+"$^{\circ}$")
        plt.legend(fontsize=12)
        plt.grid()
        plt.tick_params(labelsize=12) 
        plt.xlabel("System III West Longitude", size=15)
        if spectrals[0, ifilt_v, 1] >=0 and spectrals[-1, ifilt_v, 1]>=0:
            plt.xlim(spectrals[0, ifilt_v, 1], spectrals[-1, ifilt_v, 1]) 
            plt.xticks(ticks=np.arange(360,-1,-30), labels=list(np.arange(360,-1,-30)))
        plt.ylabel("Radiance (nW cm$^{-2}$ sr$^{-1}$ [cm$^{-1}$]$^{-1}$)", size=15)
        # Save figure showing calibation method 
        plt.savefig(f"{dir}{wave}_parallel_profiles.png", dpi=150, bbox_inches='tight')
        plt.savefig(f"{dir}{wave}_parallel_profiles.pdf", dpi=500, bbox_inches='tight')
        # Clear figure to avoid overlapping between plotting subroutines
        plt.close()

    # Create a subplots figures with all filters
    fig, ax = plt.subplots(6, 2, figsize=(12, 12), sharex=True, sharey=False)
    iax = 0
    for ifilt in [0,10,11,12,5,4,6,7,8,9,3,2,1]:
        if ifilt < 6 or ifilt > 7:
            irow = [0,1,1,2,2,3,3,4,4,5,5]
            icol = [0,0,1,0,1,0,1,0,1,0,1]
            ititle = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)']
            # Remove the frame of the empty subplot
            ax[0][1].set_frame_on(False)
            ax[0][1].tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            # Get filter index for plotting spacecraft and calibrated data
            _, wavl, wave, _, ifilt_v = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=ifilt)
            
            # subplot showing the parallel profiles for each filter
            for ifile, fname in enumerate(files):
                _, _, iwave, _, _ = SetWave(filename=fname, wavelength=None, wavenumber=None, ifilt=None)
                if iwave == wave:
                    ax[irow[iax]][icol[iax]].plot(singles[:, ifile, 1], singles[:, ifile, 3]*1.e9, lw=0, marker='.', markersize=2, color = 'black')
            ax[irow[iax]][icol[iax]].plot(spectrals[:, ifilt_v, 1], spectrals[:, ifilt_v, 3]*1.e9, color='orange', lw=0, marker='o', markersize=2)# at "+f"{abs(Globals.LCP)}"+"$^{\circ}$S")
            ax[irow[iax]][icol[iax]].set_title(ititle[iax]+f"    {(wavl)}"+" $\mu$m", fontfamily='sans-serif', loc='left', fontsize=12)
            ax[irow[iax]][icol[iax]].grid()
            ax[irow[iax]][icol[iax]].set_xlim(360, 0)
            ax[irow[iax]][icol[iax]].set_xticks([]) if (iax < 9) else ax[irow[iax]][icol[iax]].set_xticks(ticks=np.arange(360,-1,-60), labels=list(np.arange(360,-1,-60)))
            ax[irow[iax]][icol[iax]].tick_params(labelsize=14)
            iax+=1 
    plt.axes([0.1, 0.1, 0.8, 0.8], frameon=False) 
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("System III West Longitude", size=18)
    # if spectrals[0, ifilt_v, 1] >=0 and spectrals[-1, ifilt_v, 1]>=0:
    #     plt.xlim(spectrals[0, ifilt_v, 1], spectrals[-1, ifilt_v, 1]) 
    #     plt.xticks(ticks=np.arange(360,-1,-60), labels=list(np.arange(360,-1,-60)))
    plt.ylabel("Radiance (nW cm$^{-2}$ sr$^{-1}$ [cm$^{-1}$]$^{-1}$)", size=18)
    # Save figure showing calibation method 
    plt.savefig(f"{dir}all_filters_parallel_profiles.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{dir}all_filters_parallel_profiles.pdf", dpi=500, bbox_inches='tight')

    # Clear figure to avoid overlapping between plotting subroutines
    plt.close()


    night_limits = [0, 20180524120000, 20180525120000, 20180526120000, 20180527120000]

    # Create a subplots figures with all filters
    fig, ax = plt.subplots(6, 2, figsize=(12, 12), sharex=True, sharey=False)
    iax = 0
    ired, iorange, igreen, iblue = 0, 0, 0, 0
    for ifilt in [0,10,11,12,5,4,6,7,8,9,3,2,1]:
        if ifilt < 6 or ifilt > 7:
            irow = [0,1,1,2,2,3,3,4,4,5,5]
            icol = [0,0,1,0,1,0,1,0,1,0,1]
            ititle = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)']
            
            # Remove the frame of the empty subplot
            ax[0][1].set_frame_on(False)
            ax[0][1].tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            # Get filter index for plotting spacecraft and calibrated data
            _, wavl, wave, _, ifilt_v = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=ifilt)
            
            # subplot showing the parallel profiles for each filter
            for ifile, fname in enumerate(files):

                _, _, iwave, _, _ = SetWave(filename=fname, wavelength=None, wavenumber=None, ifilt=None)
                if iwave == wave:
                    # Change date format to store it as a float in singles_av_regions arrays:
                    DATE[ifile] = DATE[ifile].replace('T', '')
                    DATE[ifile] = DATE[ifile].replace('-', '')
                    DATE[ifile] = DATE[ifile].replace(':', '')
                    if (float(DATE[ifile]) < 20180524120000):
                        if ired == 0:
                            ax[irow[iax]][icol[iax]].plot(singles[:, ifile, 1], singles[:, ifile, 3]*1.e9, lw=3, color = 'red', label=r"May 24$^{th}$")
                        else:
                            ax[irow[iax]][icol[iax]].plot(singles[:, ifile, 1], singles[:, ifile, 3]*1.e9, lw=3, color = 'red')
                        ired+=1
                    elif (float(DATE[ifile]) > 20180524120000) & (float(DATE[ifile]) < 20180525120000):
                        if iorange == 0:
                            ax[irow[iax]][icol[iax]].plot(singles[:, ifile, 1], singles[:, ifile, 3]*1.e9, lw=3, color = 'orange',label=r"May 24$^{th}$-25$^{th}$")
                        else:
                            ax[irow[iax]][icol[iax]].plot(singles[:, ifile, 1], singles[:, ifile, 3]*1.e9, lw=3, color = 'orange')
                        iorange+=1
                    elif (float(DATE[ifile]) > 20180525120000) & (float(DATE[ifile]) < 20180526120000):
                        if igreen == 0:
                            ax[irow[iax]][icol[iax]].plot(singles[:, ifile, 1], singles[:, ifile, 3]*1.e9, lw=3, color = 'green', label=r"May 25$^{th}$-26$^{th}$")
                        else:
                            ax[irow[iax]][icol[iax]].plot(singles[:, ifile, 1], singles[:, ifile, 3]*1.e9, lw=3, color = 'green')
                        igreen+=1
                    elif (float(DATE[ifile]) > 20180526120000) & (float(DATE[ifile]) < 20180527120000):
                        if iblue == 0:
                            ax[irow[iax]][icol[iax]].plot(singles[:, ifile, 1], singles[:, ifile, 3]*1.e9, lw=3, color = 'blue', label=r"May 26$^{th}$-27$^{th}$")
                        else:
                            ax[irow[iax]][icol[iax]].plot(singles[:, ifile, 1], singles[:, ifile, 3]*1.e9, lw=3, color = 'blue')
                        iblue+=1
            #ax[irow[iax]][icol[iax]].plot(spectrals[:, ifilt_v, 1], spectrals[:, ifilt_v, 3]*1.e9, color='black', lw=0, marker='.', markersize=1)# at "+f"{abs(Globals.LCP)}"+"$^{\circ}$S")
            ax[irow[iax]][icol[iax]].set_title(ititle[iax]+f"    {(wavl)}"+" $\mu$m", fontfamily='sans-serif', loc='left', fontsize=12)
            ax[irow[iax]][icol[iax]].grid()
            ax[irow[iax]][icol[iax]].set_xlim(360, 0)
            ax[irow[iax]][icol[iax]].set_xticks([]) if (iax < 9) else ax[irow[iax]][icol[iax]].set_xticks(ticks=np.arange(360,-1,-60), labels=list(np.arange(360,-1,-60)))
            ax[irow[iax]][icol[iax]].tick_params(labelsize=14)
            iax+=1
            
    handles, labels = ax[0][0].get_legend_handles_labels()  
    fig.legend(handles, labels,fontsize=15, bbox_to_anchor=[0.85, 0.91]) 
    plt.axes([0.1, 0.1, 0.8, 0.8], frameon=False) 
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("System III West Longitude", size=18)
    # if spectrals[0, ifilt_v, 1] >=0 and spectrals[-1, ifilt_v, 1]>=0:
    #     plt.xlim(spectrals[0, ifilt_v, 1], spectrals[-1, ifilt_v, 1]) 
    #     plt.xticks(ticks=np.arange(360,-1,-60), labels=list(np.arange(360,-1,-60)))
    if Globals.LCP > 0:
        plt.ylabel(f"Radiance at {abs(Globals.LCP)}"+"$^{\circ}$N (nW cm$^{-2}$ sr$^{-1}$ [cm$^{-1}$]$^{-1}$)", size=18)
    elif Globals.LCP < 0:
        plt.ylabel(f"Radiance at {abs(Globals.LCP)}"+"$^{\circ}$S (nW cm$^{-2}$ sr$^{-1}$ [cm$^{-1}$]$^{-1}$)", size=18)
    # Save figure showing calibation method 
    plt.savefig(f"{dir}all_filters_parallel_profiles_nightcolored.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{dir}all_filters_parallel_profiles_nightcolored.pdf", dpi=500, bbox_inches='tight')

    # Clear figure to avoid overlapping between plotting subroutines
    plt.close()

def PlotGlobalSpectrals(dataset, spectrals):
    """Basic code to plot the central meridian profiles with wavenumber
    (or spectral profiles with latitude, depending on the perspective).
    Displays the global pseudo-spectrum in a way resembling a normal spectrum."""
    
    # If subdirectory does not exist, create it
    dir = f'../outputs/{dataset}/calibration_profiles_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    lat = copy(spectrals[:, :, 0])
    wave = copy(spectrals[:, :, 5])
    rad =  copy(spectrals[:, :, 3])
    rad_res1 = copy(spectrals[:, :, 3])
    rad_res2 = copy(spectrals[:, :, 3])
    for i in range(Globals.nfilters):
        rad_res1[:, i] = rad[:, i] - np.nanmean(rad[:, i])
        rad_res2[:, i] = (rad_res1[:, i]/np.nanmean(rad[:, i]))*100

    plt.figure
    ax1 = plt.contourf(wave, lat, rad, levels=200, cmap='nipy_spectral')
    for i in range(Globals.nfilters):
        plt.plot((wave[i], wave[i]), (-90, 90), ls=':', lw=0.7, color='white')
    cbar = plt.colorbar(ax1)
    cbar.set_label('Radiance (W cm$^{-2}$ sr$^{-1}$ (cm$^{-1}$)$^{-1}$)', size=15)
    plt.xlabel('Wavenumber (cm$^{-1}$)', size=15)
    #plt.xlim((xmin, xmax))
    plt.ylabel('Latitude', size=15)
    plt.ylim((-90, 90))
    plt.savefig(f"{dir}global_spectrals.png", dpi=150, bbox_inches='tight')
    plt.close()
 
    plt.figure
    ax2 = plt.contourf(wave, lat, rad_res1, vmin=-1*np.nanmax(rad_res1),vmax=np.nanmax(rad_res1), levels=200, cmap='seismic')
    for i in range(Globals.nfilters):
        plt.plot((wave[i], wave[i]), (-90, 90), ls=':', lw=0.7, color='black')
    cbar = plt.colorbar(ax2)
    cbar.set_label('Residual radiance (W cm$^{-2}$ sr$^{-1}$ (cm$^{-1}$)$^{-1}$)', size=15)
    plt.xlabel('Wavenumber (cm$^{-1}$)', size=15)
    #plt.xlim((xmin, xmax))
    plt.ylabel('Latitude', size=15)
    plt.ylim((-90, 90))
    plt.savefig(f"{dir}global_spectrals_res1.png", dpi=150, bbox_inches='tight')
    plt.close()

    plt.figure
    ax3 = plt.contourf(wave, lat, rad_res2, vmin=-1*np.nanmax(rad_res2),vmax=np.nanmax(rad_res2), levels=200, cmap='seismic')
    for i in range(Globals.nfilters):
        plt.plot((wave[i], wave[i]), (-90, 90), ls=':', lw=0.7, color='black')
    cbar = plt.colorbar(ax3)
    cbar.set_label('Difference (pourcent)', size=15)
    plt.xlabel('Wavenumber (cm$^{-1}$)', size=15)
    #plt.xlim((xmin, xmax))
    plt.ylabel('Latitude', size=15)
    plt.ylim((-90, 90))
    plt.savefig(f"{dir}global_spectrals_res2.png", dpi=150, bbox_inches='tight')
    plt.close()

def PlotCentreTotLimbProfiles(mode, singles, spectrals):
    print('Plotting profiles...')

    
    a = 1

def PlotRegionalMaps(dataset, mode, spectrals):
    """ Plot radiance maps """

    print('Plotting radiances maps...')

    # If subdirectory does not exist, create it
    dir = f'../outputs/{dataset}/maps_radiance_lat{Globals.lat_target}_lon{Globals.lon_target}_{Globals.latstep}x{Globals.lonstep}_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
   
    for ifilt in range(Globals.nfilters):
        # Get retrieve wavenumber value from ifilt index
        _, _, wave, _, _ = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=ifilt)  
        # Set extreme values for mapping
        # spectrals[:, :, ifilt, 3] *= 1.e9
        max = np.nanmax(spectrals[:, :, ifilt, 3]*1.e9) 
        min = np.nanmin(spectrals[:, :, ifilt, 3]*1.e9)
        # Create a figure per filter
        fig = plt.figure(figsize=(8, 3))
        plt.imshow(spectrals[:, :, ifilt, 3]*1.e9, vmin=min, vmax=max, origin='lower', extent = [360,0,-90,90],  cmap='inferno')
        plt.xlim(Globals.lon_target+Globals.merid_width, Globals.lon_target-Globals.merid_width)
        plt.xticks(np.arange(Globals.lon_target-Globals.merid_width, Globals.lon_target+Globals.merid_width+1,  step = Globals.merid_width/2))
        plt.xlabel('System III West Longitude', size=15)
        plt.ylim(Globals.lat_target-Globals.para_width, Globals.lat_target+Globals.para_width)
        plt.yticks(np.arange(Globals.lat_target-Globals.para_width, Globals.lat_target+Globals.para_width+1, step = 5))
        plt.ylabel('Planetocentric Latitude', size=15)
        plt.tick_params(labelsize=12)
        cbar = plt.colorbar(extend='both', fraction=0.04, pad=0.05)#, orientation='horizontal')
        cbar.ax.tick_params(labelsize=12)
        cbar.ax.locator_params(nbins=6)
        cbar.set_label("Radiance (nW cm$^{-1}$ sr$^{-1}$)", size=15)
        # Save figure showing calibation method 
        plt.savefig(f"{dir}{wave}_radiance_maps.png", dpi=150, bbox_inches='tight')
        plt.savefig(f"{dir}{wave}_radiance_maps.pdf", dpi=500, bbox_inches='tight')
        # Clear figure to avoid overlapping between plotting subroutines
        plt.clf()

# Create a subplots figures with all filters
    fig, ax = plt.subplots(6, 2, figsize=(10, 12), sharex=True, sharey=True)
    iax = 0
    for ifilt in [0,10,11,12,5,4,6,7,8,9,3,2,1]:
        if ifilt < 6 or ifilt > 7:
            irow = [0,1,1,2,2,3,3,4,4,5,5]
            icol = [0,0,1,0,1,0,1,0,1,0,1]
            ititle = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)']
            # Remove the frame of the empty subplot
            ax[0][1].set_frame_on(False)
            ax[0][1].tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            # Get filter index for plotting spacecraft and calibrated data
            _, wavl, wave, _, _ = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=ifilt)
            # Set extreme values for mapping
            # spectrals[:, :, ifilt, 3] *= 1.e9 # ALREADY DONE IN THE PREVIOUS LOOP 
            max = np.nanmax(spectrals[:, :, ifilt, 3]*1.e9) 
            min = np.nanmin(spectrals[:, :, ifilt, 3]*1.e9)
            # subplot showing the regional radiance maps
            im= ax[irow[iax]][icol[iax]].imshow(spectrals[:, :, ifilt, 3]*1.e9, vmin=min, vmax=max, origin='lower', extent = [360,0,-90,90],  cmap='inferno')
            # ax[irow[iax]][icol[iax]].set_xlim(Globals.lon_target+Globals.merid_width, Globals.lon_target-Globals.merid_width)
            ax[irow[iax]][icol[iax]].set_xticks([]) if (iax < 9) else ax[irow[iax]][icol[iax]].set_xticks(np.arange(Globals.lon_target-Globals.merid_width, Globals.lon_target+Globals.merid_width+1,  step = Globals.merid_width/2))
            ax[irow[iax]][icol[iax]].set_ylim(Globals.lat_target-Globals.para_width, Globals.lat_target+Globals.para_width)
            ax[irow[iax]][icol[iax]].set_yticks(np.arange(Globals.lat_target-Globals.para_width, Globals.lat_target+Globals.para_width+1, step = 5))
            ax[irow[iax]][icol[iax]].tick_params(labelsize=14)
            ax[irow[iax]][icol[iax]].set_title(ititle[iax]+f"    {(wavl)}"+" $\mu$m", fontfamily='sans-serif', loc='left', fontsize=12)
            cbar = fig.colorbar(im, ax=ax[irow[iax]][icol[iax]], extend='both', fraction=0.04, pad=0.05)#, orientation='horizontal')
            cbar.ax.tick_params(labelsize=12)
            cbar.ax.locator_params(nbins=6)
            cbar.ax.set_title(r"[nW cm$^{-2}$ sr$^{-1}$ [cm$^{-1}$]$^{-1}$]", size=12, pad=8)
            iax+=1 
    plt.axes([0.15, 0.1, 0.8, 0.8], frameon=False) 
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel("System III West Longitude", size=18)
    plt.ylabel("Planetocentric Latitude", size=18)
    # Save figure showing calibation method 
    plt.savefig(f"{dir}all_filters_radiance_maps.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{dir}all_filters_radiance_maps.pdf", dpi=500, bbox_inches='tight')
    # Clear figure to avoid overlapping between plotting subroutines
    plt.close()


def PlotRegionalPerNight(dataset, spectrals, Nnight=4):
    """ Plot regional radiance maps per night (i.e., single spectra per filter) """

    print('Plotting regional radiance maps per night (i.e., single spectra per filter)...')

    # If subdirectory does not exist, create it
    dir = f'../outputs/{dataset}/maps_radiance_lat{Globals.lat_target}_lon{Globals.lon_target}_{Globals.latstep}x{Globals.lonstep}_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    for inight in range(Nnight):
        # Create a subplot figure with all filters per night
        fig, ax = plt.subplots(6, 2, figsize=(10, 12), sharex=True, sharey=True)
        iax = 0
        for ifilt in [0,10,11,12,5,4,6,7,8,9,3,2,1]:
            if ifilt < 6 or ifilt > 7:
                irow = [0,1,1,2,2,3,3,4,4,5,5]
                icol = [0,0,1,0,1,0,1,0,1,0,1]
                ititle = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)']
                # Remove the frame of the empty subplot
                ax[0][1].set_frame_on(False)
                ax[0][1].tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                # Get filter index for plotting spacecraft and calibrated data
                _, wavl, wave, _, _ = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=ifilt)
                # Set extreme values for mapping
                max = np.nanmax(spectrals[inight, :, :, ifilt, 3]*1.e9) 
                min = np.nanmin(spectrals[inight, :, :, ifilt, 3]*1.e9)
                # subplot showing the regional radiance maps
                im= ax[irow[iax]][icol[iax]].imshow(spectrals[inight, :, :, ifilt, 3]*1.e9, vmin=min, vmax=max, origin='lower', extent = [360,0,-90,90],  cmap='inferno')
                # ax[irow[iax]][icol[iax]].set_xlim(Globals.lon_target+Globals.merid_width, Globals.lon_target-Globals.merid_width)
                ax[irow[iax]][icol[iax]].set_xticks([]) if (iax < 9) else ax[irow[iax]][icol[iax]].set_xticks(np.arange(Globals.lon_target-Globals.merid_width, Globals.lon_target+Globals.merid_width+1,  step = Globals.merid_width/2))
                ax[irow[iax]][icol[iax]].set_ylim(Globals.lat_target-Globals.para_width, Globals.lat_target+Globals.para_width)
                ax[irow[iax]][icol[iax]].set_yticks(np.arange(Globals.lat_target-Globals.para_width, Globals.lat_target+Globals.para_width+1, step = 5))
                ax[irow[iax]][icol[iax]].tick_params(labelsize=14)
                ax[irow[iax]][icol[iax]].set_title(ititle[iax]+f"    {(wavl)}"+" $\mu$m", fontfamily='sans-serif', loc='left', fontsize=12)
                cbar = fig.colorbar(im, ax=ax[irow[iax]][icol[iax]], extend='both', fraction=0.04, pad=0.05)#, orientation='horizontal')
                cbar.ax.tick_params(labelsize=12)
                cbar.ax.locator_params(nbins=6)
                cbar.ax.set_title(r"[nW cm$^{-2}$ sr$^{-1}$ [cm$^{-1}$]$^{-1}$]", size=12, pad=8)
                iax+=1 
        plt.axes([0.15, 0.1, 0.8, 0.8], frameon=False) 
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.xlabel("System III West Longitude", size=18)
        plt.ylabel("Planetocentric Latitude", size=18)
        # Save figure showing calibation method 
        plt.savefig(f"{dir}all_filters_radiance_maps_night{inight}.png", dpi=150, bbox_inches='tight')
        plt.savefig(f"{dir}all_filters_radiance_maps_night{inight}.pdf", dpi=500, bbox_inches='tight')
        # Clear figure to avoid overlapping between plotting subroutines
        plt.close()

    