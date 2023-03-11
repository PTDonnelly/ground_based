import os
import re
from sqlite3 import Time
from turtle import color
import numpy as np
from copy import copy
from scipy.special import xlogy
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter, LogFormatterMathtext, LogLocator
from matplotlib import colors
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Rectangle
from math import *
import Globals
from Tools.SetWave import SetWave
from Tools.CalculateWindShear import CalculateWindShear
from Tools.RetrieveGasesNames import RetrieveGasesNames
from Tools.LatitudeConversion import convert_to_planetocentric
from Read.ReadPrior import ReadTemperatureGasesPriorProfile, ReadAerosolPriorProfile
from Read.ReadRetrievalOutputFiles import ReadprfFiles, ReadmreFiles, ReadaerFiles, ReadLogFiles, ReadmreParametricTest
from Read.ReadRetrievalOutputFiles import ReadAerFromMreFiles, ReadContributionFunctions, ReadAllForAuroraOverTime
from Read.ReadRetrievalOutputFiles import ReadPreviousWork
from Read.ReadExcelFiles import ReadExcelFiles
from Read.ReadDatFiles import ReadDatFiles
from Read.ReadZonalWind import ReadZonalWind
from Read.ReadSolarWindPredi import ReadSolarWindPredi

# Colormap definition
cmap = plt.get_cmap("turbo") #magma
# Definition selected retrieval tests names (for simpler labelisation on figures)
test_names = [
            r'Temp',
            r'Temp Aer ',
            r'Temp Aer C$_2$H$_2$ C$_2$H$_6$',
            r'Temp Aer NH$_3$',
            r'Temp Aer C$_2$H$_2$ C$_2$H$_6$ NH$_3$',
            r'Temp Aer C$_2$H$_2$ C$_2$H$_{6,p}$',
            r'Temp Aer C$_2$H$_{2,p}$ C$_2$H$_6$',
            r'Temp Aer C$_2$H$_{2,p}$ C$_2$H$_{6,p}$',
            r'Temp Aer C$_2$H$_{2,p}$ C$_2$H$_{6,p}$ NH$_{3}$', 
            r'Temp Aer C$_2$H$_{2}$ C$_2$H$_{6}$ NH$_{3,p}$', 
            r'Temp Aer NH$_{3,p}$', 
            r'Temp Aer C$_2$H$_{2,p}$ C$_2$H$_{6,p}$ NH$_{3,p}$', 
            ] 

#Deborah
# # If subdirectory does not exist, create it
# dir = '../retrievals/retrieved_figures/'
# if not os.path.exists(dir):
#     os.makedirs(dir)
# # Retrieval outputs directory path
# fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/"
# # Array of prior file names
# prior = ['jupiter_v2023']#, 'jupiter_v2016']

# retrieval_test = [
#             f"{iprior}_temp_ktable-highreso_no852_no887",
#             f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_ktable-highreso_no852_no887",
#             f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H6_ktable-highreso_no852_no887",
#             f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_NH3_ktable-highreso_no852_no887",
#             f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H6_NH3_ktable-highreso_no852_no887",
#             f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H6pknee02mbar_ktable-highreso_no852_no887",
#             f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H6_ktable-highreso_no852_no887",
#             f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H6pknee02mbar_ktable-highreso_no852_no887",
#             f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H6pknee02mbar_NH3_ktable-highreso_no852_no887",
#             f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H6_NH3p_ktable-highreso_no852_no887",
#             f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_NH3p_ktable-highreso_no852_no887",
#             f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H6pknee02mbar_NH3p_ktable-highreso_no852_no887",
#                     ]
# If subdirectory does not exist, create it
dir = '../retrievals/retrieved_figures/'
if not os.path.exists(dir):
    os.makedirs(dir)
# Retrieval outputs directory path
fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/retrievals_pdt11/experiment_2_maximum_emission_angle/"
# Array of prior file names
prior = ['ptd11']#, 'jupiter_v2016']
retrieval_test =[
    # Experiment 1
        # # "cmerid_flat5_jupiter2021_greycloud_0",
        # "cmerid_flat5_jupiter2021_greycloud_0_1p",
        # "cmerid_flat5_jupiter2021_greycloud_0_1p_27s",
        # "cmerid_flat5_jupiter2021_greycloud_0_1p_27s_26s",
        # # "cmerid_flat5_jupiter2021_nh3cloud_0",
        # "cmerid_flat5_jupiter2021_nh3cloud_0_1p",
        # "cmerid_flat5_jupiter2021_nh3cloud_0_1p_27s",
        # "cmerid_flat5_jupiter2021_nh3cloud_0_1p_27s_26s",
        # # "ctl_flat5_jupiter2021_greycloud_0",
        # "ctl_flat5_jupiter2021_greycloud_0_1p",
        # "ctl_flat5_jupiter2021_greycloud_0_1p_27s",
        # "ctl_flat5_jupiter2021_greycloud_0_1p_27s_26s",
        # # "ctl_flat5_jupiter2021_nh3cloud_0",
        # "ctl_flat5_jupiter2021_nh3cloud_0_1p",
        # "ctl_flat5_jupiter2021_nh3cloud_0_1p_27s",
        # "ctl_flat5_jupiter2021_nh3cloud_0_1p_27s_26s",
        # "limb_flat5_jupiter2021_greycloud_0",
        # "limb_flat5_jupiter2021_greycloud_0_1p",
        # "limb_flat5_jupiter2021_greycloud_0_1p_27s",
        # "limb_flat5_jupiter2021_greycloud_0_1p_27s_26s",
        # # "limb_flat5_jupiter2021_nh3cloud_0",
        # "limb_flat5_jupiter2021_nh3cloud_0_1p",
        # "limb_flat5_jupiter2021_nh3cloud_0_1p_27s",
        # "limb_flat5_jupiter2021_nh3cloud_0_1p_27s_26s"
    #Experiment 2
        "limb_50_flat5_jupiter2021_nh3cloud_0_1p_27s_26s",
        "limb_55_flat5_jupiter2021_nh3cloud_0_1p_27s_26s",
        "limb_60_flat5_jupiter2021_nh3cloud_0_1p_27s_26s",
        "limb_65_flat5_jupiter2021_nh3cloud_0_1p_27s_26s",
        "limb_70_flat5_jupiter2021_nh3cloud_0_1p_27s_26s",
        "limb_75_flat5_jupiter2021_nh3cloud_0_1p_27s_26s",
        "limb_80_flat5_jupiter2021_nh3cloud_0_1p_27s_26s",
        "limb_85_flat5_jupiter2021_nh3cloud_0_1p_27s_26s",
        "limb_90_flat5_jupiter2021_nh3cloud_0_1p_27s_26s"
]


# Plotting subroutines:
def PlotContributionFunction(over_axis="latitude"):

    # If subdirectory does not exist, create it
    dir = '../retrievals/retrieved_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Retrieval outputs directory path
    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/"
    # Array of prior file names
    prior = ['jupiter_v2023']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        
        ntest = len(retrieval_test)
        for itest in retrieval_test:
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{iprior}_selected_meridien_tests/{itest}/kk_figures/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            # Read retrieved profiles from .prf outputs files
            weighting_function, nwave, latitude, nlevel, pressure = ReadContributionFunctions(filepath=f"{fpath}{itest}", over_axis=over_axis)


            palette = copy(plt.get_cmap('magma'))
            palette.set_under('white', 0.5) # 1.0 represents not transparent
            fig, ax = plt.subplots(6, 2, figsize=(10, 12), sharex=True, sharey=True)
            iax = 0
            for iy in [0,8,9,10,5,4,6,7,3,2,1]:
                # Get filter index for plotting spacecraft and calibrated data
                if iy > 5:
                    _, wavl, wave, _, _ = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=iy+2)
                else: 
                    _, wavl, wave, _, _ = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=iy)
                irow = [0,1,1,2,2,3,3,4,4,5,5]
                icol = [0,0,1,0,1,0,1,0,1,0,1]
                ititle = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)']
                # Remove the frame of the empty subplot
                ax[0][1].set_frame_on(False)
                ax[0][1].tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

                im = ax[irow[iax]][icol[iax]].contourf(latitude, pressure, np.transpose(weighting_function[:, :, iy]), cmap="magma", vmax=1, vmin=0)
                ax[irow[iax]][icol[iax]].set_yscale("log")
                ax[irow[iax]][icol[iax]].grid()
                ax[irow[iax]][icol[iax]].invert_yaxis()
                ax[irow[iax]][icol[iax]].tick_params(labelsize=15)
                ax[irow[iax]][icol[iax]].set_xticks([-90, -60, -30, 0, 30, 60 ])
                ax[irow[iax]][icol[iax]].set_yticks([10000, 100, 1, 0.01])
                ax[irow[iax]][icol[iax]].set_title(ititle[iax]+f"    {(wavl)}"+" $\mu$m", fontfamily='sans-serif', loc='left', fontsize=12)
                im.cmap.set_under('w')
                im.set_clim(0.15)
                iax+=1 
            plt.axes([0.1, 0.1, 0.8, 0.8], frameon=False) 
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel("Planetographic Latitude", size=18)
            plt.ylabel("Pressure [mbar]", size=18)
            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            im.cmap.set_under('w')
            im.set_clim(0.15)
            cbar = plt.colorbar(im, cax=cbar_ax, extend='both', fraction=0.025, pad=0.05)#, orientation='horizontal')
            cbar.ax.tick_params(labelsize=12)
            cbar.ax.set_title("Normalized dR/dx", size=15, pad=15)
            # Save figure in the retrievals outputs directory
            plt.savefig(f"{subdir}contribution_fonctions.png", dpi=150, bbox_inches='tight')
            plt.close()






def PlotCheckPente():
    x = np.linspace(0,1,100)
    y = np.linspace(0,1,100)
    for f in [0.4, 0.41, 0.42, 0.43, 0.45, 0.46, 0.47, 0.48, 0.49, 0.5]: 
        result = x*y**(-(1-f)/f)
        plt.plot(x, result, label=f"{f}")
    plt.gca().invert_yaxis()
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.show()
    plt.close()


####### ChiSquare plotting and mapping routines ####### 
def PlotChiSquareOverNy(over_axis):

    print('Plotting NEMESIS ChiSquare over latitude...')
     # If subdirectory does not exist, create it
    dir = '../retrievals/retrieved_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Retrieval outputs directory path
    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/"
    # Array of prior file names
    prior = ['jupiter_v2021']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        # If retrieval test comparison subdirectory does not exist, create it
        # subdir = f"{dir}/{iprior}_temp_aerosol1-10mu_greycloud_70-300mbar/chisquare_comparison/"
        # subdir = f"{dir}/{iprior}_temp_aerosol1-10mu_paraH2_greycloud_70-300mbar/chisquare_comparison/"
        subdir = f"{dir}/{iprior}_all_tests/chisquare_comparison/"
        if not os.path.exists(subdir):
                os.makedirs(subdir)
        # List of retrieval tests for comparison...
        retrieval_test = [
            # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_NH3pt_PH3pt_lat80S_no852_no887_reduce", 
                        # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_NH3pt_PH3_lat80S_no852_no887_reduce", 
                        # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_NH3_PH3pt_lat80S_no852_no887_reduce", 
                        # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_NH3pt_lat80S_no852_no887_reduce", 
                        # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_NH3_lat80S_no852_no887_reduce", 
                        # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_NH3_lat80S_no852_no887_reduce"
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_NH3_no852_no887",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_NH3_no852_no887_epsilon-prior_7",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_NH3pt_no852_no887",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_NH3_PH3pt_no852_no887",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_NH3pt_PH3pt_no852_no887",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_NH3_no852_no887_aprmodif",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_no852_no887_aprmodif",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_NH3p_no852_no887_aprmodif",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_NH3pt_no852_no887_aprmodif",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_knee02mbar_NH3p_no852_no887_aprmodif",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6p_knee02mbar_NH3p_no852_no887_aprmodif",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6p_knee02mbar_NH3p_vmrfixed_no852_no887_aprmodif",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6p_knee02mbar_NH3p4_vmrfixed_no852_no887_aprmodif",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6pknee02mbar_NH3p_vmr-fsh10pourcent_no852_no887_aprmodif",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6pknee02mbar_NH3p_vmrfixed_no852_no887_aprmodif",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6pknee02mbar_NH3p4_vmrfixed_no852_no887_aprmodif",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6pknee02mbar_NH3p_fshvary_no852_no887_aprmodif",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6pknee02mbar_NH3p_fshfix_no852_no887_aprmodif",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H4-C2H6_NH3p_fshfix_no852_no887_aprmodif",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6_NH3p_fshfix_no852_no887_aprmodif",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H4-C2H6pknee02mbar_NH3p_fshfix_no852_no887_aprmodif",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H4-C2H6_NH3_no852_no887",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6pknee02mbar_NH3_no852_no887",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6pknee02mbar_no852_no887",
                        f"{iprior}_temp_aerosol1-1mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6pknee02mbar_no852_no887"
                        ]
        ntest = len(retrieval_test)
        # Loop over each prior used for retrievals tests
        for itest in retrieval_test:
            # # If retrieval test subdirectory does not exist, create it
            # subdir = f"{dir}{itest}/chisq_ny/"
            # if not os.path.exists(subdir):
            #     os.makedirs(subdir)
            # Read retrieved profiles from .prf outputs files
            chisquare, latitude, _ = ReadLogFiles(filepath=f"{fpath}{itest}", over_axis=over_axis)
            maxchi = np.nanmax(chisquare)
            # Plot Figure of chisq/ny over latitude
            fig = plt.subplots(1, 1, figsize=(12, 6))
            plt.plot(latitude, chisquare, lw=2)
            plt.grid()
            if maxchi > 1:
                plt.ylim(0, ceil(maxchi))
                plt.yticks(np.arange(ceil(maxchi)+1))
            else:
                plt.ylim(0, 1)
                plt.yticks(np.arange(0, 1.01, 0.1))
            plt.tick_params(labelsize=15)        
            plt.ylabel('\u03C7'r'$^{2}/N_y$', size=20)
            if over_axis=="longitude":
                plt.xlabel("System III West Longitude", size=20)
            elif over_axis=="latitude":
                plt.xlabel("Planetographic Latitude", size=20)
            # Save figure in the retrievals outputs directory
            plt.savefig(f"{subdir}{itest}_chisquare.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{subdir}{itest}_chisquare.eps", dpi=100)
            # Close figure to avoid overlapping between plotting subroutines
            plt.close()

def PlotChiSquareOverNySuperpose(over_axis):

    print('Plotting NEMESIS ChiSquare over latitude (superpose figure of several tests) ...')
    
    for iprior in prior:
        # Loop over each prior used for retrievals tests
        # If retrieval test subdirectory does not exist, create it
        subdir = f"{dir}{iprior}_selected_meridien_tests_experiment2/chisquare_comparison/"
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        ntest = len(retrieval_test)
        # Create the array to store chisquare over latitude for each test 
        maxchisg = []
        # Plot Figure of chisq/ny over latitude
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        # Loop over each prior used for retrievals tests
        for i, itest in enumerate(retrieval_test):
            print(f"        ...{itest}")
            col = cmap(i/ntest)
            # Read retrieved profiles from .prf outputs files
            chisquare, latitude, nlat = ReadLogFiles(filepath=f"{fpath}{itest}", over_axis=over_axis)
            maxchisg.append(np.nanmax(chisquare))
            axes[0].plot(latitude, chisquare, lw=2, color = col, label=f"{itest}"[:7])#label=f"{test_names[i]}")#, label=r'T_aer10${\mu}$m_C$_2$H$_2$p-C$_2$H$_4$-C$_2$H$_6$p_'+f"{itest}"[69:-7])
            # plt.plot(latitude, chisquare, lw=2, label=f"{itest}"[14:32]+"C2H2_C2H6_NH3")
            # plt.plot(latitude, chisquare, lw=2, label=f"{itest}"[14:32])
            # plt.plot(latitude, chisquare, lw=2, label=f"{itest}"[14:])
            # if np.nanmax(chisquare) <= 1.5:
            axes[1].plot(latitude, chisquare, lw=2, color = col, label=f"{test_names[i]}")
            # axes[1].legend()
        axes[0].set_title("(a)", fontfamily='sans-serif', loc='left', fontsize=15)
        maxchi = np.nanmax(maxchisg)
        if maxchi > 1:
            axes[0].set_ylim(0, ceil(maxchi))
            axes[0].set_yticks(np.arange(ceil(maxchi)+1))
        else:
            axes[0].set_ylim(0, 1)
            axes[0].set_yticks(np.arange(0, 1.01, 0.1))
        axes[0].set_xlim(-90, 90)
        axes[0].tick_params(labelsize=15)
        axes[0].grid()

        axes[1].set_title("(b)", fontfamily='sans-serif', loc='left', fontsize=15)
        axes[1].set_ylim(0,1.0,0.1)
        axes[1].set_xlim(-90, 90)
        axes[1].tick_params(labelsize=15) 
        axes[1].grid()
        handles, labels = axes[0].get_legend_handles_labels() 

        plt.axes([0.1, 0.08, 0.8, 0.85], frameon=False) 
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.legend(handles=handles, labels=labels, bbox_to_anchor=(0,1.02,1,0.2), loc="lower left", mode="expand", ncol=3, fontsize=10)       
        plt.ylabel('\u03C7'r'$^{2}/N_y$', size=20)
        if over_axis=="longitude":
            plt.xlabel("System III West Longitude", size=20)
        elif over_axis=="latitude":
            plt.xlabel("Planetographic Latitude", size=20)
        # Save figure in the retrievals outputs directory
        plt.savefig(f"{subdir}limb_chisquare_comparison.png", dpi=150, bbox_inches='tight')
        #plt.savefig(f"{subdir}{itest}_chisquare.eps", dpi=100)
        # Close figure to avoid overlapping between plotting subroutines
        plt.close()

def PlotChiSquareMap(over_axis="2D"):

    print('Plotting NEMESIS retrieved ChiSquare map...')
    # If subdirectory does not exist, create it
    dir = '../retrievals/retrieved_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Retrieval outputs directory path
    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/retrieval_GreatRedSpot/"
    # Array of prior file names
    prior = ['jupiter_v2021']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        retrieval_test = [
                        # f"jupiter_v2021_temp_aerosol1-10mu-800mbar-05scale-01_GRS_no852_no887",
                        # f"jupiter_v2021_temp_aerosol1-10mu-800mbar-05scale-01_NH3_GRS_no852_no887",
                        "jupiter_v2021_temp_aerosol1-10mu-800mbar-05scale-01_NH3_GRS_halfdeg_no852_no887"
                        ]
        ntest = len(retrieval_test)
        # Loop over each prior used for retrievals tests
        for itest in retrieval_test:
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{iprior}_selected_meridien_tests/{itest}/GRS_chisquare_map/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            # Read retrieved profiles from .prf outputs files
            chisquare, latitude, nlat, longitude, nlon = ReadLogFiles(filepath=f"{fpath}{itest}", over_axis="2D")
            maxchi = np.nanmax(chisquare)
            plt.figure(figsize=(8, 6))
            im = plt.imshow(chisquare[:,:], vmin=0, vmax=ceil(maxchi)-1, cmap="magma",
                            origin='lower', extent=[longitude[0],longitude[-1],latitude[0],latitude[-1]])
            plt.xlabel('System III West Longitude', size=15)
            plt.ylabel('Planetographic Latitude', size=15)
            plt.tick_params(labelsize=12)
            cbar = plt.colorbar(im, extend='both', fraction=0.025, pad=0.05)#, orientation='horizontal')
            cbar.ax.tick_params(labelsize=12)
            cbar.ax.locator_params(nbins=6)
            cbar.ax.set_title("$\chi^{2}/N_y$", size=15, pad=15)
            # Save figure showing calibation method 
            plt.savefig(f"{subdir}chisquare_maps.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{dir}{wave}_radiance_maps.eps", dpi=900)
            # Clear figure to avoid overlapping between plotting subroutines
            plt.close()

def stat_test():

    print('Statistical calculation to determine the best retrieval test ...')
     # If subdirectory does not exist, create it
    dir = '../retrievals/retrieved_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Retrieval outputs directory path
    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/"
    # Array of prior file names
    prior = ['jupiter_v2023']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        # If retrieval test subdirectory does not exist, create it
        subdir = f"{dir}{iprior}_selected_meridien_tests/stats_calcul/"
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        retrieval_test = [
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H6pknee02mbar_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H6pknee02mbar_NH3_ktable-highreso_no852_no887",
                        ]
        ntest = len(retrieval_test)
        nconv = 11
        chisquare = np.empty((ntest, 177))
        for i, itest in enumerate(retrieval_test):
            # Read retrieved profiles from .prf outputs files
            chisquare[i,:], latitude, nlat = ReadLogFiles(filepath=f"{fpath}{itest}", over_axis='latitude')


        # NEMESIS does the Chi-Square Goodness of fit test which is a non-parametric statistical hypothesis test 
        # that's used to determine how considerably the observed value of an event differs from 
        # the expected value. it helps us check whether a variable comes from a certain distribution 
        # or if a sample represents a population. 
        plt.plot(latitude, chisquare[0,:]/chisquare[1,:])
        plt.xlim(-90, 90)
        plt.grid()
        plt.show()
        # ANOVA give a F scores so: Hypotheses of interest are about the differences between population means.
        fvalue, pvalue = stats.f_oneway(chisquare[0,:], chisquare[1,:])#, chisquare[3,:], chisquare[4,:], chisquare[5,:], chisquare[6,:], chisquare[7,:], chisquare[8,:])
        print("         ... ANOVA calculation with all tests") 
        print("                      fvalue =", fvalue, "pvalue =", pvalue)

        # t-test calculation: Hypothesis testing for small sample size.
        print("         ... Individual t-test for each retrieval tests")
        for i, itest in enumerate(retrieval_test):
            # Normalization of the chisquare distribution
            arr = chisquare[i,:]
            tmin, tmax = 0, 1
            diff = tmax - tmin
            diff_arr = np.max(arr) - np.min(arr)
            norm_arr = []
            for ilat, chisq in enumerate(arr):
                    tmp = ( ( (ilat - min(arr)) * diff) / diff_arr) + tmin
                    norm_arr.append(tmp)
            # t-test calculation
            results = stats.ttest_1samp(a=arr, popmean=0.5)
            # print(itest)
            print("                     ",results)
        print("         ... Individual t-test for two retrieval tests")
        results_2 = stats.ttest_ind(a=chisquare[0,:],b=chisquare[1,:],equal_var=False)
        print("                     ",results_2)
        # print("         ... Individual t-test for two retrieval tests")
        # results_2 = stats.ttest_ind(a=chisquare[2,:],b=chisquare[1,:],equal_var=False)
        # print("                     ",results_2)
        # print("         ... Individual t-test for two retrieval tests")
        # results_2 = stats.ttest_ind(a=chisquare[4,:],b=chisquare[2,:],equal_var=False)
        # print("                     ",results_2)
        # print("         ... Individual t-test for two retrieval tests")
        # results_2 = stats.ttest_ind(a=chisquare[1,:],b=chisquare[1,:],equal_var=False)
        # print("                     ",results_2)



####### Temperature plotting and mapping routines ####### 
def PlotRetrievedTemperature(over_axis):

    print('Plotting NEMESIS retrieved temperature over latitude...')
     # If subdirectory does not exist, create it
    dir = '../retrievals/retrieved_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Retrieval outputs directory path
    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/"
    # Array of prior file names
    prior = ['jupiter_v2021']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        retrieval_test = [
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6pknee02mbar_NH3p_fshfix_no852_no887_aprmodif"
                        ]
        ntest = len(retrieval_test)
        # Loop over each prior used for retrievals tests
        for itest in retrieval_test:
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{itest}/meridians/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            # # Read profile data from NEMESIS prior file 
            # _, prior_p, prior_temperature, prior_error, _, _, nlevel, _ = ReadTemperatureGasesPriorProfile(f"{fpath}{itest}/core_1/")
            # Read retrieved profiles from .prf outputs files
            temperature, _, latitude, _, pressure, _, nlevel, _, _= ReadprfFiles(filepath=f"{fpath}{itest}", over_axis=over_axis)
            # Fill an 2D arrays with the prior temperature and temperature error profiles
            # prior_temp = np.empty((nlevel, nlat))
            # prior_err = np.empty((nlevel, nlat))
            # for ilat in range(nlat):
            #     prior_temp[:, ilat] = prior_temperature[:]
            #     prior_err[:, ilat] = prior_error[:]
            for ilev in range(nlevel):
                fig, axes = plt.subplots(1, 1, figsize=(10, 6), sharex=True, sharey=True)
                # axes.plot(latitude, prior_temp[ilev, :], lw=2, label=f"{iprior}", color='orange')
                # axes.fill_between(latitude, prior_temp[ilev, :]-prior_err[ilev, :], prior_temp[ilev, :]+prior_err[ilev, :], color='orange', alpha=0.2)        
                axes.plot(latitude, temperature[ilev, :].T, lw=2)
                axes.grid()
                #axes.legend(loc="upper center", fontsize=10, handletextpad=0, handlelength=0, markerscale=0)
                axes.tick_params(labelsize=15)
                # Add a big axis 
                plt.axes([0.1, 0.09, 0.8, 0.8], frameon=False) 
                plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)        
                plt.ylabel(f"Retrieved Temperature at {pressure[ilev]} mbar", size=20)
                if over_axis=="longitude":
                    plt.xlabel("System III West Longitude", size=20)
                elif over_axis=="latitude":
                    plt.xlabel("Planetographic Latitude", size=20)
                # Save figure in the retrievals outputs directory
                plt.savefig(f"{subdir}{itest}_retrieved_temperature_{pressure[ilev]}mbar.png", dpi=150, bbox_inches='tight')
                #plt.savefig(f"{subdir}{itest}_retrieved_temperature_{pressure[ilev]}mbar.eps", dpi=100)
                # Close figure to avoid overlapping between plotting subroutines
                plt.close()

def PlotRetrievedTemperatureProfile(over_axis):

    print('Plotting NEMESIS retrieved temperature profiles...')
    # If subdirectory does not exist, create it
    dir = '../retrievals/retrieved_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Retrieval outputs directory path
    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/"
    # Array of prior file names
    prior = ['jupiter_v2021']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        retrieval_test = [
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6pknee02mbar_NH3p_fshfix_no852_no887_aprmodif"
                        ]
        ntest = len(retrieval_test)
        # Loop over each retrieval tests for the current prior file
        for itest in retrieval_test:
            # Read profile data from NEMESIS prior file 
            _, prior_p, prior_temperature, prior_err, _, _, _, _ = ReadTemperatureGasesPriorProfile(f"{fpath}{itest}/core_1/")
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{itest}/profiles/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            # Read retrieved profiles from .prf outputs files
            temperature, _, latitude, _, pressure, nlat, _, _, _ = ReadprfFiles(filepath=f"{fpath}{itest}", over_axis=over_axis)
            # Plotting retrieved temperature profile for each latitude
            for ilat in range(nlat):
                fig, axes = plt.subplots(1, 1, figsize=(7, 10), sharex=True, sharey=True)        
                axes.plot(prior_temperature, prior_p, lw=2, label=f"{iprior}", color='orange')
                axes.fill_betweenx(prior_p, prior_temperature-prior_err, prior_temperature+prior_err, color='orange', alpha=0.2)
                axes.plot(temperature[:, ilat], pressure, lw=2, label=f"Retrieved Temperature at {latitude[ilat]}")
                axes.set_yscale('log')
                axes.invert_yaxis()
                axes.grid()
                axes.legend(loc="upper right", fontsize=15)
                axes.tick_params(labelsize=15)
                # Add a big axis 
                plt.axes([0.1, 0.09, 0.8, 0.8], frameon=False) 
                plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)        
                plt.xlabel(f"Temperature [K]", size=20)
                plt.ylabel(f"Presssure [mbar]", size=20)
                # Save figure in the retrievals outputs directory
                plt.savefig(f"{subdir}{itest}_retrieved_temperature_profile_at_{latitude[ilat]}.png", dpi=150, bbox_inches='tight')
                #plt.savefig(f"{subdir}{itest}_retrieved_temperature_profile_at_{latitude[ilat]}.eps", dpi=100)
                # Close figure to avoid overlapping between plotting subroutines
                plt.close()

def PlotRetrievedTemperatureProfileSuperpose(over_axis):

    print('Plotting NEMESIS retrieved temperature profiles (superpose figure of several tests)...')
    # If subdirectory does not exist, create it
    dir = '../retrievals/retrieved_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Retrieval outputs directory path
    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/"
    # Array of prior file names
    prior = ['jupiter_v2021']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        # If retrieval test comparison subdirectory does not exist, create it
        # subdir = f"{dir}/{iprior}_temp_aerosol1-10mu_greycloud_70-300mbar/chisquare_comparison/"
        # subdir = f"{dir}/{iprior}_temp_aerosol1-10mu_paraH2_greycloud_70-300mbar/chisquare_comparison/"
        subdir = f"{dir}/zonal_parametric_hydrocb_NH3/temperature_profile_comparison/"
        # subdir = f"{dir}/{iprior}_temp_aerosol1-10or1mu_nospecies/chisquare_comparison/"
        # subdir = f"{dir}/{iprior}_temp_aerosol1-10mu_scale05or075or1/chisquare_comparison/"
        if not os.path.exists(subdir):
                os.makedirs(subdir)
        # List of retrieval tests for comparison...
        retrieval_test = [f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H4-C2H6_NH3_no852_no887",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H4-C2H6_NH3p_fshfix_no852_no887_aprmodif",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H4-C2H6pknee02mbar_NH3p_fshfix_no852_no887_aprmodif",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6_NH3p_fshfix_no852_no887_aprmodif",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6pknee02mbar_NH3_no852_no887",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6pknee02mbar_no852_no887",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6pknee02mbar_NH3p_fshfix_no852_no887_aprmodif"
        ]
        ntest = len(retrieval_test)
        
        # Plotting retrieved temperature profile for each latitude
        for ilat in range(176):
            fig, axes = plt.subplots(1, 1, figsize=(7, 10))
            # Loop over each prior used for retrievals tests
            for i, itest in enumerate(retrieval_test):
                col = cmap(i/ntest)
                # Read retrieved profiles from .prf outputs files
                temperature, _, latitude, _, pressure, _, _, _, _ = ReadprfFiles(filepath=f"{fpath}{itest}", over_axis=over_axis)
                # Plot 
                axes.plot(temperature[:, ilat], pressure, lw=2, label=r'T_aer10${\mu}$m_'+f"{itest}"[52:], color = col)
            # Read profile data from NEMESIS prior file 
            _, prior_p, prior_temperature, prior_err, _, _, _, _ = ReadTemperatureGasesPriorProfile(f"{fpath}{itest}/core_1/")
            # Plot the prior only for the last itest (because it's the same for all itest)
            axes.plot(prior_temperature, prior_p, lw=2, label=f"{iprior} at {latitude[ilat]}", color='black')
            axes.fill_betweenx(prior_p, prior_temperature-prior_err, prior_temperature+prior_err, color='black', alpha=0.2)
            axes.set_yscale('log')
            axes.invert_yaxis()
            axes.grid()
            axes.legend(loc="upper right", fontsize=15)
            axes.tick_params(labelsize=15)
            # Add a big axis 
            plt.axes([0.1, 0.09, 0.8, 0.8], frameon=False) 
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)        
            plt.xlabel(f"Temperature [K]", size=20)
            plt.ylabel(f"Presssure [mbar]", size=20)
            # Save figure in the retrievals outputs directory
            plt.savefig(f"{subdir}temperature_profile_at_{latitude[ilat]}.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{subdir}{itest}_retrieved_temperature_profile_at_{latitude[ilat]}.eps", dpi=100)
            # Close figure to avoid overlapping between plotting subroutines
            plt.close()

def PlotRetrievedTemperatureCrossSection(over_axis):

    print('Plotting NEMESIS retrieved temperature cross-section...')
    # If subdirectory does not exist, create it
    dir = '../retrievals/retrieved_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    #  Load Jupiter zonal jets data to determine belts and zones location
    ejets_c, wjets_c, nejet, nwjet = ReadZonalWind("../inputs/jupiter_jets.dat")
    # Retrieval outputs directory path
    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/"
    # Array of prior file names
    prior = ['jupiter_v2023']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        retrieval_test = [
            f"{iprior}_temp_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H6_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_NH3_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H6_NH3_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H6pknee02mbar_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H6_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H6pknee02mbar_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H6pknee02mbar_NH3_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H6_NH3p_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_NH3p_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H6pknee02mbar_NH3p_ktable-highreso_no852_no887",
                    ]
        ntest = len(retrieval_test)
        # Loop over each retrieval tests for the current prior file
        for itest in retrieval_test:
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{iprior}_selected_meridien_tests/{itest}/temperature_section/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            # Read retrieved profiles from .prf outputs files
            temperature, _, latitude, _, pressure, _, _, _, _ = ReadprfFiles(filepath=f"{fpath}{itest}", over_axis=over_axis)
            # Determine extreme values of temperature and levels for mapping
            max = 185
            min = 95
            levels_cmap = np.linspace(min, max, num=37, endpoint=True)
            levels = np.linspace(90, 190, num=10, endpoint=True)

            # Mapping the temperature cross-section with zind location
            plt.figure(figsize=(8, 6))
            im = plt.contourf(latitude, pressure, temperature, cmap='viridis', levels=levels_cmap)
            plt.contour(latitude, pressure, temperature, levels=levels_cmap, colors="white", linewidths=0.7)
            for iejet in range(0,nejet):
                plt.plot([ejets_c[iejet],ejets_c[iejet]],[0.1, 1000],color='black',linestyle="dashed")
            for iwjet in range(0,nwjet):
                plt.plot([wjets_c[iwjet],wjets_c[iwjet]],[0.1, 1000],color='black',linestyle="dotted")
            plt.ylim(0.1, 1000)
            # plt.xlim(-80, 80)
            plt.yscale('log')
            plt.gca().invert_yaxis()
            plt.tick_params(labelsize=15)        
            if over_axis=="longitude":
                plt.xlabel("System III West Longitude", size=20)
            elif over_axis=="latitude":
                plt.xlabel("Planetographic Latitude", size=20)
            plt.ylabel(f"Presssure [mbar]", size=20)
            cbar = plt.colorbar(im, extend='both', fraction=0.046, pad=0.05, orientation='vertical', format="%.0f")
            cbar.ax.tick_params(labelsize=15)
            cbar.ax.get_yaxis().set_ticks(np.arange(min, max+1, 10))
            # cbar.set_ticklabels(np.arange(95, 186, 5))
            cbar.ax.set_title("[K]",fontsize=15)   
            # Save figure in the retrievals outputs directory
            plt.savefig(f"{subdir}retrieved_temperature_zonal_wind.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{subdir}{itest}_retrieved_temperature_zonal_wind.eps", dpi=100)
            # Close figure to avoid overlapping between plotting subroutines
            plt.close()


            # Mapping the temperature cross-section alone
            plt.figure(figsize=(8, 6))
            im = plt.contourf(latitude, pressure, temperature, cmap='viridis', levels=levels_cmap)
            plt.contour(latitude, pressure, temperature, levels=levels_cmap, colors="white", linewidths=0.5)            
            plt.ylim(0.1, 1000)
            # plt.xlim(-80, 80)
            plt.yscale('log')
            plt.gca().invert_yaxis()
            plt.tick_params(labelsize=15)     
            if over_axis=="longitude":
                plt.xlabel("System III West Longitude", size=20)
            elif over_axis=="latitude":
                plt.xlabel("Planetographic Latitude", size=20)
            plt.ylabel(f"Presssure [mbar]", size=20)
            cbar = plt.colorbar(im, extend='both', fraction=0.046, pad=0.05, orientation='vertical', format="%.0f")
            cbar.ax.tick_params(labelsize=15)
            cbar.ax.get_yaxis().set_ticks(np.arange(min, max+1, 10))

            cbar.ax.set_title("[K]",fontsize=15)   
            # Save figure in the retrievals outputs directory
            plt.savefig(f"{subdir}retrieved_temperature.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{subdir}{itest}_retrieved_temperature.eps", dpi=100)
            # Close figure to avoid overlapping between plotting subroutines
            plt.close()

def PlotRetrievedTemperatureMaps(over_axis="2D"):

    print('Plotting NEMESIS retrieved temperature maps...')
     # If subdirectory does not exist, create it
    dir = '../retrievals/retrieved_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Retrieval outputs directory path
    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/retrieval_GreatRedSpot/"
    # Array of prior file names
    prior = ['jupiter_v2021']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        retrieval_test = [
                        # f"jupiter_v2021_temp_aerosol1-10mu-800mbar-05scale-01_GRS_no852_no887",
                        # f"jupiter_v2021_temp_aerosol1-10mu-800mbar-05scale-01_NH3_GRS_no852_no887",
                        "jupiter_v2021_temp_aerosol1-10mu-800mbar-05scale-01_NH3_GRS_halfdeg_no852_no887"
                        ]
        ntest = len(retrieval_test)
        # Loop over each prior used for retrievals tests
        for itest in retrieval_test:
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{iprior}_selected_meridien_tests/{itest}/GRS_temperature_map/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            print(f" ... {itest} ...")
            # Read retrieved profiles from .prf outputs files
            temperature, gases, latitude, longitude, height, pressure, ncoor, nlevel, nlat, nlon, ngas, gases_id = ReadprfFiles(filepath=f"{fpath}{itest}", over_axis=over_axis)
            # Read profile data from NEMESIS prior file 
            _, prior_p, temp_prior, prior_err, _, _, _, _ = ReadTemperatureGasesPriorProfile(filepath=f"{fpath}{itest}/core_1/")

            # Determine extreme values of temperature and levels for mapping
            tmax = np.nanmax(temperature)
            tmin = np.nanmin(temperature)
            levels_cmap = np.linspace(tmin, 200, num=20, endpoint=True)
            levels = np.linspace(tmin, 200, num=10, endpoint=True)

            # print("      ... retrieved temperature meridian cross-section")
            # # Mapping the meridian temperature cross-section
            # lon_index = (longitude == 157.5)
            # tempkeep = temperature[:, :, lon_index]
            # plt.figure(figsize=(8, 6))
            # im = plt.contourf(latitude, pressure, tempkeep[:, :, 0], cmap='viridis', levels=levels_cmap)
            # # plt.contour(latitude, pressure, tempkeep[:, :, 0], levels=levels, colors="white")
            # plt.ylim(0.001, 1000)
            # # plt.xlim(-80, 80)
            # plt.yscale('log')
            # plt.gca().invert_yaxis()
            # plt.tick_params(labelsize=15)        
            # plt.xlabel("Planetographic Latitude", size=20)
            # plt.ylabel(f"Presssure [mbar]", size=20)
            # plt.title(f"Great Red Spot structure at {float(longitude[lon_index])}"+"$^{\circ}$W")
            # cbar = plt.colorbar(im, extend='both', fraction=0.046, pad=0.05, orientation='vertical', format="%.0f")
            # cbar.ax.tick_params(labelsize=15)
            # cbar.set_label("Retrieved Temperature [K]", fontsize=20)   
            # # Save figure in the retrievals outputs directory
            # plt.savefig(f"{subdir}{itest}_meridian_cross_section_at_lon{float(longitude[lon_index])}.png", dpi=150, bbox_inches='tight')
            # #plt.savefig(f"{subdir}{itest}_meridian_cross_section_at_lon{float(longitude[lon_index])}.eps", dpi=100)
            # # Close figure to avoid overlapping between plotting subroutines
            # plt.close()

            # print("      ... retrieved temperature zonal cross-section")
            # # Mapping the zonal temperature cross-section
            # lat_index = (latitude == -20.5)
            # tempkeep = temperature[:, lat_index, :]
            # plt.figure(figsize=(8, 6))
            # im = plt.contourf(longitude, pressure, tempkeep[:, 0, :], cmap='viridis', levels=levels_cmap)
            # # plt.contour(longitude, pressure, tempkeep[:, :, 0], levels=levels, colors="white")
            # plt.ylim(0.001, 1000)
            # # plt.xlim(-80, 80)
            # plt.yscale('log')
            # plt.gca().invert_yaxis()
            # plt.gca().invert_xaxis()
            # plt.tick_params(labelsize=15)        
            # plt.xlabel("System III West longitude", size=20)
            # plt.ylabel(f"Presssure [mbar]", size=20)
            # plt.title(f"Great Red Spot structure at {float(latitude[lat_index])}"+"$^{\circ}$")
            # cbar = plt.colorbar(im, extend='both', fraction=0.046, pad=0.05, orientation='vertical', format="%.0f")
            # cbar.ax.tick_params(labelsize=15)
            # cbar.set_label("Retrieved Temperature [K]", fontsize=20)   
            # # Save figure in the retrievals outputs directory
            # plt.savefig(f"{subdir}{itest}_zonal_cross_section_at_lat{float(latitude[lat_index])}.png", dpi=150, bbox_inches='tight')
            # #plt.savefig(f"{subdir}{itest}_zonal_cross_section_at_lat{float(latitude[lat_index])}.eps", dpi=100)
            # # Close figure to avoid overlapping between plotting subroutines
            # plt.close()

            print("      ... retrieved temperature maps at several pressure levels")
            # Mapping the vertical temperature cross-section
            ptarget = [5, 80, 180, 380, 440, 500, 550, 600, 700, 800, 1000]

            fig, ax = plt.subplots(6, 2, figsize=(10, 12), sharex=True, sharey=True)
            iax = 0
            for ipressure in range(len(ptarget)):
                irow = [0,1,1,2,2,3,3,4,4,5,5]
                icol = [0,0,1,0,1,0,1,0,1,0,1]
                ititle = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)']
                # Remove the frame of the empty subplot
                ax[0][1].set_frame_on(False)
                ax[0][1].tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                # find pressure index 
                ind_pres = np.where(pressure >= ptarget[ipressure])
                # set extreme values 
                tmax = np.nanmax(temperature[ind_pres[0][-1], :, :])
                tmin = np.nanmin(temperature[ind_pres[0][-1], :, :])
                levels_cmap = np.linspace(tmin, tmax, num=20, endpoint=True)
                # subplot showing the regional radiance maps
                im = ax[irow[iax]][icol[iax]].imshow(temperature[ind_pres[0][-1], :, :], cmap='viridis', vmin=tmin, vmax=tmax, # levels=levels_cmap,
                                                        origin='lower', extent=[longitude[0],longitude[-1],latitude[0],latitude[-1]])
                ax[irow[iax]][icol[iax]].tick_params(labelsize=14)
                ax[irow[iax]][icol[iax]].set_title(ititle[iax]+f"    {int(ptarget[ipressure])} mbar", fontfamily='sans-serif', loc='left', fontsize=12)
                cbar = fig.colorbar(im, ax=ax[irow[iax]][icol[iax]], extend='both', fraction=0.04, pad=0.05, format="%.0f")#, orientation='horizontal')
                cbar.ax.tick_params(labelsize=12)
                cbar.ax.locator_params(nbins=6)
                cbar.ax.set_title("[K]", fontsize=12, pad=10)
                iax+=1 
            plt.axes([0.15, 0.1, 0.8, 0.8], frameon=False) 
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel("System III West Longitude", size=18)
            plt.ylabel("Planetographic Latitude", size=18)
            # Save figure showing calibation method 
            plt.savefig(f"{subdir}{itest}_temperature_maps_at_11_pressure_levels.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{subdir}{itest}_temperature_maps_at_11_pressure_levels.eps", dpi=900)
            # Clear figure to avoid overlapping between plotting subroutines
            plt.close()

            # Create a subplot figure of temperature residual with all filters
            print("      ... residual temperature error maps at several pressure levels")
            fig, ax = plt.subplots(6, 2, figsize=(10, 12), sharex=True, sharey=True)
            iax = 0
            for ipressure in range(len(ptarget)):
                irow = [0,1,1,2,2,3,3,4,4,5,5]
                icol = [0,0,1,0,1,0,1,0,1,0,1]
                ititle = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)']
                # Remove the frame of the empty subplot
                ax[0][1].set_frame_on(False)
                ax[0][1].tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                # find pressure index 
                ind_pres = np.where(pressure >= ptarget[ipressure])
                # Set extreme values temperature[ifilt, :]-temp_prior[ifilt, :])*100/temp_prior[ifilt, :]
                tempmax = np.nanmax((temperature[ind_pres[0][-1], :, :]-temp_prior[ind_pres[0][-1]])*100/temp_prior[ind_pres[0][-1]]) 
                tempmin = np.nanmin((temperature[ind_pres[0][-1], :, :]-temp_prior[ind_pres[0][-1]])*100/temp_prior[ind_pres[0][-1]])
                if tempmax > 0 and tempmin < 0:
                    norm = colors.TwoSlopeNorm(vmin=tempmin, vmax=tempmax, vcenter=0)
                elif tempmax < 0 and tempmin < 0:
                    norm = colors.TwoSlopeNorm(vmin=tempmin, vmax=0.5, vcenter=0)
                elif tempmax > 0 and tempmin > 0:
                    norm = colors.TwoSlopeNorm(vmin=-0.5, vmax=tempmax, vcenter=0)
                # subplot showing the regional temperature maps
                im = ax[irow[iax]][icol[iax]].imshow((temperature[ind_pres[0][-1], :, :]-temp_prior[ind_pres[0][-1]])*100/temp_prior[ind_pres[0][-1]], norm=norm, cmap='seismic', 
                                                    origin='lower', extent=[longitude[0],longitude[-1],latitude[0],latitude[-1]])
                ax[irow[iax]][icol[iax]].tick_params(labelsize=14)
                ax[irow[iax]][icol[iax]].set_title(ititle[iax]+f"    {int(int(ptarget[ipressure]))} mbar", fontfamily='sans-serif', loc='left', fontsize=12)
                cbar = fig.colorbar(im, ax=ax[irow[iax]][icol[iax]], extend='both', fraction=0.04, pad=0.05)#, orientation='horizontal')
                cbar.ax.tick_params(labelsize=11)
                cbar.ax.locator_params(nbins=7)
                cbar.set_label("Temperature errror (%)", size=10)
                iax+=1 
            plt.axes([0.15, 0.1, 0.8, 0.8], frameon=False) 
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel("System III West Longitude", size=18)
            plt.ylabel("Planetographic Latitude", size=18)
            # Save figure showing calibation method 
            plt.savefig(f"{subdir}{itest}_all_filters_temperature_residual_maps.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{dir}{wave}_calibration_merid_profiles.eps", dpi=900)
            # Clear figure to avoid overlapping between plotting subroutines
            plt.close()

####### Windshear from retrieved temperature plotting #######
def PlotWindShearFromRetrievedTemperature(over_axis):

    print('Plotting windshear retrieved temperature over latitude...')
    #  Load Jupiter zonal jets data to determine belts and zones location
    ejets_c, wjets_c, nejet, nwjet = ReadZonalWind("../inputs/jupiter_jets.dat")
     # If subdirectory does not exist, create it
    dir = '../retrievals/retrieved_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Retrieval outputs directory path
    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/"
    # Array of prior file names
    prior = ['jupiter_v2023']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        retrieval_test = [
            f"{iprior}_temp_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H6_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_NH3_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H6_NH3_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H6pknee02mbar_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H6_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H6pknee02mbar_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H6pknee02mbar_NH3_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H6_NH3p_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_NH3p_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H6pknee02mbar_NH3p_ktable-highreso_no852_no887",
                    ]
        ntest = len(retrieval_test)
        # Loop over each prior used for retrievals tests
        for itest in retrieval_test:
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{iprior}_selected_meridien_tests/{itest}/windshear/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            # # Read profile data from NEMESIS prior file 
            # _, prior_p, prior_temperature, prior_error, _, _, nlevel, _ = ReadTemperatureGasesPriorProfile(f"{fpath}{itest}/core_1/")
            # Read retrieved profiles from .prf outputs files
            temperature, _, latitude, _, pressure, nlat, nlevel, _, _= ReadprfFiles(filepath=f"{fpath}{itest}", over_axis=over_axis)
            # Calculate windshear from retrieved temperature profile
            windshear, dTdy = CalculateWindShear(temperature, latitude, nlat, nlevel)
            
            # Determine extreme values of windshear and levels for mapping
            max = np.nanmax(windshear)
            min = np.nanmin(windshear)
            levels_cmap = np.logspace(-1, 1., num=61, endpoint=True)
            # Plotting retrieved windshear over latitude
            plt.figure(figsize=(10, 6))
            im = plt.pcolormesh(latitude, pressure, windshear, cmap='seismic', norm=SymLogNorm(linthresh=0.1, linscale=0.03,
                                              vmin=-3.0, vmax=3.0), shading='auto')
            for iejet in range(0,nejet):
                plt.plot([ejets_c[iejet],ejets_c[iejet]],[0.1, 10000],color='black',linestyle="dashed")
            for iwjet in range(0,nwjet):
                plt.plot([wjets_c[iwjet],wjets_c[iwjet]],[0.1, 10000],color='black',linestyle="dotted") 
            plt.ylim(0.1, 1000)
            # plt.xlim(-80, 80)
            plt.yscale('log')
            plt.gca().invert_yaxis()
            plt.tick_params(labelsize=15)       
            if over_axis=="longitude":
                plt.xlabel("System III West Longitude", size=20)
            elif over_axis=="latitude":
                plt.xlabel("Planetographic Latitude", size=20)
            plt.ylabel(f"Presssure [mbar]", size=20)
            cbar = plt.colorbar(im, extend='both', fraction=0.046, pad=0.05, orientation='vertical', format = LogFormatter())
            cbar.ax.tick_params(labelsize=15)
            # cbar.ax.locator_params(nbins=21)
            cbar.ax.set_title(r'[m s$^{-1}$ km$^{-1}$]',fontsize=15)
            # Save figure in the retrievals outputs directory
            plt.savefig(f"{subdir}retrieved_windshear.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{subdir}{itest}_retrieved_windshear_profile_at_{latitude[ilat]}.eps", dpi=100)
            # Close figure to avoid overlapping between plotting subroutines
            plt.close()

            # Plotting retrieved windshear over latitude with equatorial mask
            # Make the shaded region
            latmask = np.where((latitude >-6) & (latitude <6))

            ix = latitude[latmask]
            iy = pressure
            # xs, ys = zip(*(ix, iy))
                        
            plt.figure(figsize=(10, 6))
            # levels_cmap = np.linspace(-1, 1., num=61, endpoint=True)
            im = plt.pcolormesh(latitude, pressure, windshear, cmap='seismic', norm=SymLogNorm(linthresh=0.05, linscale=0.05,
                                              vmin=-4.0, vmax=4.0), shading='gouraud')
            for iejet in range(0,nejet):
                plt.plot([ejets_c[iejet],ejets_c[iejet]],[0.1, 10000],color='black',linestyle="dashed")
            for iwjet in range(0,nwjet):
                plt.plot([wjets_c[iwjet],wjets_c[iwjet]],[0.1, 10000],color='black',linestyle="dotted")
            # plt.plot(xs, ys) 
            plt.gca().add_patch(Rectangle((-6, pressure[-1]), len(ix), 100000, facecolor="grey"))
            plt.ylim(0.1, 1000)
            # plt.xlim(-80, 80)
            plt.yscale('log')
            plt.gca().invert_yaxis()
            # plt.gca().add_patch(Rectangle((ix[0], iy[0]), len(ix), len(iy), facecolor="grey"))
            plt.tick_params(labelsize=15)       
            if over_axis=="longitude":
                plt.xlabel("System III West Longitude", size=20)
            elif over_axis=="latitude":
                plt.xlabel("Planetographic Latitude", size=20)
            plt.ylabel(f"Presssure [mbar]", size=20)
            cbar = plt.colorbar(im, extend='both', fraction=0.046, pad=0.05, orientation='vertical', format="%.1f", ticks=[-4.0, -2.0, -1.0, -0.5, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 4.0])
            cbar.ax.tick_params(labelsize=15)
            # cbar.ax.locator_params(nbins=21)
            cbar.ax.set_title(r'[m s$^{-1}$ km$^{-1}$]',fontsize=15, pad=20)
            # Save figure in the retrievals outputs directory
            plt.savefig(f"{subdir}retrieved_windshear_mask.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{subdir}{itest}_retrieved_windshear_profile_at_{latitude[ilat]}.eps", dpi=100)
            # Close figure to avoid overlapping between plotting subroutines
            plt.close()

# Determine extreme values of dTdy and levels for mapping
            max = np.nanmax(dTdy)
            min = np.nanmin(dTdy)
            levels_cmap = np.logspace(-1, 1., num=61, endpoint=True)
            # Plotting retrieved dTdy over latitude
            plt.figure(figsize=(10, 6))
            im = plt.contourf(latitude, pressure, dTdy, cmap='seismic')
            for iejet in range(0,nejet):
                plt.plot([ejets_c[iejet],ejets_c[iejet]],[0.1, 10000],color='black',linestyle="dashed")
            for iwjet in range(0,nwjet):
                plt.plot([wjets_c[iwjet],wjets_c[iwjet]],[0.1, 10000],color='black',linestyle="dotted") 
            plt.ylim(0.1, 1000)
            # plt.xlim(-80, 80)
            plt.yscale('log')
            plt.gca().invert_yaxis()
            plt.tick_params(labelsize=15)       
            if over_axis=="longitude":
                plt.xlabel("System III West Longitude", size=20)
            elif over_axis=="latitude":
                plt.xlabel("Planetographic Latitude", size=20)
            plt.ylabel(f"Presssure [mbar]", size=20)
            cbar = plt.colorbar(im, extend='both', fraction=0.046, pad=0.05, orientation='vertical')
            cbar.ax.tick_params(labelsize=15)
            # cbar.ax.locator_params(nbins=21)
            cbar.ax.set_title(r'[K km$^{-1}$]',fontsize=15, pad=20)
            # Save figure in the retrievals outputs directory
            plt.savefig(f"{subdir}retrieved_dTdy.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{subdir}{itest}_retrieved_dTdy_profile_at_{latitude[ilat]}.eps", dpi=100)
            # Close figure to avoid overlapping between plotting subroutines
            plt.close()

            # Plotting retrieved dTdy over latitude with equatorial mask
            # Make the shaded region
            ptarget = 5.
            ind_pres = np.where(pressure >= ptarget)
            vortex_nord = 68
            vortex_sud = -64
            dtdymax = np.nanmax(dTdy[ind_pres[0][-1], :])
            dtdymin = np.nanmin(dTdy[ind_pres[0][-1], :])
            # xs, ys = zip(*(ix, iy))
                        
            plt.figure(figsize=(10, 6))
            # levels_cmap = np.linspace(-1, 1., num=61, endpoint=True)
            plt.plot(latitude, dTdy[ind_pres[0][-1], :], color='crimson', lw=2)
            for iejet in range(0,nejet):
                plt.plot([ejets_c[iejet],ejets_c[iejet]],[dtdymin, dtdymax],color='black',linestyle="dashed")
            for iwjet in range(0,nwjet):
                plt.plot([wjets_c[iwjet],wjets_c[iwjet]],[dtdymin, dtdymax],color='black',linestyle="dotted")
            plt.plot([vortex_nord, vortex_nord],[dtdymin, dtdymax])
            plt.plot([vortex_sud, vortex_sud],[dtdymin, dtdymax])
            # # plt.plot(xs, ys) 
            
            plt.tick_params(labelsize=15)       
            if over_axis=="longitude":
                plt.xlabel("System III West Longitude", size=20)
                
            elif over_axis=="latitude":
                plt.xlabel("Planetographic Latitude", size=20)
                plt.xlim(-90, 90)
            plt.ylabel(f"Thermal gradient "+r'[K km$^{-1}$]', size=20)
            plt.ylim(dtdymin, dtdymax)
            plt.grid()
            
            # Save figure in the retrievals outputs directory
            plt.savefig(f"{subdir}retrieved_dTdy_at_{ptarget}mbar.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{subdir}{itest}_retrieved_dTdy_profile_at_{latitude[ilat]}.eps", dpi=100)
            # Close figure to avoid overlapping between plotting subroutines
            plt.close()




####### Radiances plotting and mapping routines ####### 
def PlotRetrievedRadiance(over_axis):

    print('Plotting NEMESIS retrieved radiance for each latitude...')
    # If subdirectory does not exist, create it
    dir = '../retrievals/retrieved_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Retrieval outputs directory path
    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/"
    # Array of prior file names
    prior = ['jupiter_v2021']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        retrieval_test = [
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6pknee02mbar_NH3p_fshfix_no852_no887_aprmodif",
                        ]
        ntest = len(retrieval_test)
        # Loop over each retrieval tests for the current prior file
        for itest in retrieval_test:
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{itest}/radiances/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            # Read retrieved profiles from .mre outputs files
            radiance, wavenumb, rad_err, rad_fit, latitude, nlat = ReadmreFiles(filepath=f"{fpath}{itest}", over_axis=over_axis) 
            # Plotting retrieved radiance over wavenumber for each latitude
            for ilat in range(nlat):
                fig, axes = plt.subplots(1, 1, figsize=(10, 7), sharex=True, sharey=True)        
                axes.plot(wavenumb[:, ilat], radiance[:, ilat], lw=2, label=f"Obs Radiance at {latitude[ilat]}", color='orange')
                axes.fill_between(wavenumb[:, ilat], radiance[:, ilat]-rad_err[:, ilat], radiance[:, ilat]+rad_err[:, ilat], color='orange', alpha=0.2)
                axes.plot(wavenumb[:, ilat], rad_fit[:, ilat], lw=2, label=f"Retrieved Radiance")
                axes.grid()
                axes.legend(loc="upper right", fontsize=15)
                axes.tick_params(labelsize=15)
                # Add a big axis 
                plt.axes([0.1, 0.09, 0.8, 0.8], frameon=False) 
                plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)        
                plt.xlabel("Wavenumber [cm$^{-1}$]", size=20)
                plt.ylabel("Radiance [nW cm$^{-2}$ sr$^{-1}$ cm]", size=20)
                # Save figure in the retrievals outputs directory
                plt.savefig(f"{subdir}{itest}_retrieved_radiance_at_{latitude[ilat]}.png", dpi=150, bbox_inches='tight')
                #plt.savefig(f"{subdir}{itest}_retrieved_radiance_at_{latitude[ilat]}.eps", dpi=100)
                # Close figure to avoid overlapping between plotting subroutines
                plt.close()

def PlotRetrievedRadianceMeridian(over_axis):
    
    print('Plotting NEMESIS retrieved radiance meridians...')
    # If subdirectory does not exist, create it
    dir = '../retrievals/retrieved_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Retrieval outputs directory path
    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/"
    # Array of prior file names
    prior = ['jupiter_v2021']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        retrieval_test = [
            #             f"{iprior}_temp_no852_no887",
            # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H4-C2H6pknee02mbar_no852_no887",
            #  f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H6_no852_no887",
            # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H6_ktable-highreso_no852_no887",
            # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H6pknee02mbar_no852_no887",
            # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H6pknee02mbar_ktable-highreso_no852_no887",
            # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H6pknee02mbar_ktable-highreso_no852_no887",
            # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H6pknee02mbar_ktable-highreso_no852_no887",
            # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H4-C2H6pknee02mbar_ktable-highreso_no852_no887",
            # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H4-C2H6pknee02mbar_ktable-highreso_no852_no887",
            "jupiter_v2023_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H4-C2H6pknee02mbar_ktable-highreso_no852_no887",
            "jupiter_v2023_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H4-C2H6pknee02mbar_ktable-highreso_no852_no887",
            "jupiter_v2023_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H6pknee02mbar_ktable-highreso_no852_no887",
            "jupiter_v2023_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H6pknee02mbar_ktable-highreso_no852_no887"
                        
                        ]
        ntest = len(retrieval_test)
        # Loop over each retrieval tests for the current prior file
        for itest in retrieval_test:
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{iprior}_selected_meridien_tests/{itest}/merid_radiances/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            # Read retrieved profiles from .mre outputs files
            radiance, wavenumb, rad_err, rad_fit, _, _, _, _, \
                            _, _,  _, _, _, _,  _, _, \
                            _, _, _, _, _, _, _, _, _, \
                            latitude, nlat = ReadmreFiles(filepath=f"{fpath}{itest}", over_axis=over_axis, gas_name=['C2H6']) 
            # Plotting retrieved radiance over wavenumber for each wavenumber
            for ifilt in range(11):
                fig, axes = plt.subplots(1, 1, figsize=(8, 3), sharex=True, sharey=True)        
                axes.plot(latitude, radiance[ifilt, :], lw=2, label=f"Obs Radiance at {int(wavenumb[ifilt, 1])} "+r'cm$^{-1}$', color='black')
                axes.fill_between(latitude, radiance[ifilt, :]-rad_err[ifilt, :], radiance[ifilt, :]+rad_err[ifilt, :], color='black', alpha=0.2)
                axes.plot(latitude, rad_fit[ifilt, :], lw=2, label=f"Retrieved Radiance", color='red')
                axes.grid()
                axes.legend(fontsize=12)
                axes.tick_params(labelsize=12)
                # Add a big axis 
                plt.axes([0.1, 0.09, 0.8, 0.8], frameon=False) 
                plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)        
                if over_axis=="longitude":
                    plt.xlabel("System III West Longitude", size=15)
                elif over_axis=="latitude":
                    plt.xlabel("Planetographic Latitude", size=15)
                plt.ylabel("Radiance [nW cm$^{-2}$ sr$^{-1}$ cm]", size=15)
                # Save figure in the retrievals outputs directory
                plt.savefig(f"{subdir}{itest}_retrieved_merid_radiance_at_{wavenumb[ifilt, 1]}.png", dpi=150, bbox_inches='tight')
                #plt.savefig(f"{subdir}{itest}_retrieved_radiance_at_{latitude[ilat]}.eps", dpi=100)
                # Close figure to avoid overlapping between plotting subroutines
                plt.close()

                fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True) 
                axes[0].plot(latitude, radiance[ifilt, :], lw=2, label=f"Obs Radiance at {int(wavenumb[ifilt, 1])} "+r'cm$^{-1}$', color='black')
                axes[0].fill_between(latitude, radiance[ifilt, :]-rad_err[ifilt, :], radiance[ifilt, :]+rad_err[ifilt, :], color='black', alpha=0.2)
                axes[0].plot(latitude, rad_fit[ifilt, :], lw=2, label=f"Retrieved Radiance", color='red')
                axes[0].grid()
                axes[0].set_xlim((-90, 90))
                axes[0].set_xticks(np.arange(-90, 91, 30))
                axes[0].legend(fontsize=15)
                axes[0].tick_params(labelsize=15)
                axes[0].set_ylabel("Radiance [nW cm$^{-1}$ sr$^{-1}$]", size=20)

                axes[1].plot(latitude, (rad_fit[ifilt, :]-radiance[ifilt, :])*100/radiance[ifilt, :], lw=2, color='black')
                axes[1].plot((-90, 90), (0, 0), lw=1, ls=':', color='k')
                axes[1].plot((-90, 90), (-5, -5), lw=1, color='k')
                axes[1].plot((-90, 90), (5, 5), lw=1, color='k')
                axes[1].set_ylim(-10,10)
                axes[1].set_xlim((-90, 90))
                axes[1].set_xticks(np.arange(-90, 91, 30))
                axes[1].grid()
                axes[1].tick_params(labelsize=15)
                axes[1].set_ylabel(r"$\Delta R$ (%)", size=20)
                # Add a big axis 
                plt.axes([0.1, 0.09, 0.8, 0.8], frameon=False) 
                plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)        
                if over_axis=="longitude":
                    plt.xlabel("System III West Longitude", size=20)
                elif over_axis=="latitude":
                    plt.xlabel("Planetographic Latitude", size=20)
                
                # Save figure in the retrievals outputs directory
                plt.savefig(f"{subdir}{itest}_residual_merid_radiance_at_{wavenumb[ifilt, 1]}.png", dpi=150, bbox_inches='tight')
                #plt.savefig(f"{subdir}{itest}_retrieved_radiance_at_{latitude[ilat]}.eps", dpi=100)
                # Close figure to avoid overlapping between plotting subroutines
                plt.close()

def PlotRetrievedRadianceMeridianSuperpose(over_axis):
    test_names = [
            r'Temp',
            r'Temp Aer C$_2$H$_{2, p}$ C$_2$H$_4$ C$_2$H$_{6, p}$', 
            r'Temp Aer C$_2$H$_6$',
            r'Temp Aer C$_2$H$_6$, ktables HighReso',
            r'Temp Aer C$_2$H$_{6,p}$'

            ] 
    print('Plotting NEMESIS retrieved radiance meridians (superpose figure of several tests)...')
    # If subdirectory does not exist, create it
    dir = '../retrievals/retrieved_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Retrieval outputs directory path
    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/"
    # Array of prior file names
    prior = ['jupiter_v2021']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        retrieval_test = [
                        f"{iprior}_temp_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H4-C2H6pknee02mbar_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H6_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H6_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H6pknee02mbar_no852_no887"

                        ]
        ntest = len(retrieval_test)
        radiance = np.empty((ntest, Globals.nfilters, 176))
        rad_fit = np.empty((ntest, Globals.nfilters, 176))
        rad_err = np.empty((ntest, Globals.nfilters, 176))
        # Loop over each retrieval tests for the current prior file
        iretrieve = 0
        for itest in retrieval_test:
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{iprior}_selected_meridien_tests/{itest}/merid_radiances/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            # Read retrieved profiles from .mre outputs files
            radiance[iretrieve, :, :], wavenumb, rad_err[iretrieve, :, :], rad_fit[iretrieve, :, :], _, _, _, _, \
                            _, _,  _, _, _, _,  _, _, \
                            _, _, _, _, _, _, _, _, _, \
                            latitude, nlat = ReadmreFiles(filepath=f"{fpath}{itest}", over_axis=over_axis, gas_name=['C2H6'])
            iretrieve +=1
        
        for ifilt in range(Globals.nfilters):
            iretrieve = 0
            # Plotting retrieved radiance over wavenumber for each wavenumber
            fig = plt.subplots(1, 1, figsize=(8, 3))
            plt.plot(latitude, radiance[iretrieve, ifilt, :], lw=2, label=f"Obs Radiance at {int(wavenumb[ifilt, 1])}", color='blue')
            plt.fill_between(latitude, radiance[iretrieve, ifilt, :]-rad_err[iretrieve, ifilt, :], radiance[iretrieve, ifilt, :]+rad_err[iretrieve, ifilt, :], color='blue', alpha=0.2) 
            for i, itest in enumerate(retrieval_test):
                col = cmap(i/ntest)
                plt.plot(latitude, rad_fit[iretrieve, ifilt, :], lw=2, label=test_names[i], color=col)
                # plt.plot(latitude, rad_fit[iretrieve, ifilt, :], lw=2, label=f"{itest}"[14:32]+"C2H2_C2H6_NH3")
                # plt.plot(latitude, rad_fit[iretrieve, ifilt, :], lw=2, label=f"{itest}"[14:32])
                # plt.plot(latitude, rad_fit[iretrieve, ifilt, :], lw=2, label=f"{itest}"[14:])
                iretrieve +=1
            plt.grid()
            plt.legend(fontsize=6)
            plt.tick_params(labelsize=12)        
            if over_axis=="longitude":
                plt.xlabel("System III West Longitude", size=15)
            elif over_axis=="latitude":
                plt.xlabel("Planetographic Latitude", size=15)
            plt.ylabel("Radiance [nW cm$^{-1}$ sr$^{-1}$]", size=15)
            # Save figure in the retrievals outputs directory
            plt.savefig(f"{subdir}comparison_merid_radiance_at_{wavenumb[ifilt, 1]}.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{subdir}comparison_merid_radiance_at_{wavenumb[ifilt, 1]}.eps", dpi=100)
            # Close figure to avoid overlapping between plotting subroutines
            plt.close()

def PlotRetrievedRadianceMap(over_axis="2D"):

    print('Plotting observed, NEMESIS retrieved radiance and residual error maps...')
    # If subdirectory does not exist, create it
    dir = '../retrievals/retrieved_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/retrieval_GreatRedSpot/"
    # Array of prior file names
    prior = ['jupiter_v2021']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        retrieval_test = [
                        # f"jupiter_v2021_temp_aerosol1-10mu-800mbar-05scale-01_GRS_no852_no887",
                        # f"jupiter_v2021_temp_aerosol1-10mu-800mbar-05scale-01_NH3_GRS_no852_no887",
                        "jupiter_v2021_temp_aerosol1-10mu-800mbar-05scale-01_NH3_GRS_halfdeg_no852_no887"
                        ]
        ntest = len(retrieval_test)
        # Loop over each prior used for retrievals tests
        for itest in retrieval_test:
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{iprior}_selected_meridien_tests/{itest}/GRS_temperature_map/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            print(f" ... {itest} ...")
            # Read retrieved profiles from .mre outputs files
            radiance, wavenumb, rad_err, rad_fit, _, _, _, _, \
                            _, _,  _, _, _, _,  _, _, \
                            _, _, _, _, _, _, _, _, _, \
                            latitude, nlat, longitude, nlon = ReadmreFiles(filepath=f"{fpath}{itest}", over_axis=over_axis, gas_name=['NH3'])
            # Create a subplot figure of radiance with all filters
            print("      ... observed radiance map")
            fig, ax = plt.subplots(6, 2, figsize=(10, 12), sharex=True, sharey=True)
            iax = 0
            for ifilt in [0,8,9,10,5,4,6,7,3,2,1]:
                irow = [0,1,1,2,2,3,3,4,4,5,5]
                icol = [0,0,1,0,1,0,1,0,1,0,1]
                ititle = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)']
                # Remove the frame of the empty subplot
                ax[0][1].set_frame_on(False)
                ax[0][1].tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                # Get filter index for plotting spacecraft and calibrated data
                if ifilt > 5:
                    _, wavl, wave, _, _ = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=ifilt+2)
                else: 
                    _, wavl, wave, _, _ = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=ifilt)
                # Set extreme values
                radmax = np.nanmax(radiance[ifilt, :, :]) 
                radmin = np.nanmin(radiance[ifilt, :, :])
                # subplot showing the regional radiance maps
                im = ax[irow[iax]][icol[iax]].imshow(radiance[ifilt, :, :], vmin=radmin, vmax=radmax, cmap='inferno', 
                                                    origin='lower', extent=[longitude[0],longitude[-1],latitude[0],latitude[-1]])
                ax[irow[iax]][icol[iax]].tick_params(labelsize=14)
                ax[irow[iax]][icol[iax]].set_title(ititle[iax]+f"    {(wavl)}"+" $\mu$m", fontfamily='sans-serif', loc='left', fontsize=12)
                cbar = fig.colorbar(im, ax=ax[irow[iax]][icol[iax]], extend='both', fraction=0.04, pad=0.05)#, orientation='horizontal')
                cbar.ax.tick_params(labelsize=12)
                cbar.ax.locator_params(nbins=6)
                cbar.ax.set_title("[nW cm$^{-1}$ sr$^{-1}$]", size=12, pad=8)
                iax+=1 
            plt.axes([0.15, 0.1, 0.8, 0.8], frameon=False) 
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel("System III West Longitude", size=18)
            plt.ylabel("Planetographic Latitude", size=18)
            # Save figure showing calibation method 
            plt.savefig(f"{subdir}{itest}_all_filters_radiance_maps.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{dir}{wave}_calibration_merid_profiles.eps", dpi=900)
            # Clear figure to avoid overlapping between plotting subroutines
            plt.close()

            # Create a subplot figure of radiance fit with all filters
            print("      ... retrieved radiance map")
            fig, ax = plt.subplots(6, 2, figsize=(10, 12), sharex=True, sharey=True)
            iax = 0
            for ifilt in [0,8,9,10,5,4,6,7,3,2,1]:
                irow = [0,1,1,2,2,3,3,4,4,5,5]
                icol = [0,0,1,0,1,0,1,0,1,0,1]
                ititle = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)']
                # Remove the frame of the empty subplot
                ax[0][1].set_frame_on(False)
                ax[0][1].tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                # Get filter index for plotting spacecraft and calibrated data
                if ifilt > 5:
                    _, wavl, wave, _, _ = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=ifilt+2)
                else: 
                    _, wavl, wave, _, _ = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=ifilt)
                # Set extreme values
                radmax = np.nanmax(rad_fit[ifilt, :, :]) 
                radmin = np.nanmin(rad_fit[ifilt, :, :])
                # subplot showing the regional rad_fit maps
                im = ax[irow[iax]][icol[iax]].imshow(rad_fit[ifilt, :, :], vmin=radmin, vmax=radmax, cmap='inferno', 
                                                    origin='lower', extent=[longitude[0],longitude[-1],latitude[0],latitude[-1]])
                ax[irow[iax]][icol[iax]].tick_params(labelsize=14)
                ax[irow[iax]][icol[iax]].set_title(ititle[iax]+f"    {(wavl)}"+" $\mu$m", fontfamily='sans-serif', loc='left', fontsize=12)
                cbar = fig.colorbar(im, ax=ax[irow[iax]][icol[iax]], extend='both', fraction=0.04, pad=0.05)#, orientation='horizontal')
                cbar.ax.tick_params(labelsize=12)
                cbar.ax.locator_params(nbins=6)
                cbar.ax.set_title("[nW cm$^{-1}$ sr$^{-1}$]", size=12, pad=8)
                iax+=1 
            plt.axes([0.15, 0.1, 0.8, 0.8], frameon=False) 
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel("System III West Longitude", size=18)
            plt.ylabel("Planetographic Latitude", size=18)
            # Save figure showing calibation method 
            plt.savefig(f"{subdir}{itest}_all_filters_rad_fit_maps.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{dir}{wave}_calibration_merid_profiles.eps", dpi=900)
            # Clear figure to avoid overlapping between plotting subroutines
            plt.close()

            # Create a subplot figure of radiance residual with all filters
            print("      ... residual error of radiance (%) map")
            fig, ax = plt.subplots(6, 2, figsize=(10, 12), sharex=True, sharey=True)
            iax = 0
            for ifilt in [0,8,9,10,5,4,6,7,3,2,1]:
                irow = [0,1,1,2,2,3,3,4,4,5,5]
                icol = [0,0,1,0,1,0,1,0,1,0,1]
                ititle = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)']
                # Remove the frame of the empty subplot
                ax[0][1].set_frame_on(False)
                ax[0][1].tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                # Get filter index for plotting spacecraft and calibrated data
                if ifilt > 5:
                    _, wavl, wave, _, _ = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=ifilt+2)
                else: 
                    _, wavl, wave, _, _ = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=ifilt)
                # Set extreme values rad_fit[ifilt, :]-radiance[ifilt, :])*100/radiance[ifilt, :]
                radmax = np.nanmax((rad_fit[ifilt, :, :]-radiance[ifilt, :, :])*100/radiance[ifilt, :, :]) 
                radmin = np.nanmin((rad_fit[ifilt, :, :]-radiance[ifilt, :, :])*100/radiance[ifilt, :, :])
                if radmax > 0 and radmin < 0:
                    norm = colors.TwoSlopeNorm(vmin=radmin, vmax=radmax, vcenter=0)
                elif radmax < 0 and radmin < 0:
                    norm = colors.TwoSlopeNorm(vmin=radmin, vmax=0.5, vcenter=0)
                elif radmax > 0 and radmin > 0:
                    norm = colors.TwoSlopeNorm(vmin=-0.5, vmax=radmax, vcenter=0)
                # subplot showing the regional rad_fit maps
                im = ax[irow[iax]][icol[iax]].imshow((rad_fit[ifilt, :, :]-radiance[ifilt, :, :])*100/radiance[ifilt, :, :], norm=norm, cmap='seismic', 
                                                    origin='lower', extent=[longitude[0],longitude[-1],latitude[0],latitude[-1]])
                ax[irow[iax]][icol[iax]].tick_params(labelsize=14)
                ax[irow[iax]][icol[iax]].set_title(ititle[iax]+f"    {(wavl)}"+" $\mu$m", fontfamily='sans-serif', loc='left', fontsize=12)
                cbar = fig.colorbar(im, ax=ax[irow[iax]][icol[iax]], extend='both', fraction=0.04, pad=0.05)#, orientation='horizontal')
                cbar.ax.tick_params(labelsize=11)
                cbar.ax.locator_params(nbins=7)
                cbar.ax.set_title("Radiance errror (%)", size=10, pad=10)
                iax+=1 
            plt.axes([0.15, 0.1, 0.8, 0.8], frameon=False) 
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel("System III West Longitude", size=18)
            plt.ylabel("Planetographic Latitude", size=18)
            # Save figure showing calibation method 
            plt.savefig(f"{subdir}{itest}_all_filters_rad_residual_maps.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{dir}{wave}_calibration_merid_profiles.eps", dpi=900)
            # Clear figure to avoid overlapping between plotting subroutines
            plt.close()








####### Aerosol plotting and mapping routines ####### 
def PlotRetrievedAerosolProfile(over_axis):

    print('Plotting NEMESIS retrieved aerosol profiles...')
    # If subdirectory does not exist, create it
    dir = '../retrievals/retrieved_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Retrieval outputs directory path
    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/"
    # Array of prior file names
    prior = ['jupiter_v2021']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        retrieval_test = [
                        f"{iprior}_temp_aerosol1-1mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6pknee02mbar_no852_no887",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6pknee02mbar_no852_no887"
                        ]
        ntest = len(retrieval_test)
        # Loop over each retrieval tests for the current prior file
        for itest in retrieval_test:
            # Read profile data from NEMESIS prior file 
            prior_aer, prior_alt, prior_ncloud, prior_nlevel = ReadAerosolPriorProfile(f"{fpath}{itest}/core_1/aerosol.ref")
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{itest}/profiles/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            # Read retrieved profiles from aerosol.prf outputs files
            aerosol, altitude, latitude, nlevel, nlat = ReadaerFiles(filepath=f"{fpath}{itest}", over_axis=over_axis)
            # Plotting retrieved aerosol profile for each latitude
            for ilat in range(nlat):
                fig, axes = plt.subplots(1, prior_ncloud, figsize=(7, 10), sharey=True)
                if prior_ncloud > 1:
                    for icloud in range(prior_ncloud):
                        axes[icloud].plot(prior_aer[:, icloud], prior_alt, lw=2, label=f"aerosol apriori", color='orange')
                        #axes.fill_betweenx(prior_p, prior_temperature-prior_err, prior_temperature+prior_err, color='orange', alpha=0.2)
                        axes[icloud].plot(aerosol[:, ilat], altitude[:, ilat], lw=2, label=f"Retrieved aerosol at {latitude[ilat]}")
                        axes[icloud].grid()
                        axes[icloud].legend(loc="upper right", fontsize=15)
                        axes[icloud].tick_params(labelsize=15)
                else:
                    axes.plot(prior_aer, prior_alt, lw=2, label=f"aerosol apriori", color='orange')
                    #axes.fill_betweenx(prior_p, prior_temperature-prior_err, prior_temperature+prior_err, color='orange', alpha=0.2)
                    axes.plot(aerosol[:, ilat], altitude[:, ilat], lw=2, label=f"Retrieved aerosol at {latitude[ilat]}")
                    axes.grid()
                    axes.legend(loc="upper right", fontsize=15)
                    axes.tick_params(labelsize=15)
                # Add a big axis 
                plt.axes([0.1, 0.09, 0.8, 0.8], frameon=False) 
                plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)        
                plt.xlabel(f"Aerosol []", size=20)
                plt.ylabel(f"Height [km]", size=20)
                # Save figure in the retrievals outputs directory
                plt.savefig(f"{subdir}{itest}_retrieved_aerosol_profile_at_{latitude[ilat]}.png", dpi=150, bbox_inches='tight')
                #plt.savefig(f"{subdir}{itest}_retrieved_aerosol_profile_at_{latitude[ilat]}.eps", dpi=100)
                # Close figure to avoid overlapping between plotting subroutines
                plt.close()

def PlotRetrievedAerosolCrossSection(over_axis):

    print('Plotting NEMESIS retrieved aerosol profiles...')
    #  Load Jupiter zonal jets data to determine belts and zones location
    ejets_c, wjets_c, nejet, nwjet = ReadZonalWind("../inputs/jupiter_jets.dat")
    # If subdirectory does not exist, create it
    dir = '../retrievals/retrieved_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Retrieval outputs directory path
    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/"
    # Array of prior file names
    prior = ['jupiter_v2023']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        retrieval_test = [
            # f"{iprior}_temp_ktable-highreso_no852_no887",
            # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_ktable-highreso_no852_no887",
            # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H6_ktable-highreso_no852_no887",
            # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_NH3_ktable-highreso_no852_no887",
            # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H6_NH3_ktable-highreso_no852_no887",
            # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H6pknee02mbar_ktable-highreso_no852_no887",
            # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H6_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H6pknee02mbar_ktable-highreso_no852_no887",
            # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H6pknee02mbar_NH3_ktable-highreso_no852_no887",
            # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H6_NH3p_ktable-highreso_no852_no887",
            # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_NH3p_ktable-highreso_no852_no887",
            # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H6pknee02mbar_NH3p_ktable-highreso_no852_no887",
                    ]
        ntest = len(retrieval_test)
        for itest in retrieval_test:
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{iprior}_selected_meridien_tests/{itest}/aerosol_section/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            # Read retrieved profiles from aerosol.prf outputs files
            aerosol, altitude, latitude, nlevel, nlat = ReadaerFiles(filepath=f"{fpath}{itest}", over_axis=over_axis)
            # Read retrieved profiles from .prf outputs files
            _, _, _, _, pressure, _, _, _, _ = ReadprfFiles(filepath=f"{fpath}{itest}", over_axis=over_axis)
            # Determine extreme values of aerosol and levels for mapping
            max = np.nanmax(aerosol)
            min = np.nanmin(aerosol)
            levels_cmap = np.linspace(min, max, num=15, endpoint=True)
            # Plotting retrieved aerosol profile for each latitude
            fig, axes = plt.subplots(1, 1, figsize=(8, 6), sharey=True)
            
            axes = plt.contourf(latitude, pressure, aerosol, cmap='cividis', levels=levels_cmap)
            for iejet in range(0,nejet):
                plt.plot([ejets_c[iejet],ejets_c[iejet]],[100, 1000],color='white',linestyle="dashed")
            for iwjet in range(0,nwjet):
                plt.plot([wjets_c[iwjet],wjets_c[iwjet]],[100, 1000],color='white',linestyle="dotted") 
            plt.ylim(100, 1000)
            # plt.xlim(-80, 80)
            plt.yscale('log')
            plt.gca().invert_yaxis()
            plt.tick_params(labelsize=15)
            # axes.legend(loc="upper right", fontsize=15)
            # axes.tick_params(labelsize=15)
            # Add a big axis 
            plt.axes([0.1, 0.09, 0.8, 0.8], frameon=False) 
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)        
            if over_axis=="longitude":
                plt.xlabel("System III West Longitude", size=20)
            elif over_axis=="latitude":
                plt.xlabel("Planetographic Latitude", size=20)
            plt.ylabel(f"Pressure [mbar]", size=20)
            # Save figure in the retrievals outputs directory
            plt.savefig(f"{subdir}{itest}_retrieved_aerosol.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{subdir}{itest}_retrieved_aerosol_profile_at_{latitude[ilat]}.eps", dpi=100)
            # Close figure to avoid overlapping between plotting subroutines
            plt.close()

def PlotRetrievedAerosolMaps(over_axis="2D"):

    print('Plotting NEMESIS retrieved aerosol maps...')
     # If subdirectory does not exist, create it
    dir = '../retrievals/retrieved_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Retrieval outputs directory path
    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/retrieval_GreatRedSpot/"
    # Array of prior file names
    prior = ['jupiter_v2021']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        retrieval_test = [
                        # f"jupiter_v2021_temp_aerosol1-10mu-800mbar-05scale-01_GRS_no852_no887",
                        # f"jupiter_v2021_temp_aerosol1-10mu-800mbar-05scale-01_NH3_GRS_no852_no887",
                        "jupiter_v2021_temp_aerosol1-10mu-800mbar-05scale-01_NH3_GRS_halfdeg_no852_no887"
                        ]
        ntest = len(retrieval_test)
        # Loop over each prior used for retrievals tests
        for itest in retrieval_test:
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{iprior}_selected_meridien_tests/{itest}/GRS_aerosol_map/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            # Read retrieved profiles from .prf outputs files
            aerosol, height, nlevel, latitude, nlat, longitude, nlon, ncoor = ReadaerFiles(filepath=f"{fpath}{itest}", over_axis=over_axis)
            aer_mre, aer_err, aer_fit, fit_err, lat_mre, nlat, long_mre, nlon = ReadAerFromMreFiles(filepath=f"{fpath}{itest}", over_axis=over_axis)

            
            # Determine extreme values of temperature and levels for mapping
            aermax = np.nanmax(aerosol[:, :, :])
            aermin = np.nanmin(aerosol[:, :, :])
            levels_cmap = np.linspace(aermin, aermax, num=20, endpoint=True)

            # # Mapping the meridian aerosol cross-section
            # lon_index = (longitude == 157.5)
            # gaskeep = aerosol[:, :, lon_index]
            # plt.figure(figsize=(8, 6))
            # im = plt.contourf(latitude, height[:, 0, 0], gaskeep[:, :, 0], cmap='viridis', levels=levels_cmap)
            # # plt.contour(latitude, height, gaskeep[:, :, 0], levels=levels, colors="white")
            # # plt.ylim(0.001, 1000)
            # # plt.xlim(-80, 80)
            # # plt.yscale('log')
            # # plt.gca().invert_yaxis()
            # plt.tick_params(labelsize=15)        
            # plt.xlabel("Planetographic Latitude", size=20)
            # plt.ylabel(f"Height [km]", size=20)
            # plt.title(f"Great Red Spot structure at {float(longitude[lon_index])}"+"$^{\circ}$W")
            # cbar = plt.colorbar(im, extend='both', fraction=0.046, pad=0.05, orientation='vertical')
            # cbar.ax.tick_params(labelsize=15)
            # # Save figure in the retrievals outputs directory
            # plt.savefig(f"{subdir}{itest}_aerosol_meridian_cross_section_at_lon{float(longitude[lon_index])}.png", dpi=150, bbox_inches='tight')
            # #plt.savefig(f"{subdir}{itest}_meridian_cross_section_at_lon{float(longitude[lon_index])}.eps", dpi=100)
            # # Close figure to avoid overlapping between plotting subroutines
            # plt.close()

            # # Mapping the zonal aerosol cross-section
            # lat_index = (latitude == -20.5)
            # gaskeep = aerosol[:, lat_index, :]
            # plt.figure(figsize=(8, 6))
            # im = plt.contourf(longitude, height[:, 0, 0], gaskeep[:, 0, :], cmap='viridis', levels=levels_cmap)
            # # plt.contour(longitude, height, gaskeep[:, :, 0], levels=levels, colors="white")
            # # plt.ylim(0.001, 1000)
            # # plt.xlim(-80, 80)
            # # plt.yscale('log')
            # # plt.gca().invert_yaxis()
            # plt.gca().invert_xaxis()
            # plt.tick_params(labelsize=15)        
            # plt.xlabel("System III West longitude", size=20)
            # plt.ylabel(f"Height [km]", size=20)
            # plt.title(f"Great Red Spot structure at {float(latitude[lat_index])}"+"$^{\circ}$")
            # cbar = plt.colorbar(im, extend='both', fraction=0.046, pad=0.05, orientation='vertical')
            # cbar.ax.tick_params(labelsize=15)
            # # Save figure in the retrievals outputs directory
            # plt.savefig(f"{subdir}{itest}_aerosol_zonal_cross_section_at_lat{float(latitude[lat_index])}.png", dpi=150, bbox_inches='tight')
            # #plt.savefig(f"{subdir}{itest}_aerosol_zonal_cross_section_at_lat{float(latitude[lat_index])}.eps", dpi=100)
            # # Close figure to avoid overlapping between plotting subroutines
            # plt.close()

            # # Mapping the vertical aerosol cross-section
            # find pressure index 
            aerosol_max = (aerosol == np.nanmax(aerosol))
            height_index = np.argmax(aerosol, axis=0)
            lon_max_index = np.argmax(aerosol, axis=2)
            lat_max_index = np.argmax(aerosol, axis=1)
            aerkeep = aerosol[aerosol_max[:, 0, 0], :, :]
            plt.figure(figsize=(8, 4))
            im = plt.contourf(longitude, latitude, aerosol[height_index[0][0], :, :], cmap='viridis', levels=levels_cmap)
            # plt.contour(longitude, height, aerkeep[:, :, 0], levels=levels, colors="white")
            # plt.ylim(0.001, 1000)
            # plt.xlim(-80, 80)
            # plt.yscale('log')
            # plt.gca().invert_yaxis()
            plt.gca().invert_xaxis()
            plt.tick_params(labelsize=15)        
            plt.xlabel("System III West longitude", size=20)
            plt.ylabel(f"Planetographic Latitude", size=20)
            plt.title(f"Great Red Spot structure at {int(height[height_index[0][0], lat_max_index[0][0], lon_max_index[0][0]])} km")
            cbar = plt.colorbar(im, extend='both', fraction=0.025, pad=0.05, orientation='vertical')
            cbar.ax.tick_params(labelsize=15)
            # Save figure in the retrievals outputs directory
            plt.savefig(f"{subdir}{itest}_aerosol_vertical_cross_section_at_alt{float(height[height_index[0][0], lat_max_index[0][0], lon_max_index[0][0]])}.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{subdir}{itest}_aerosol_vertical_cross_section_at_alt{float(height[height_index[0][0]])lat_max_index[}][0].elon_max_index[ps[0]", dpi=100)
            # Close figure to avoid overlapping between plotting subroutines
            plt.close()

            # # Mapping the aerosol scale factor from .mre file
            plt.figure(figsize=(8, 4))
            #set extreme values
            aer_fitmax = np.nanmax(aer_fit[:, :])
            aer_fitmin = np.nanmin(aer_fit[:, :])
            levels_mre = np.linspace(aer_fitmin, aer_fitmax, num=20, endpoint=True)
            im = plt.imshow(aer_fit[:, ::-1], cmap='bone', vmin=aer_fitmin, vmax=aer_fitmax, # levels=levels_cmap,
                                                             origin='lower', extent=[long_mre[-1],long_mre[0],lat_mre[0],lat_mre[-1]])
            plt.gca().invert_xaxis()
            plt.tick_params(labelsize=15)        
            plt.xlabel("System III West longitude", size=15)
            plt.ylabel(f"Planetographic Latitude", size=15)
            cbar = plt.colorbar(im, extend='both', fraction=0.025, pad=0.05, orientation='vertical')
            cbar.ax.tick_params(labelsize=15)
            cbar.ax.set_title(f"Aerosol scale factor", fontsize=12, pad=15)
            # Save figure in the retrievals outputs directory
            plt.savefig(f"{subdir}{itest}_aerosol_scale_factor_from-mre.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{subdir}{itest}_aerosol_scale_factor_from-mre.eps", dpi=100)
            # Close figure to avoid overlapping between plotting subroutines
            plt.close()

def PlotRetrievedAerosolsMeridianProfiles(over_axis):

    print("Plot meridian profiles of aerosol ... ")
    #  Load Jupiter zonal jets data to determine belts and zones location
    ejets_c, wjets_c, nejet, nwjet = ReadZonalWind("../inputs/jupiter_jets.dat")
    # If subdirectory does not exist, create it
    dir = '../retrievals/retrieved_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Retrieval outputs directory path
    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/"
    # Array of prior file names
    prior = ['jupiter_v2023']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        retrieval_test = [
            # f"{iprior}_temp_ktable-highreso_no852_no887",
            # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_ktable-highreso_no852_no887",
            # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H6_ktable-highreso_no852_no887",
            # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_NH3_ktable-highreso_no852_no887",
            # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H6_NH3_ktable-highreso_no852_no887",
            # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H6pknee02mbar_ktable-highreso_no852_no887",
            # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H6_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H6pknee02mbar_ktable-highreso_no852_no887",
            # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H6pknee02mbar_NH3_ktable-highreso_no852_no887",
            # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H6_NH3p_ktable-highreso_no852_no887",
            # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_NH3p_ktable-highreso_no852_no887",
            # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H6pknee02mbar_NH3p_ktable-highreso_no852_no887",
                    ]
        ntest = len(retrieval_test)
        # Loop over each retrieval tests for the current prior file
        for i, itest in enumerate(retrieval_test):
            col = cmap(i/ntest)
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{iprior}_selected_meridien_tests/{itest}/aerosol_merid/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            # Read ChiSquare values from log files
            # chisquare, latitude, nlat = ReadLogFiles(filepath=f"{fpath}{itest}", over_axis="latitude")
            # Read retrieved profiles from .prf outputs files
            temperature, gases, latitude, _, pressure, _, _, ngas, gases_id = ReadprfFiles(filepath=f"{fpath}{itest}", over_axis="latitude")
            
            radiance, wavenumb, rad_err, rad_fit, temp_prior_mre, \
                                                    temp_errprior_mre, temp_fit_mre, temp_errfit_mre, \
                                                    aer_mre, aer_err,  aer_fit, fit_err, \
                                                    gas_scale, gas_scaleerr,  gas_scalefit, gas_errscalefit, \
                                                    gas_vmr, gas_vmrerr, gas_vmrfit, gas_errvmrfit,\
                                                    gas_fsh, gas_fsherr, gas_fshfit, gas_errfshfit, \
                                                    gas_abunerr, \
                                                    _, _  = ReadmreFiles(filepath=f"{fpath}{itest}", over_axis="latitude", gas_name=['C2H2', 'C2H4', 'C2H6'])
            fig = plt.figure(figsize=(10, 4))
            # Current work 
            plt.errorbar(latitude, aer_fit, yerr=fit_err, color='grey', lw=1, marker='.', markersize=7, 
                        markerfacecolor='black', markeredgewidth=0, alpha=0.7, capthick=0.4)
            for iejet in range(0,nejet):
                plt.plot([ejets_c[iejet],ejets_c[iejet]],[0, 2],color='crimson',linestyle="dashed")
            for iwjet in range(0,nwjet):
                plt.plot([wjets_c[iwjet],wjets_c[iwjet]],[0, 2],color='crimson',linestyle="dotted")
            plt.tick_params(labelsize=18)            
            plt.grid()
            plt.xlabel(f"Planetographic Latitude", size=20)                
            plt.xlim(-90, 90)
            plt.ylim(0, 2)
            plt.title(f"Cumulative Aerosol opacity at 1 bar", size=20)
            # plt.gca().set_yscale("log")
            # plt.legend(bbox_to_anchor=(0,1.1,1,0.2), loc="lower left", mode="expand", fontsize=12)
            plt.savefig(f"{subdir}cumul_aerosol_opacity_1bar.png", dpi=150, bbox_inches='tight')
            plt.close()






####### Gases plotting and mapping routines ####### 
def PlotRetrievedGasesProfile(over_axis):

    print('Plotting NEMESIS retrieved gases profiles ...')
    # If subdirectory does not exist, create it
    dir = '../retrievals/retrieved_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Retrieval outputs directory path
    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/"
    # Array of prior file names
    prior = ['jupiter_v2021']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        retrieval_test = [
                        f"{iprior}_temp_aerosol1-1mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6pknee02mbar_no852_no887",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6pknee02mbar_no852_no887"
                        ]
        ntest = len(retrieval_test)
        # Loop over each retrieval tests for the current prior file
        for itest in retrieval_test:
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{itest}/profiles/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            # Read profile data from NEMESIS prior file 
            _, prior_p, _, _, prior_gases, _, _, _ = ReadTemperatureGasesPriorProfile(f"{fpath}{itest}/core_1/")
            # Read retrieved profiles from .prf outputs files
            _, gases, latitude, _, pressure, nlat, _, ngas, gases_id = ReadprfFiles(filepath=f"{fpath}{itest}", over_axis=over_axis)
            
            for ilat in range(nlat):
                # Mapping the gases cross-section with zind location
                for igas in range(ngas):               
                    # Plot cross-section figure of gases abundances
                    plt.figure(figsize=(7, 10), dpi=100)
                    gases_name, _ = RetrieveGasesNames(gas_name=False, gas_id=gases_id[igas])
                    # Plot prior profile of the current gas
                    plt.plot(prior_gases[:, igas], prior_p, color='black', label=f"Prior profile of {gases_name}")
                    # Plot the retrieved profile of the current gas
                    plt.plot(gases[:, ilat, igas], pressure, label=f"Retrieved profile of {gases_name}")
                    plt.grid()
                    plt.ylim(0.01, 1000)
                    plt.yscale('log')
                    plt.gca().invert_yaxis()
                    plt.tick_params(labelsize=15)        
                    plt.xlabel(f"Volume Mixing Ratio", size=20)
                    plt.ylabel(f"Presssure [mbar]", size=20)
                    plt.legend(fontsize=20) 
                    # Save figure in the retrievals outputs directory
                    plt.savefig(f"{subdir}{itest}_retrieved_gas_{gases_name}_profile_at_{latitude[ilat]}.png", dpi=150, bbox_inches='tight')
                    #plt.savefig(f"{subdir}{itest}_retrieved_temperature_zonal_wind.eps", dpi=100)
                    # Close figure to avoid overlapping between plotting subroutines
                    plt.close()

def PlotRetrievedGasesProfileSuperpose(over_axis):

    print('Plotting NEMESIS retrieved gases profiles superpose...')
    # If subdirectory does not exist, create it
    dir = '../retrievals/retrieved_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Retrieval outputs directory path
    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/"
    # Array of prior file names
    prior = ['jupiter_v2021']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        # If retrieval test subdirectory does not exist, create it
        subdir = f"{dir}/{iprior}_temp_aerosol1-10mu_C2H2p-C2H4-C2H6p_NH3-PH3-parametric_no852_no887/"
        if not os.path.exists(subdir):
                os.makedirs(subdir)

        retrieval_test = [
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_NH3_no852_no887",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_NH3pt_no852_no887",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_NH3_PH3pt_no852_no887",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_NH3pt_PH3pt_no852_no887"
                        ]
        ntest = len(retrieval_test)
        # Plot figure of gases abundances profiles
        plt.figure(figsize=(7, 10), dpi=100)
        # Loop over each retrieval tests for the current prior file
        for itest in retrieval_test:           
            # Read profile data from NEMESIS prior file 
            _, prior_p, _, _, prior_gases, _, _, _ = ReadTemperatureGasesPriorProfile(f"{fpath}{itest}/core_1/")
            # Read retrieved profiles from .prf outputs files
            _, gases, latitude, _, pressure, nlat, _, ngas, gases_id = ReadprfFiles(filepath=f"{fpath}{itest}", over_axis=over_axis)
            for ilat in range(nlat):
                # Mapping the gases cross-section with zind location
                for igas in range(ngas):
                    gases_name, _ = RetrieveGasesNames(gas_name=False, gas_id=gases_id[igas])               
                    # Plot prior profile of the current gas
                    plt.plot(prior_gases[:, igas], prior_p, color='black', label=f"Prior profile of {gases_name}")
                    # Plot the retrieved profile of the current gas
                    plt.plot(gases[:, ilat, igas], pressure, label=f"Retrieved profile of {gases_name}")
            plt.grid()
            # plt.ylim(0.01, 1000)
            plt.yscale('log')
            plt.gca().invert_yaxis()
            plt.tick_params(labelsize=15)        
            plt.xlabel(f"Volume Mixing Ratio", size=20)
            plt.ylabel(f"Presssure [mbar]", size=20)
            plt.legend(fontsize=20) 
            # Save figure in the retrievals outputs directory
            plt.savefig(f"{subdir}{itest}_retrieved_gas_{gases_name}_profile_at_{latitude[ilat]}.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{subdir}{itest}_retrieved_temperature_zonal_wind.eps", dpi=100)
            # Close figure to avoid overlapping between plotting subroutines
            plt.close()

def PlotRetrievedGasesCrossSection(over_axis):

    print('Plotting NEMESIS retrieved gases cross-section...')
    # If subdirectory does not exist, create it
    dir = '../retrievals/retrieved_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    #  Load Jupiter zonal jets data to determine belts and zones location
    ejets_c, wjets_c, nejet, nwjet = ReadZonalWind("../inputs/jupiter_jets.dat")
    # Retrieval outputs directory path
    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/"
    # Array of prior file names
    prior = ['jupiter_v2021']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        # If retrieval test subdirectory does not exist, create it
        subdir = f"{dir}{iprior}_selected_meridien_tests/GasesHydrocarbons/"
        if not os.path.exists(subdir):
            os.makedirs(subdir)
        retrieval_test = [
            f"{iprior}_temp_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H4-C2H6_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_NH3_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H4-C2H6_NH3_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6pknee02mbar_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6pknee02mbar_NH3_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H4-C2H6_NH3p_fshfix_no852_no887_aprmodif",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6pknee02mbar_NH3p_fshfix_no852_no887_aprmodif"
                        ]
        ntest = len(retrieval_test)
        # Loop over each retrieval tests for the current prior file
        for itest in retrieval_test:
            # Read retrieved profiles from .prf outputs files
            _, gases, latitude, _, pressure, _, _, ngas, gases_id = ReadprfFiles(filepath=f"{fpath}{itest}", over_axis=over_axis)
            # Mapping the gases cross-section with zind location
            for igas in range(ngas):
                gases_name, _ = RetrieveGasesNames(gas_name=False, gas_id=gases_id[igas])   
                # Transform into log base 10 
                gases[:, :, igas] = xlogy(np.sign(gases[:, :, igas]), gases[:, :, igas]) / np.log(10)

                min = np.nanmin(gases[:, :, igas]) 
                max = np.nanmax(gases[:, :, igas])             

                # Mapping the gases cross-section alone
                plt.figure(figsize=(8, 6), dpi=100)
                im = plt.contourf(latitude, pressure, gases[:, :, igas], cmap='GnBu', levels=15)
                plt.contour(latitude, pressure, gases[:, :, igas], colors="white", levels=15, linestyles = '-' , linewidths=0.25)
                plt.ylim(0.1, 1000)
                # plt.xlim(-80, 80)
                plt.yscale('log')
                plt.gca().invert_yaxis()
                plt.tick_params(labelsize=15)     
                if over_axis=="longitude":
                    plt.xlabel("System III West Longitude", size=20)
                elif over_axis=="latitude":
                    plt.xlabel("Planetographic Latitude", size=20)
                plt.ylabel(f"Presssure [mbar]", size=20)
                cbar = plt.colorbar(im, extend='both', fraction=0.046, pad=0.05, orientation='vertical')
                cbar.ax.tick_params(labelsize=15)
                cbar.set_label(f"Retrieved {gases_name}", fontsize=20)   
                # Save figure in the retrievals outputs directory
                plt.savefig(f"{subdir}{itest}_retrieved_gas_{gases_name}.png", dpi=150, bbox_inches='tight')
                #plt.savefig(f"{subdir}{itest}_retrieved_temperature.eps", dpi=100)
                # Close figure to avoid overlapping between plotting subroutines
                plt.close()

def PlotRetrievedGasesMaps(over_axis="2D"):

    print('Plotting NEMESIS retrieved gases maps...')
     # If subdirectory does not exist, create it
    dir = '../retrievals/retrieved_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Retrieval outputs directory path
    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/retrieval_GreatRedSpot/"
    # Array of prior file names
    prior = ['jupiter_v2021']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        retrieval_test = [
                        # f"jupiter_v2021_temp_aerosol1-10mu-800mbar-05scale-01_GRS_no852_no887",
                        # f"jupiter_v2021_temp_aerosol1-10mu-800mbar-05scale-01_NH3_GRS_no852_no887",
                        "jupiter_v2021_temp_aerosol1-10mu-800mbar-05scale-01_NH3_GRS_halfdeg_no852_no887"
                        ]
        ntest = len(retrieval_test)
        # Loop over each prior used for retrievals tests
        for itest in retrieval_test:
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{iprior}_selected_meridien_tests/{itest}/GRS_aerosol_map/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            # Read retrieved profiles from .prf outputs files
            temperature, gases, latitude, longitude, height, pressure, ncoor, nlevel, nlat, nlon, ngas, gases_id = ReadprfFiles(filepath=f"{fpath}{itest}", over_axis=over_axis)

            for igas in [0, 1, 2, 3, 4]:
                gases_name, _ = RetrieveGasesNames(gas_name=False, gas_id=gases_id[igas])
                # Determine extreme values of temperature and levels for mapping
                gmax = np.nanmax(gases[:, :, :, igas])
                gmin = np.nanmin(gases[:, :, :, igas])
                levels_cmap = np.linspace(gmin, gmax, num=20, endpoint=True)

                # # Mapping the meridian gases cross-section
                # lon_index = (longitude == 157.5)
                # gaskeep = gases[:, :, lon_index, igas]
                # plt.figure(figsize=(8, 6))
                # im = plt.contourf(latitude, pressure, gaskeep[:, :, 0], cmap='viridis', levels=levels_cmap)
                # # plt.contour(latitude, pressure, gaskeep[:, :, 0], levels=levels, colors="white")
                # plt.ylim(500, 1000) if igas < 2 else plt.ylim(0.001,1)
                # # plt.xlim(-80, 80)
                # plt.yscale('log')
                # plt.gca().invert_yaxis()
                # plt.tick_params(labelsize=15)        
                # plt.xlabel("Planetographic Latitude", size=20)
                # plt.ylabel(f"Presssure [mbar]", size=20)
                # plt.title(f"Great Red Spot structure at {float(longitude[lon_index])}"+"$^{\circ}$W")
                # cbar = plt.colorbar(im, extend='both', fraction=0.046, pad=0.05, orientation='vertical')
                # cbar.ax.tick_params(labelsize=15)
                # cbar.set_label("Volume Mixing Ratio", fontsize=20)   
                # # Save figure in the retrievals outputs directory
                # plt.savefig(f"{subdir}{itest}_gas_{gases_name}_meridian_cross_section_at_lon{float(longitude[lon_index])}.png", dpi=150, bbox_inches='tight')
                # #plt.savefig(f"{subdir}{itest}_meridian_cross_section_at_lon{float(longitude[lon_index])}.eps", dpi=100)
                # # Close figure to avoid overlapping between plotting subroutines
                # plt.close()

                # # Mapping the zonal gases cross-section
                # lat_index = (latitude == -20.5)
                # gaskeep = gases[:, lat_index, :, igas]
                # plt.figure(figsize=(8, 6))
                # im = plt.contourf(longitude, pressure, gaskeep[0, :, :], cmap='viridis', levels=levels_cmap)
                # # plt.contour(longitude, pressure, gaskeep[:, :, 0], levels=levels, colors="white")
                # plt.ylim(500, 1000) if igas < 2 else plt.ylim(0.001,1)
                # # plt.xlim(-80, 80)
                # plt.yscale('log')
                # plt.gca().invert_yaxis()
                # plt.gca().invert_xaxis()
                # plt.tick_params(labelsize=15)        
                # plt.xlabel("System III West longitude", size=20)
                # plt.ylabel(f"Presssure [mbar]", size=20)
                # plt.title(f"Great Red Spot structure at {float(latitude[lat_index])}"+"$^{\circ}$")
                # cbar = plt.colorbar(im, extend='both', fraction=0.046, pad=0.05, orientation='vertical')
                # cbar.ax.tick_params(labelsize=15)
                # cbar.set_label("Volume Mixing Ratio", fontsize=20)   
                # # Save figure in the retrievals outputs directory
                # plt.savefig(f"{subdir}{itest}_gas_{gases_name}_zonal_cross_section_at_lat{float(latitude[lat_index])}.png", dpi=150, bbox_inches='tight')
                # #plt.savefig(f"{subdir}{itest}_zonal_cross_section_at_lat{float(latitude[lat_index])}.eps", dpi=100)
                # # Close figure to avoid overlapping between plotting subroutines
                # plt.close()

                # Mapping the vertical gases cross-section
                ptarget = [440, 500, 550, 600, 700, 800, 1000] if igas < 2 else [0.001, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03, 0.02, 0.01, 0.1]
                nrow = int((len(ptarget)+1)/2)
                yfig = int(len(ptarget)+1)
                fig, ax = plt.subplots(nrow, 2, figsize=(10, yfig), sharex=True, sharey=True)
                iax = 0
                for ipressure in range(len(ptarget)):
                    irow = [0,1,1,2,2,3,3] if igas <2 else [0,1,1,2,2,3,3,4,4,5,5]
                    icol = [0,0,1,0,1,0,1] if igas <2 else [0,0,1,0,1,0,1,0,1,0,1]
                    ititle = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)']
                    # Remove the frame of the empty subplot
                    ax[0][1].set_frame_on(False)
                    ax[0][1].tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                    # find pressure index 
                    ind_pres = np.where(pressure >= ptarget[ipressure])
                    # print(pressure)
                    # set extreme values 
                    gmax = np.nanmax(gases[ind_pres[0][-1], :, :, igas])
                    gmin = np.nanmin(gases[ind_pres[0][-1], :, :, igas])
                    levels_cmap = np.linspace(gmin, gmax, num=20, endpoint=True)
                    # subplot showing the regional radiance maps
                    im = ax[irow[iax]][icol[iax]].imshow(gases[ind_pres[0][-1], :, :, igas], cmap='viridis', vmin=gmin, vmax=gmax, # levels=levels_cmap,
                                                            origin='lower', extent=[longitude[0],longitude[-1],latitude[0],latitude[-1]])
                    ax[irow[iax]][icol[iax]].tick_params(labelsize=14)
                    ax[irow[iax]][icol[iax]].set_title(ititle[iax]+f"    {ptarget[ipressure]} mbar    {gases_name}", fontfamily='sans-serif', loc='left', fontsize=12)
                    cbar = fig.colorbar(im, ax=ax[irow[iax]][icol[iax]], extend='both', fraction=0.04, pad=0.05, format='%.2e')#, orientation='horizontal')
                    cbar.ax.tick_params(labelsize=12)
                    cbar.ax.locator_params(nbins=6)
                    cbar.set_label("Volume Mixing Ratio", fontsize=8)
                    iax+=1 
                plt.axes([0.15, 0.1, 0.8, 0.8], frameon=False) 
                plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                plt.xlabel("System III West Longitude", size=18)
                plt.ylabel("Planetographic Latitude", size=18)
                # Save figure showing calibation method 
                plt.savefig(f"{subdir}{itest}_gas_{gases_name}_maps_at_11_pressure_levels.png", dpi=150, bbox_inches='tight')
                #plt.savefig(f"{subdir}{itest}_gases_maps_at_11_pressure_levels.eps", dpi=900)
                # Clear figure to avoid overlapping between plotting subroutines
                plt.close()

            
                fig= plt.figure(figsize=(8, 4))
                ipressure = 1
                # find pressure index 
                ind_pres = np.where(pressure >= ptarget[ipressure])
                # print(pressure)
                # set extreme values 
                gmax = 3.e-5#np.nanmax(gases[ind_pres[0][-1], :, :, igas])
                gmin = 1.7e-5#np.nanmin(gases[ind_pres[0][-1], :, :, igas])
                levels_cmap = np.linspace(gmin, gmax, num=20, endpoint=True)
                # subplot showing the regional radiance maps
                im = plt.imshow(gases[ind_pres[0][-1], :, :, igas], cmap='bone', vmin=gmin, vmax=gmax, # levels=levels_cmap,
                                                            origin='lower', extent=[longitude[0],longitude[-1],latitude[0],latitude[-1]])
                plt.tick_params(labelsize=14)
                cbar = fig.colorbar(im, extend='both', fraction=0.04, pad=0.05, format='%.2e')#, orientation='horizontal')
                cbar.ax.tick_params(labelsize=12)
                # cbar.ax.locator_params(nbins=6)
                cbar.ax.set_title(f"{gases_name} at {ptarget[ipressure]} mbar \n Volume Mixing Ratio ", fontsize=12, pad=12)
                plt.xlabel("System III West Longitude", size=15)
                plt.ylabel("Planetographic Latitude", size=15)
                # Save figure showing calibation method 
                plt.savefig(f"{subdir}{itest}_gas_{gases_name}_maps_at_pressure_{ptarget[ipressure]}.png", dpi=150, bbox_inches='tight')
                #plt.savefig(f"{subdir}{itest}_gases_maps_at_11_pressure_levels.eps", dpi=900)
                # Clear figure to avoid overlapping between plotting subroutines
                plt.close()

def PlotRetrievedGasesMeridianProfiles(over_axis):

    print("Plot meridian 'comparison with previous studies' profiles of hydrocarbons ... ")
    # If subdirectory does not exist, create it
    dir = '../retrievals/retrieved_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Retrieval outputs directory path
    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/"
    # Array of prior file names
    prior = ['jupiter_v2023']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        retrieval_test = [
            f"{iprior}_temp_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H6_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_NH3_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H6_NH3_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H6pknee02mbar_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H6_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H6pknee02mbar_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H6pknee02mbar_NH3_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H6_NH3p_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_NH3p_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H6pknee02mbar_NH3p_ktable-highreso_no852_no887",
                    ]
        ntest = len(retrieval_test)
        # Reading previous studies observations:
        NixonCIRSpressure, NixonCIRSlatitude, NixonCIRSTemp, NixonCIRSc2h2, NixonCIRSc2h6= ReadExcelFiles(fpath="/Users/db496/Documents/Research/Observations/archives/Nixon_CxHy_observations/cirs_hcs_temp_errs.xls")

        NixonVoyagerpressure, NixonVoyagerlatitude, NixonVoyagerTemp, NixonVoyagerc2h2, NixonVoyagerc2h6= ReadExcelFiles(fpath="/Users/db496/Documents/Research/Observations/archives/Nixon_CxHy_observations/v1_hcs_temp_errs.xls")


        Fletcher16CIRStemp, Fletcher16CIRSgases, Fletcher16CIRSlatitude, Fletcher16CIRSheight, \
        Fletcher16CIRSpressure, Fletcher16CIRSnlat, Fletcher16CIRSnlevel, Fletcher16CIRSngas, \
        Fletcher16CIRSgases_id= ReadPreviousWork(fpath="/Users/db496/Documents/Research/Observations/archives/Fletcher_2016_Icarus_jupiter_TEXES_CIRS_comparison/CIRS_flyby2000/",
                            ncore= 78, corestart=1, namerun="nh3knee_0.8", over_axis="latitude")

        Fletcher16TEXEStemp, Fletcher16TEXESgases, Fletcher16TEXESlatitude, Fletcher16TEXESheight, \
        Fletcher16TEXESpressure, Fletcher16TEXESnlat, Fletcher16TEXESnlevel, Fletcher16TEXESngas, \
        Fletcher16TEXESgases_id= ReadPreviousWork(fpath="/Users/db496/Documents/Research/Observations/archives/Fletcher_2016_Icarus_jupiter_TEXES_CIRS_comparison/TEXES2014/",
                            ncore= 75, corestart=2, namerun="newscale", over_axis="latitude")            

        Melin18TEXEStemp, Melin18TEXESgases, Melin18TEXESlatitude, Melin18TEXESheight, \
        Melin18TEXESpressure, Melin18TEXESnlat, Melin18TEXESnlevel, Melin18TEXESngas, \
        Melin18TEXESgases_id= ReadPreviousWork(fpath="/Users/db496/Documents/Research/Observations/archives/Melin_2018_paper_longterm_variability_Jupiter_TEXES/texes_2017_jan_temp_hcs2/",
                            ncore= 69, corestart=1, namerun="nemesis", over_axis="latitude") 
        
        Sinclair_lat, Sinclair_lon, c2h2, c2h2err, c2h6, c2h6err = ReadDatFiles(filepath="/Users/db496/Documents/Research/Observations/archives/Sinclair_2017_Icarus_Jupiter_auroral_related_stratospheric_heating_chemistry_part1/atmos02A_cxhy_results_for_deborah.dat")
        # Zonally-average James Sinclair 2017 Icarus paper data:
        Sinclair_c2h2 = np.empty((len(Sinclair_lat)))
        Sinclair_c2h6 = np.empty((len(Sinclair_lat)))
        Sinclair_c2h2err = np.empty((len(Sinclair_lat)))
        Sinclair_c2h6err = np.empty((len(Sinclair_lat)))
        for ilat in range(len(Sinclair_lat)):
            Sinclair_c2h2[ilat] = np.nanmean(c2h2[ilat,:])
            Sinclair_c2h6[ilat] = np.nanmean(c2h6[ilat,:])
            Sinclair_c2h2err[ilat] = max(np.sqrt(np.sum(c2h2err[ilat,:]**2)), np.std(np.nanmean(c2h2err[ilat,:])))
            Sinclair_c2h6err[ilat] = max(np.sqrt(np.sum(c2h6err[ilat,:]**2)), np.std(np.nanmean(c2h6err[ilat,:])))
        # Convert Sinclair's planetographic latitude to planetographic latitude:
        # Sinclair_lat = convert_to_planetocentric(Sinclair_lat, 'jupiter')
        # plt.contourf(Sinclair_lon, Sinclair_lat, c2h2)#, lw=2, marker='.')
        # plt.colorbar()
        # plt.show()
        # plt.contourf(Sinclair_lon, Sinclair_lat, c2h6)#, lw=2, marker='.')
        # plt.colorbar()
        # plt.show()
        
        # Loop over each retrieval tests for the current prior file
        for i, itest in enumerate(retrieval_test):
            col = cmap(i/ntest)
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{iprior}_selected_meridien_tests/{itest}/comparison_previous_study/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            # Read ChiSquare values from log files
            # chisquare, latitude, nlat = ReadLogFiles(filepath=f"{fpath}{itest}", over_axis="latitude")
            # Read retrieved profiles from .prf outputs files
            temperature, gases, latitude, _, pressure, _, _, ngas, gases_id = ReadprfFiles(filepath=f"{fpath}{itest}", over_axis="latitude")
            
            radiance, wavenumb, rad_err, rad_fit, temp_prior_mre, \
                                                    temp_errprior_mre, temp_fit_mre, temp_errfit_mre, \
                                                    aer_mre, aer_err,  aer_fit, fit_err, \
                                                    gas_scale, gas_scaleerr,  gas_scalefit, gas_errscalefit, \
                                                    gas_vmr, gas_vmrerr, gas_vmrfit, gas_errvmrfit,\
                                                    gas_fsh, gas_fsherr, gas_fshfit, gas_errfshfit, \
                                                    gas_abunerr, \
                                                    _, _  = ReadmreFiles(filepath=f"{fpath}{itest}", over_axis="latitude", gas_name=['C2H2', 'C2H4', 'C2H6'])
            # if i == 0:
            #     ind_pres = np.where(pressure >= 1.)
            #     # plt.plot(Fletcher16CIRSlatitude, Fletcher16CIRSgases[ind_pres[0][-1], :, 2], color='blue', lw=1, marker='x', markersize=5, label='Cassini Fletcher+ 2016')
            #     plt.plot(Sinclair_lat, Sinclair_c2h2, lw=1, marker='v', markersize=7, label='Cassini Sinclair+ 2017')
            #     plt.legend()
            #     plt.show()           
            # reducedlat = np.where((latitude >0)&(latitude<20))
            # lat= latitude[reducedlat]
            # for ilat in range(len(lat)):
            #     col = cmap(ilat/len(lat))
            #     plt.plot(gas_abunerr[0,:,ilat], pressure, label=f"err lat {lat[ilat]}", lw=1, marker='.', color=col, alpha=0.6)
            # # plt.plot(gases[:, 90, 2], pressure, label="prf", lw=1, marker='.')
            # plt.gca().set_xscale("log")
            # plt.gca().set_yscale("log")
            # plt.gca().invert_yaxis()
            # plt.legend()
            # lat_average = np.where((latitude > -85) & (latitude < -75))
            # c2h4 = gases[:, lat_average, 3]
            # c2h4_average = np.empty((120))
            # for z in range(len(pressure)):
            #     c2h4_average[z] = np.nanmean(c2h4[z, :])
            # print(c2h4_average)
            # plt.plot(c2h4_average, pressure)
            # plt.gca().set_xscale("log")
            # plt.gca().set_yscale("log")
            # plt.gca().invert_yaxis()
            # plt.show()

            ptarget = [800., 500., 300., 100., 10., 5., 3., 2., 1., 0.5, 0.1, 0.05, 0.01]
            for ipressure in range(len(ptarget)):
                ind_pres = np.where(pressure >= ptarget[ipressure])
                ind_Nixpres = np.where(NixonCIRSpressure >= ptarget[ipressure])
                
                c2h2_err = np.empty(len(latitude))
                c2h6_err = np.empty(len(latitude))
                for ilat in range(len(latitude)):
                    if ~np.isnan(gas_abunerr[0, ind_pres[0][-1], ilat]):
                        c2h2_err[ilat] = gas_abunerr[0, ind_pres[0][-1], ilat] 
                    else:
                        c2h2_err[ilat] = gases[ind_pres[0][-1], ilat, 2]*gas_errscalefit[0,ilat]
                        # print(gas_errscalefit[0,ilat])
                        # print(gases[ind_pres[0][-1], ilat, 2]*gas_errscalefit[0,ilat])

                    if ~np.isnan(gas_abunerr[2, ind_pres[0][-1], ilat]):
                        c2h6_err[ilat] = np.nanmax(gas_abunerr[2, ind_pres[0][-1], :])
                    else:
                        c2h6_err[ilat] = gases[ind_pres[0][-1], ilat, 4]*gas_errscalefit[2,ilat]

                fig = plt.figure(figsize=(10, 6))
                plt.title(r'C$_{2}$H$_{2}$'+f" at {ptarget[ipressure]} mbar")
                # Current work 
                plt.errorbar(latitude, gases[ind_pres[0][-1], :, 2], yerr=c2h2_err[:], color='grey', lw=1, marker='.', markersize=7, 
                            markerfacecolor='black', markeredgewidth=0, alpha=0.7, capthick=0.4, label=r'VLT/VISIR 2018 May 24$^{th}$-27$^{th}$')
                # Nixon et al., 2016 Icarus data. Cassini CIRS flyby data 2000
                plt.plot(NixonVoyagerlatitude, NixonVoyagerc2h2[ind_Nixpres[0][-1], :], color='red', lw=1, marker='*', markersize=7, label=r'Nixon et al.,(2010)     Voyager 1/IRIS 1979')
                # Nixon et al., 2016 Icarus data. Cassini CIRS flyby data 2000
                plt.plot(NixonCIRSlatitude, NixonCIRSc2h2[ind_Nixpres[0][-1], :], color='dodgerblue', lw=1, marker='P', markersize=7, label=r'Nixon et al.,(2010)    Cassini/CIRS 2000/2001')
                # Fletcher et al., 2016 Icarus data. Cassini CIRS flyby data 2000
                plt.plot(Fletcher16CIRSlatitude, Fletcher16CIRSgases[ind_pres[0][-1], :, 2], color='blue', lw=1, marker='x', markersize=5, label=r'Fletcher et al.,(2016)  Cassini/CIRS 2000/2001')
                # Fletcher et al., 2016 Icarus data. IRTF TEXES data 2014
                plt.plot(Fletcher16TEXESlatitude, Fletcher16TEXESgases[ind_pres[0][-1], :, 3], color='green', lw=1, marker='d', markersize=5, label=r'Fletcher et al.,(2016)  IRTF/TEXES 2014')
                # Melin et al,. 2018 Icarus data. IRTF TEXES data 2013-2017
                plt.plot(Melin18TEXESlatitude, Melin18TEXESgases[ind_pres[0][-1], :, 3], color='limegreen', lw=1, marker='p', markersize=5, label=r'Melin et al.,(2018)     IRTF/TEXES 2017 January')
                if ptarget[ipressure] == 1.:
                    # Sinclair et al,. 2017 Icarus PartI data
                    plt.plot(Sinclair_lat, Sinclair_c2h2, color='orange', lw=1, marker='v', markersize=5, label=r'Sinclair et al.,(2017)   Cassini/CIRS 2000/2001')
                plt.tick_params(labelsize=12)            
                plt.grid()
                plt.xlabel(f"Planetographic Latitude", size=20)                
                plt.xlim(-90, 90)
                plt.ylabel(f"Volume Mixing Ratio", size=20)
                # plt.gca().set_yscale("log")
                plt.legend(bbox_to_anchor=(0,1.1,1,0.2), loc="lower left", mode="expand", fontsize=12)
                plt.savefig(f"{subdir}hydrocarbons_C2H2_at{ptarget[ipressure]}mbar_comparison_with_previous_studies.png", dpi=150, bbox_inches='tight')
                plt.close()

                fig = plt.figure(figsize=(10, 6))
                plt.title(r'C$_{2}$H$_{6}$'+f" at {ptarget[ipressure]} mbar")
                # Current work
                plt.errorbar(latitude, gases[ind_pres[0][-1], :, 4], yerr=c2h6_err[:], color='grey', lw=1, marker='.', markersize=7, 
                            markerfacecolor='black', markeredgewidth=0, alpha=0.7, capthick=0.4, label=r'VLT/VISIR 2018 May 24$^{th}$-27$^{th}$')
                # Nixon et al., 2016 Icarus data. Cassini CIRS flyby data 2000
                plt.plot(NixonVoyagerlatitude, NixonVoyagerc2h6[ind_Nixpres[0][-1], :], color='red', lw=1, marker='*', markersize=7, label=r'Nixon et al.,(2010) Voyager 1/IRIS')
                # Nixon et al., 2016 Icarus data. Cassini CIRS flyby data 2000
                plt.plot(NixonCIRSlatitude, NixonCIRSc2h6[ind_Nixpres[0][-1], :], color='dodgerblue', lw=1, marker='P', markersize=7, label=r'Nixon et al.,(2010) Cassini/CIRS')
                # Fletcher et al., 2016 Icarus data. Cassini CIRS flyby data 2000
                plt.plot(Fletcher16CIRSlatitude, Fletcher16CIRSgases[ind_pres[0][-1], :, 4], color='blue', lw=1, marker='x', markersize=5, label=r'Fletcher et al.,(2016) Cassini/CIRS')
                # Fletcher et al., 2016 Icarus data. IRTF TEXES data 2014
                plt.plot(Fletcher16TEXESlatitude, Fletcher16TEXESgases[ind_pres[0][-1], :, 5], color='green', lw=1, marker='d', markersize=5, label=r'Fletcher et al.,(2016) IRTF/TEXES 2014')
                # Melin et al,. 2018 Icarus data. IRTF TEXES data 2013-2017
                plt.plot(Melin18TEXESlatitude, Melin18TEXESgases[ind_pres[0][-1], :, 5], color='limegreen', lw=1, marker='p', markersize=5, label=r'Melin et al.,(2018) IRTF/TEXES 2017 January')
                if ptarget[ipressure] == 1.:
                    # Sinclair et al,. 2017 Icarus PartI data
                    plt.plot(Sinclair_lat, Sinclair_c2h6, color='orange', lw=1, marker='v', markersize=5, label=r'Sinclair et al.,(2017) Cassini/CIRS 2000/2001')
                plt.tick_params(labelsize=12)            
                plt.grid()
                plt.xlabel(f"Planetographic Latitude", size=20)
                plt.xlim(-90, 90)
                plt.ylabel(f"Volume Mixing Ratio", size=20)
                plt.legend(bbox_to_anchor=(0,1.1,1,0.2), loc="lower left", mode="expand", fontsize=10)
                plt.savefig(f"{subdir}hydrocarbons_C2H6_at{ptarget[ipressure]}mbar_comparison_with_previous_studies.png", dpi=150, bbox_inches='tight')
                plt.close()

                fig = plt.figure(figsize=(10, 4))
                plt.title(f"Temperature at {ptarget[ipressure]} mbar")
                # Current work 
                plt.errorbar(latitude, temperature[ind_pres[0][-1], :], yerr=temp_errfit_mre[ind_pres[0][-1], :], color='grey', lw=1, marker='.', markersize=7, 
                            markerfacecolor='black', markeredgewidth=0, alpha=0.7, capthick=0.4,  label=r'VLT/VISIR 2018 May 24$^{th}$-27$^{th}$')
                # Nixon et al., 2016 Icarus data. Cassini CIRS flyby data 2000
                plt.plot(NixonVoyagerlatitude, NixonVoyagerTemp[ind_Nixpres[0][-1], :], color='red', lw=1, marker='*', markersize=7, label=r'Nixon et al.,(2010) Voyager 1/IRIS 1979')
                # Nixon et al., 2016 Icarus data. Cassini CIRS flyby data 2000
                plt.plot(NixonCIRSlatitude, NixonCIRSTemp[ind_Nixpres[0][-1], :], color='dodgerblue', lw=1, marker='P', markersize=7, label=r'Nixon et al.,(2010) Cassini/CIRS 2000/2001')
                # Fletcher et al., 2016 Icarus data. Cassini CIRS flyby data 2000
                plt.plot(Fletcher16CIRSlatitude, Fletcher16CIRStemp[ind_pres[0][-1], :], color='blue', lw=1, marker='x', markersize=5, label=r'Fletcher et al.,(2016) Cassini/CIRS 2000/2001')
                # Fletcher et al., 2016 Icarus data. IRTF TEXES data 2014
                plt.plot(Fletcher16TEXESlatitude, Fletcher16TEXEStemp[ind_pres[0][-1], :], color='green', lw=1, marker='d', markersize=5, label=r'Fletcher et al.,(2016) IRTF/TEXES 2014')
                # Melin et al,. 2018 Icarus data. IRTF TEXES data 2013-2017
                plt.plot(Melin18TEXESlatitude, Melin18TEXEStemp[ind_pres[0][-1], :], color='limegreen', lw=1, marker='p', markersize=5, label=r'Melin et al.,(2018) IRTF/TEXES 2017 January')
                plt.tick_params(labelsize=12)            
                plt.grid()
                plt.xlabel(f"Planetographic Latitude", size=20)
                plt.xlim(-90, 90)
                plt.ylabel(f"Temperature [K]", size=20)
                plt.legend(bbox_to_anchor=(0,1.1,1,0.2), loc="lower left", mode="expand", fontsize=12)
                plt.savefig(f"{subdir}temperature_at{ptarget[ipressure]}mbar_comparison_with_previous_studies.png", dpi=150, bbox_inches='tight')
                plt.close()
                




            ptarget = [0.5, 5., 100., 300.]
            ititle = ['(a)', '(b)', '(c)', '(d)'] 
            fig, ax = plt.subplots(len(ptarget), 1, figsize=(9, 14), sharex=True, sharey=False)
            for iax in range(len(ptarget)):
                ind_pres = np.where(pressure >= ptarget[iax])
                ind_Nixpres = np.where(NixonCIRSpressure >= ptarget[iax])
                ax[iax].set_title(ititle[iax]+f"      {ptarget[iax]} mbar", size=18, fontfamily='sans-serif', loc='left')
                # Current work 
                ax[iax].errorbar(latitude, temperature[ind_pres[0][-1], :], yerr=temp_errfit_mre[ind_pres[0][-1], :], color='grey', lw=1, marker='.', markersize=7, 
                            markerfacecolor='black', markeredgewidth=0, alpha=0.7, capthick=0.4,  label=r'VLT/VISIR 2018 May 24$^{th}$-27$^{th}$')
                # Nixon et al., 2016 Icarus data. Cassini CIRS flyby data 2000
                ax[iax].plot(NixonVoyagerlatitude, NixonVoyagerTemp[ind_Nixpres[0][-1], :], color='red', lw=1, marker='*', markersize=7, label=r'Nixon et al.,(2010) Voyager 1/IRIS 1979')
                # Nixon et al., 2016 Icarus data. Cassini CIRS flyby data 2000
                ax[iax].plot(NixonCIRSlatitude, NixonCIRSTemp[ind_Nixpres[0][-1], :], color='dodgerblue', lw=1, marker='P', markersize=7, label=r'Nixon et al.,(2010) Cassini/CIRS 2000/2001')
                # Fletcher et al., 2016 Icarus data. Cassini CIRS flyby data 2000
                ax[iax].plot(Fletcher16CIRSlatitude, Fletcher16CIRStemp[ind_pres[0][-1], :], color='blue', lw=1, marker='x', markersize=5, label=r'Fletcher et al.,(2016) Cassini/CIRS 2000/2001')
                # Fletcher et al., 2016 Icarus data. IRTF TEXES data 2014
                ax[iax].plot(Fletcher16TEXESlatitude, Fletcher16TEXEStemp[ind_pres[0][-1], :], color='green', lw=1, marker='d', markersize=5, label=r'Fletcher et al.,(2016) IRTF/TEXES 2014')
                # Melin et al,. 2018 Icarus data. IRTF TEXES data 2013-2017
                ax[iax].plot(Melin18TEXESlatitude, Melin18TEXEStemp[ind_pres[0][-1], :], color='limegreen', lw=1, marker='p', markersize=5, label=r'Melin et al.,(2018) IRTF/TEXES 2017 January')
                ax[iax].grid()
                ax[iax].tick_params(labelsize=18)
                ax[iax].set_xlim(-90, 90)
            handles, labels = ax[0].get_legend_handles_labels()  
            fig.legend(handles, labels, bbox_to_anchor=(0.11,0.9,0.8,0.2), loc="lower left", mode="expand", fontsize=15)
            plt.axes([0.1, 0.1, 0.8, 0.8], frameon=False) 
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)            
            plt.xlabel(f"Planetographic Latitude", size=20)
            plt.ylabel(f"Temperature [K]", size=20)
            plt.savefig(f"{subdir}temperature_at_several_pressures_comparison_with_previous_studies.png", dpi=150, bbox_inches='tight')
            plt.close()






            ptarget = [0.1, 0.5, 1., 5., 100., 300.]
            for ipres in range(len(ptarget)):
                ind_pres = np.where(pressure >= ptarget[ipres])
                ind_Nixpres = np.where(NixonCIRSpressure >= ptarget[ipres])
                
                c2h2_err = np.empty(len(latitude))
                c2h6_err = np.empty(len(latitude))
                for ilat in range(len(latitude)):
                    if ~np.isnan(gas_abunerr[0, ind_pres[0][-1], ilat]):
                        c2h2_err[ilat] = gas_abunerr[0, ind_pres[0][-1], ilat] 
                    else:
                        c2h2_err[ilat] = gases[ind_pres[0][-1], ilat, 2]*gas_errscalefit[0,ilat]
                        # print(gas_errscalefit[0,ilat])
                        # print(gases[ind_pres[0][-1], ilat, 2]*gas_errscalefit[0,ilat])

                    if ~np.isnan(gas_abunerr[2, ind_pres[0][-1], ilat]):
                        c2h6_err[ilat] = np.nanmax(gas_abunerr[2, ind_pres[0][-1], :])
                    else:
                        c2h6_err[ilat] = gases[ind_pres[0][-1], ilat, 4]*gas_errscalefit[2,ilat]
                    # print(gas_abunerr[2, ind_pres[0][-1], ilat], np.nanmax(gas_abunerr[2, ind_pres[0][-1], :]))

                fig, ax = plt.subplots(2, 1, figsize=(10, 9), sharex=True, sharey=False)

                # first subplot with C2H2
                ax[0].set_title(ititle[0]+r'      C$_{2}$H$_{2}$', size=18, fontfamily='sans-serif', loc='left')
                # Nixon et al., 2016 Icarus data. Cassini CIRS flyby data 2000
                ax[0].plot(NixonVoyagerlatitude, NixonVoyagerc2h2[ind_Nixpres[0][-1], :], color='red', lw=1, marker='*', markersize=7, label=r'Nixon et al.,(2010) Voyager 1/IRIS 1979')
                # Nixon et al., 2016 Icarus data. Cassini CIRS flyby data 2000
                ax[0].plot(NixonCIRSlatitude, NixonCIRSc2h2[ind_Nixpres[0][-1], :], color='dodgerblue', lw=1, marker='P', markersize=7, label=r'Nixon et al.,(2010) Cassini/CIRS 2000/2001')
                # Fletcher et al., 2016 Icarus data. Cassini CIRS flyby data 2000
                ax[0].plot(Fletcher16CIRSlatitude, Fletcher16CIRSgases[ind_pres[0][-1], :, 2], color='blue', lw=1, marker='x', markersize=5, label=r'Fletcher et al.,(2016) Cassini/CIRS 2000/2001')
                # Fletcher et al., 2016 Icarus data. IRTF TEXES data 2014
                ax[0].plot(Fletcher16TEXESlatitude, Fletcher16TEXESgases[ind_pres[0][-1], :, 3], color='green', lw=1, marker='d', markersize=5, label=r'Fletcher et al.,(2016) IRTF/TEXES 2014')
                # Melin et al,. 2018 Icarus data. IRTF TEXES data 2013-2017
                ax[0].plot(Melin18TEXESlatitude, Melin18TEXESgases[ind_pres[0][-1], :, 3], color='limegreen', lw=1, marker='p', markersize=5, label=r'Melin et al.,(2018) IRTF/TEXES 2017 January')
                # Sinclair et al,. 2017 Icarus PartI data
                if ptarget[ipres] == 1.:
                    ax[0].errorbar(Sinclair_lat, Sinclair_c2h2, Sinclair_c2h2err, color='orange', lw=1, marker='v', markersize=5, 
                            markerfacecolor='orange', markeredgewidth=2, alpha=1, capthick=2, label=r'Sinclair et al.,(2017) Cassini/CIRS 2000/2001')
                # VLT/VISIR
                ax[0].errorbar(latitude, gases[ind_pres[0][-1], :, 2], yerr=c2h2_err, color='grey', lw=1, marker='.', markersize=7, 
                            markerfacecolor='black', markeredgewidth=0, alpha=0.7, capthick=0.4, label=r'VLT/VISIR 2018 May 24$^{th}$-27$^{th}$')
                ax[0].set_yscale("log")
                ax[0].set_xlim(-90, 90)
                ax[0].grid()
                ax[0].tick_params(labelsize=18)
                t = ax[0].yaxis.get_offset_text()
                t.set_x(-0.06)
                # second subplot with C2H6
                ax[1].set_title(ititle[1]+r'      C$_{2}$H$_{6}$', size=18, fontfamily='sans-serif', loc='left')
                ax[1].errorbar(latitude, gases[ind_pres[0][-1], :, 4], c2h6_err, color='grey', lw=1, marker='.', markersize=7, 
                            markerfacecolor='black', markeredgewidth=0, alpha=0.6, capthick=0.4)
                # Nixon et al., 2016 Icarus data. Cassini CIRS flyby data 2000
                ax[1].plot(NixonVoyagerlatitude, NixonVoyagerc2h6[ind_Nixpres[0][-1], :], color='red', lw=1, marker='*', markersize=7)
                # Nixon et al., 2016 Icarus data. Cassini CIRS flyby data 2000
                ax[1].plot(NixonCIRSlatitude, NixonCIRSc2h6[ind_Nixpres[0][-1], :], color='dodgerblue', lw=1, marker='P', markersize=7)
                # Fletcher et al., 2016 Icarus data. Cassini CIRS flyby data 2000
                ax[1].plot(Fletcher16CIRSlatitude, Fletcher16CIRSgases[ind_pres[0][-1], :, 4], color='blue', lw=1, marker='x', markersize=5)
                # Fletcher et al., 2016 Icarus data. IRTF TEXES data 2014
                ax[1].plot(Fletcher16TEXESlatitude, Fletcher16TEXESgases[ind_pres[0][-1], :, 5], color='green', lw=1, marker='d', markersize=5)
                # Melin et al,. 2018 Icarus data. IRTF TEXES data 2013-2017
                ax[1].plot(Melin18TEXESlatitude, Melin18TEXESgases[ind_pres[0][-1], :, 5], color='limegreen', lw=1, marker='p', markersize=5)
                # Sinclair et al,. 2017 Icarus PartI data
                if ptarget[ipres] == 1.:
                    ax[1].errorbar(Sinclair_lat, Sinclair_c2h6, Sinclair_c2h6err, color='orange', lw=1, marker='v', markersize=5, 
                            markerfacecolor='orange', markeredgewidth=2, alpha=1, capthick=2)
                ax[1].grid()
                ax[1].set_xlim(-90, 90)
                ax[1].tick_params(labelsize=18) 
                t = ax[1].yaxis.get_offset_text()
                t.set_x(-0.06)
                # third subplot with C2H4
                # ax[2].set_title(ititle[2]+r'      C$_{2}$H$_{4}$', size=18, fontfamily='sans-serif', loc='left')
                # ax[2].errorbar(latitude, gases[ind_pres[0][-1], :, 3], gases[ind_pres[0][-1], :, 3]*gas_errscalefit[1, :], color='grey', lw=1, marker='.', markersize=7, 
                #             markerfacecolor='black', markeredgewidth=0, alpha=0.7, capthick=0.4)
                # ax[2].set_yscale("log")
                # ax[2].set_xlim(-90, 90)
                # ax[2].grid()
                # ax[2].tick_params(labelsize=18)
                # t = ax[2].yaxis.get_offset_text()
                # t.set_x(-0.06)
                handles, labels = ax[0].get_legend_handles_labels()  
                fig.legend(handles, labels, bbox_to_anchor=(0.11,0.92,0.8,0.2), loc="lower left", mode="expand", fontsize=15)
                plt.axes([0.1, 0.1, 0.8, 0.8], frameon=False) 
                plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)            
                plt.xlabel(f"Planetographic Latitude", size=20)
                plt.ylabel(f"Volume Mixing Ratio", size=20)
                plt.savefig(f"{subdir}hydrocarbons_at_{ptarget[ipres]}mbar_comparison_with_previous_studies.png", dpi=150, bbox_inches='tight')
                plt.close()

                fig, ax = plt.subplots(2, 1, figsize=(10, 9), sharex=True, sharey=False)
                # first subplot with C2H2
                ax[0].set_title(ititle[0]+r'      C$_{2}$H$_{2}$', size=18, fontfamily='sans-serif', loc='left')
                # Nixon et al., 2016 Icarus data. Cassini CIRS flyby data 2000
                ax[0].plot(NixonVoyagerlatitude, NixonVoyagerc2h2[ind_Nixpres[0][-1], :], color='red', lw=1, marker='*', markersize=7, label=r'Nixon et al.,(2010) Voyager 1/IRIS 1979')
                # Nixon et al., 2016 Icarus data. Cassini CIRS flyby data 2000
                ax[0].plot(NixonCIRSlatitude, NixonCIRSc2h2[ind_Nixpres[0][-1], :], color='dodgerblue', lw=1, marker='P', markersize=7, label=r'Nixon et al.,(2010) Cassini/CIRS 2000/2001')
                # Fletcher et al., 2016 Icarus data. Cassini CIRS flyby data 2000
                ax[0].plot(Fletcher16CIRSlatitude, Fletcher16CIRSgases[ind_pres[0][-1], :, 2], color='blue', lw=1, marker='x', markersize=5, label=r'Fletcher et al.,(2016) Cassini/CIRS 2000/2001')
                # Fletcher et al., 2016 Icarus data. IRTF TEXES data 2014
                ax[0].plot(Fletcher16TEXESlatitude, Fletcher16TEXESgases[ind_pres[0][-1], :, 3], color='green', lw=1, marker='d', markersize=5, label=r'Fletcher et al.,(2016) IRTF/TEXES 2014')
                # Melin et al,. 2018 Icarus data. IRTF TEXES data 2013-2017
                ax[0].plot(Melin18TEXESlatitude, Melin18TEXESgases[ind_pres[0][-1], :, 3], color='limegreen', lw=1, marker='p', markersize=5, label=r'Melin et al.,(2018) IRTF/TEXES 2017 January')
                # # Sinclair et al,. 2017 Icarus PartI data
                # if ptarget[ipres] == 1.:
                #     ax[0].errorbar(Sinclair_lat, Sinclair_c2h2, Sinclair_c2h2err, color='orange', lw=1, marker='v', markersize=5, 
                #             markerfacecolor='orange', markeredgewidth=2, alpha=1, capthick=2, label=r'Sinclair et al.,(2017) Cassini/CIRS 2000/2001')
                # VLT/VISIR
                ax[0].errorbar(latitude, gases[ind_pres[0][-1], :, 2], yerr=c2h2_err, color='grey', lw=1, marker='.', markersize=7, 
                            markerfacecolor='black', markeredgewidth=0, alpha=0.7, capthick=0.4, label=r'VLT/VISIR 2018 May 24$^{th}$-27$^{th}$')
                # ax[0].set_yscale("log")
                ax[0].set_ylim(0.3e-7, 0.8e-6)
                ax[0].set_xlim(-70, 70)
                ax[0].grid()
                ax[0].tick_params(labelsize=18)
                t = ax[0].yaxis.get_offset_text()
                t.set_x(-0.06)
                # second subplot with C2H6
                ax[1].set_title(ititle[1]+r'      C$_{2}$H$_{6}$', size=18, fontfamily='sans-serif', loc='left')
                ax[1].errorbar(latitude, gases[ind_pres[0][-1], :, 4], c2h6_err, color='grey', lw=1, marker='.', markersize=7, 
                            markerfacecolor='black', markeredgewidth=0, alpha=0.6, capthick=0.4)
                # Nixon et al., 2016 Icarus data. Cassini CIRS flyby data 2000
                ax[1].plot(NixonVoyagerlatitude, NixonVoyagerc2h6[ind_Nixpres[0][-1], :], color='red', lw=1, marker='*', markersize=7)
                # Nixon et al., 2016 Icarus data. Cassini CIRS flyby data 2000
                ax[1].plot(NixonCIRSlatitude, NixonCIRSc2h6[ind_Nixpres[0][-1], :], color='dodgerblue', lw=1, marker='P', markersize=7)
                # Fletcher et al., 2016 Icarus data. Cassini CIRS flyby data 2000
                ax[1].plot(Fletcher16CIRSlatitude, Fletcher16CIRSgases[ind_pres[0][-1], :, 4], color='blue', lw=1, marker='x', markersize=5)
                # Fletcher et al., 2016 Icarus data. IRTF TEXES data 2014
                ax[1].plot(Fletcher16TEXESlatitude, Fletcher16TEXESgases[ind_pres[0][-1], :, 5], color='green', lw=1, marker='d', markersize=5)
                # Melin et al,. 2018 Icarus data. IRTF TEXES data 2013-2017
                ax[1].plot(Melin18TEXESlatitude, Melin18TEXESgases[ind_pres[0][-1], :, 5], color='limegreen', lw=1, marker='p', markersize=5)
                # Sinclair et al,. 2017 Icarus PartI data
                # if ptarget[ipres] == 1.:
                #     ax[1].errorbar(Sinclair_lat, Sinclair_c2h6, Sinclair_c2h6err, color='orange', lw=1, marker='v', markersize=5, 
                #             markerfacecolor='orange', markeredgewidth=2, alpha=1, capthick=2)
                ax[1].grid()
                ax[1].set_xlim(-70, 70)
                ax[1].tick_params(labelsize=18) 
                t = ax[1].yaxis.get_offset_text()
                t.set_x(-0.06)
                # third subplot with C2H4
                # ax[2].set_title(ititle[2]+r'      C$_{2}$H$_{4}$', size=18, fontfamily='sans-serif', loc='left')
                # ax[2].errorbar(latitude, gases[ind_pres[0][-1], :, 3], gases[ind_pres[0][-1], :, 3]*gas_errscalefit[1, :], color='grey', lw=1, marker='.', markersize=7, 
                #             markerfacecolor='black', markeredgewidth=0, alpha=0.7, capthick=0.4)
                # ax[2].set_yscale("log")
                # ax[2].set_xlim(-90, 90)
                # ax[2].grid()
                # ax[2].tick_params(labelsize=18)
                # t = ax[2].yaxis.get_offset_text()
                # t.set_x(-0.06)
                handles, labels = ax[0].get_legend_handles_labels()  
                fig.legend(handles, labels, bbox_to_anchor=(0.11,0.92,0.8,0.2), loc="lower left", mode="expand", fontsize=15)
                plt.axes([0.1, 0.1, 0.8, 0.8], frameon=False) 
                plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)            
                plt.xlabel(f"Planetographic Latitude", size=20)
                plt.ylabel(f"Volume Mixing Ratio", size=20)
                plt.savefig(f"{subdir}hydrocarbons_at_{ptarget[ipres]}mbar_comparison_with_previous_studies_latlim70.png", dpi=150, bbox_inches='tight')
                plt.close()
               
                 
            fig, ax = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=False)
            # first subplot with VMR fit
            ax[0][0].set_title(ititle[0]+r'      C$_{2}$H$_{2}$, Volume Mixing Ratio fit', size=18, fontfamily='sans-serif', loc='left')
            ax[0][0].errorbar(latitude, gas_vmrfit[0, :], yerr=gas_errvmrfit[0, :], color='grey', lw=1, marker='.', markersize=7, 
                        markerfacecolor='black', markeredgewidth=0, alpha=0.7, capthick=0.4, label=r'VLT/VISIR 2018 May 24$^{th}$-27$^{th}$')
            ax[0][0].plot(latitude, gas_vmr[0, :], color='blue', lw=3, label="prior parameters")
            ax[0][0].set_yscale("log")
            ax[0][0].set_xlim(-90, 90)
            ax[0][0].grid()
            ax[0][0].tick_params(labelsize=18)
            t = ax[0][0].yaxis.get_offset_text()
            t.set_x(-0.07)
            # second subplot with FSH fit
            ax[1][0].set_title(ititle[1]+r'      C$_{2}$H$_{2}$, Factional Scale Height fit', size=18, fontfamily='sans-serif', loc='left')
            ax[1][0].errorbar(latitude, gas_fshfit[0, :], yerr=gas_errfshfit[0, :], color='grey', lw=1, marker='.', markersize=7, 
                        markerfacecolor='black', markeredgewidth=0, alpha=0.7, capthick=0.4, label=r'VLT/VISIR 2018 May 24$^{th}$-27$^{th}$')
            ax[1][0].plot(latitude, gas_fsh[0, :], color='blue', lw=3, label="prior parameters")
            ax[1][0].grid()
            ax[1][0].set_xlim(-90, 90)
            ax[1][0].set_ylim(99.5, 100.5)
            ax[1][0].tick_params(labelsize=18) 
            t = ax[1][0].yaxis.get_offset_text()
            t.set_x(-0.06) 
            # first subplot with VMR fit
            ax[0][1].set_title(ititle[2]+r'      C$_{2}$H$_{6}$, Volume Mixing Ratio fit', size=18, fontfamily='sans-serif', loc='left')
            ax[0][1].errorbar(latitude, gas_vmrfit[2, :], yerr=gas_errvmrfit[2, :], color='grey', lw=1, marker='.', markersize=7, 
                        markerfacecolor='black', markeredgewidth=0, alpha=0.7, capthick=0.4, label=r'VLT/VISIR 2018 May 24$^{th}$-27$^{th}$')
            ax[0][1].plot(latitude, gas_vmr[2, :], color='blue', lw=3, label="prior parameters")
            # ax[0][1].set_yscale("log")
            ax[0][1].set_xlim(-90, 90)
            ax[0][1].grid()
            ax[0][1].tick_params(labelsize=18)
            t = ax[0][1].yaxis.get_offset_text()
            t.set_x(-0.08)
            # second subplot with FSH fit
            ax[1][1].set_title(ititle[3]+r'      C$_{2}$H$_{6}$, Factional Scale Height fit', size=18, fontfamily='sans-serif', loc='left')
            ax[1][1].errorbar(latitude, gas_fshfit[2, :], yerr=gas_errfshfit[2, :], color='grey', lw=1, marker='.', markersize=7, 
                        markerfacecolor='black', markeredgewidth=0, alpha=0.7, capthick=0.4, label=r'VLT/VISIR 2018 May 24$^{th}$-27$^{th}$')
            ax[1][1].plot(latitude, gas_fsh[2, :], color='blue', lw=3, label="prior parameters")
            ax[1][1].grid()
            ax[1][1].set_xlim(-90, 90)
            ax[1][1].tick_params(labelsize=18) 
            t = ax[1][1].yaxis.get_offset_text()
            t.set_x(-0.06) 
            handles, labels = ax[0][0].get_legend_handles_labels()  
            fig.legend(handles, labels, bbox_to_anchor=(0.11,0.92,0.8,0.2), loc="lower left", mode="expand", fontsize=15)
            plt.axes([0.09, 0.1, 0.8, 0.8], frameon=False) 
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)            
            plt.xlabel(f"Planetographic Latitude", size=20)
            plt.ylabel(f"Factional Scale Height     Volume Mixing Ratio", size=20)
            plt.savefig(f"{subdir}hydrocarbons_VMR-FSH_latitude_variations.png", dpi=150, bbox_inches='tight')
            plt.close()

####### Parametric plotting routines #######
def PlotRadianceParametricTest():

    print('Plotting NEMESIS retrieved radiance for parametric test...')
    # If subdirectory does not exist, create it
    dir = '../retrievals/retrieved_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Retrieval outputs directory path
    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/"
    # Array of prior file names
    prior = ['jupiter_v2021']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        retrieval_test =[f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H4-C2H6_NH3_cubic_S60_reduce",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H4-C2H6_NH3_cubic_S65_reduce",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H4-C2H6_NH3_cubic_S70_reduce",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H4-C2H6_NH3_cubic_S75_reduce",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H4-C2H6_NH3_cubic_S80_reduce",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H4-C2H6_NH3_cubic_S85_reduce"]
    ntest = len(retrieval_test)
    # If retrieval test subdirectory does not exist, create it
    subdir = f"{dir}Parametric_hydrocarbons_coeff/radiances/"
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    # Plotting figure
    fig, axes = plt.subplots(6, 1, figsize=(6, 10), sharex=True, sharey=True)
    # Loop over each retrieval tests for the current prior file
    for iplot, itest in enumerate(retrieval_test):
        # Read retrieved profiles from .mre outputs files
        radiance, wavenumb, rad_err, rad_fit, rad_diff, coeffs, ncores = ReadmreParametricTest(f"{fpath}{itest}/")
        # Figure settings
        ymin, ymax = 0, 15
        axes[iplot].set_ylim(ymin, ymax)
        axes[iplot].set_yticks(np.arange(ymin, ymax+1, 5))
        axes[iplot].tick_params(axis = 'y', labelsize=12)
        xmin, xmax = wavenumb[0, 0], wavenumb[-1, 0]
        axes[iplot].set_xlim(xmin, xmax)
        axes[iplot].set_xticks(np.asarray(wavenumb[:, 0], dtype=int))
        axes[iplot].tick_params(axis = 'x', rotation=60, labelsize=12)
        for icore in range(ncores):
            # Plot difference
            if icore == 0:
                axes[iplot].plot(wavenumb[:, 0], rad_diff[:, 0], alpha=0.01, color='black', label = f"{itest[-9:-7]}"+r'$^{\circ}$S') 
            axes[iplot].plot(wavenumb[:, 0], rad_diff[:, icore], alpha=0.01, color='black') 
        # Plot errors
        axes[iplot].grid(axis='y', markevery=1,  ls='--', lw=0.5, color='grey')
        # Plot filters
        axes[iplot].grid(axis='x', markevery=1,  ls='--', lw=0.5, color='grey')
        axes[iplot].legend(loc="upper left",  fontsize=15, handletextpad=0, handlelength=0, markerscale=0)
    plt.axes([0.1, 0.1, 0.8, 0.8], frameon=False) 
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel('Residual (%)', size=18)     
    plt.xlabel(r'Wavenumbers (cm$^{-1}$)', size=18, labelpad=15)
    plt.savefig(f"{subdir}residual_radiance_reduce_spxfile.png", dpi=150, bbox_inches='tight')
    plt.close()

def PlotComparisonParametricGasesHydrocarbons():


    test_names = [
            r'Temp Aer H$_p$',
            r'Temp Aer H$_p$ NH$_3$',
            r'Temp Aer H$_p$ NH$_{3,p}$', 
            r'Temp Aer C$_2$H$_6$',
            r'Temp Aer C$_2$H$_6$ ktable HighReso',
            r'Temp Aer C$_2$H$_{6,p}$',
            r'Temp Aer C$_2$H$_{6,p}$ ktable HighReso'

            ] 

    print('Plotting Comparison Gases and Hydrocarbons parametric retrieval tests...')
    # If subdirectory does not exist, create it
    dir = '../retrievals/retrieved_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Retrieval outputs directory path
    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/"
    # Array of prior file names
    prior = ['jupiter_v2021']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        
        retrieval_test = [
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H4-C2H6pknee02mbar_no852_no887",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H4-C2H6pknee02mbar_NH3_no852_no887",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H4-C2H6pknee02mbar_NH3p-kneefit-vmrJuno-fsh0.15-1.0_no852_no887",
             f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H6_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H6_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H6pknee02mbar_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H6pknee02mbar_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H6pknee02mbar_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H6pknee02mbar_ktable-highreso_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H4-C2H6pknee02mbar_ktable-highreso_no852_no887"
                        ]
        ntest = len(retrieval_test)
        # Loop over each prior used for retrievals tests
        for itest in retrieval_test:
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{iprior}_selected_meridien_tests/{itest}/ComparisonParametricGasesHydrocarbons/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)

            for ilat in [-0.5, -25.5, -60.5, -65.5 ,-70.5, -75.5, -80.5, -85.5]:
                # Setting figure grid of subplots
                fig = plt.figure(figsize=(10, 8))
                grid = plt.GridSpec(1, 4, wspace=0.5, hspace=0.6)
                c2h2_prf = fig.add_subplot(grid[0,0])
                c2h4_prf = fig.add_subplot(grid[0,1], sharey=c2h2_prf)
                c2h6_prf = fig.add_subplot(grid[0,2], sharey=c2h2_prf)
                nh3_prf  = fig.add_subplot(grid[0,3])
                # ph3_prf  = fig.add_subplot(grid[0,4], sharey=c2h2_prf)
                # Loop over each retrieval tests for the current prior file
                for i, itest in enumerate(retrieval_test):
                    
                    col = cmap(i/ntest)
                    # Read ChiSquare values from log files
                    chisquare, latitude, nlat = ReadLogFiles(filepath=f"{fpath}{itest}", over_axis="latitude")
                    # Read profile data from NEMESIS prior file 
                    _, prior_p, _, _, prior_gases, _, _, _ = ReadTemperatureGasesPriorProfile(f"{fpath}{itest}/core_1/")    
                    # Read retrieved profiles from .prf outputs files
                    _, gases, latitude, _, pressure, _, _, ngas, gases_id = ReadprfFiles(filepath=f"{fpath}{itest}", over_axis="latitude")
                    

                    ind_lat = np.where(latitude==ilat)
                    # print(ind_lat[0][0])
                    if i==0:
                        c2h2_prf.plot(prior_gases[:, 2], prior_p, color='blue', lw=2, label=f"prior profile at {latitude[ind_lat[0][0]]}")
                    c2h2_prf.plot(gases[:, ind_lat[0][0], 2], pressure, label=f"{test_names[i]}", color=col, lw=2)
                    gas_name, _ = RetrieveGasesNames(gas_name=False, gas_id=gases_id[2])
                    c2h2_prf.set_title(f"{gas_name}")
                    c2h2_prf.grid()
                    c2h2_prf.set_ylim(0.001, 1000)
                    c2h2_prf.set_yscale('log')
                    c2h2_prf.invert_yaxis()
                    c2h2_prf.set_xlim(1.e-11, 5.e-4)
                    c2h2_prf.set_xscale('log')
                    c2h2_prf.tick_params(labelsize=15)        
                
                    if i==0:
                        c2h4_prf.plot(prior_gases[:, 3], prior_p, color='blue', lw=2)
                    c2h4_prf.plot(gases[:, ind_lat[0][0], 3], pressure, color=col, lw=2)
                    gas_name, _ = RetrieveGasesNames(gas_name=False, gas_id=gases_id[3])
                    c2h4_prf.set_title(label=f"{gas_name}")
                    c2h4_prf.grid()
                    c2h4_prf.set_ylim(0.001, 1000)
                    c2h4_prf.set_yscale('log')
                    c2h4_prf.invert_yaxis()
                    c2h4_prf.set_xlim(1.e-12, 5.e-6)
                    c2h4_prf.set_xscale('log')
                    c2h4_prf.tick_params(labelsize=15)        
                
                    if i==0:
                        c2h6_prf.plot(prior_gases[:, 4], prior_p, color='blue', lw=2)
                    c2h6_prf.plot(gases[:, ind_lat[0][0], 4], pressure, color=col, lw=2)
                    gas_name, _ = RetrieveGasesNames(gas_name=False, gas_id=gases_id[4])
                    c2h6_prf.set_title(f"{gas_name}")
                    c2h6_prf.grid()
                    c2h6_prf.set_ylim(0.001, 1000)
                    c2h6_prf.set_yscale('log')
                    c2h6_prf.invert_yaxis()
                    c2h6_prf.set_xlim(2.e-8, 5.e-5)
                    c2h6_prf.set_xscale('log')
                    c2h6_prf.tick_params(labelsize=15)        
                
                    if i==0: 
                        nh3_prf.plot(prior_gases[:, 0], prior_p, color='blue', lw=2)
                    nh3_prf.plot(gases[:, ind_lat[0][0], 0], pressure, color=col, lw=2)
                    gas_name, _ = RetrieveGasesNames(gas_name=False, gas_id=gases_id[0])
                    nh3_prf.set_title(f"{gas_name}")
                    nh3_prf.grid()
                    nh3_prf.set_ylim(100, 1000)
                    nh3_prf.set_yscale('log')
                    nh3_prf.invert_yaxis()
                    # nh3_prf.set_xscale('log')
                    nh3_prf.tick_params(labelsize=15)        
                
                    
                    # ph3_prf.plot(prior_gases[:, 1], prior_p, color='blue')
                    # ph3_prf.plot(gases[:, ind_lat[0][0], 1], pressure, color=col)
                    # gas_name, _ = RetrieveGasesNames(gas_name=False, gas_id=gases_id[1])
                    # ph3_prf.set_title(f"{gas_name}")
                    # ph3_prf.grid()
                    # ph3_prf.set_ylim(0.01, 1000)
                    # ph3_prf.set_yscale('log')
                    # ph3_prf.invert_yaxis()
                    # ph3_prf.tick_params(labelsize=15)        
                handles, labels = c2h2_prf.get_legend_handles_labels()  
                
                plt.axes([0.1, 0.08, 0.8, 0.85], frameon=False) 
                plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                plt.xlabel(f"Volume Mixing Ratio", size=20)
                plt.ylabel(f"Presssure [mbar]", size=20)
                plt.legend(handles, labels, bbox_to_anchor=(1.04, 1), loc="upper left",fontsize=12)
                plt.savefig(f"{subdir}parametric_test_gases_hydrocarbons_profiles_lat_{latitude[ind_lat[0][0]]}_reduce_cmaps_data.png", dpi=150, bbox_inches='tight')
                plt.close()

            # Setting figure grid of subplots
            fig = plt.figure(figsize=(6, 8))
            grid = plt.GridSpec(4, 4, wspace=0.5, hspace=1.2)
            
            # chiqsq     = fig.add_subplot(grid[0, :])
            merid_c2h2 = fig.add_subplot(grid[0, :])
            merid_c2h4 = fig.add_subplot(grid[1, :], sharex=merid_c2h2)
            merid_c2h6 = fig.add_subplot(grid[2, :], sharex=merid_c2h2)
            merid_nh3  = fig.add_subplot(grid[3, :], sharex=merid_c2h2)
            # merid_nh3_2 = fig.add_subplot(grid[5, :], sharex=merid_c2h2)



            # Loop over each retrieval tests for the current prior file
            for i, itest in enumerate(retrieval_test):
                col = cmap(i/ntest)
                # Read ChiSquare values from log files
                # chisquare, latitude, nlat = ReadLogFiles(filepath=f"{fpath}{itest}", over_axis="latitude")
                # Read retrieved profiles from .prf outputs files
                _, gases, latitude, _, pressure, _, _, ngas, gases_id = ReadprfFiles(filepath=f"{fpath}{itest}", over_axis="latitude")

                # chiqsq.plot(latitude, chisquare, label=f"{test_names[i]}", color=col)
                # chiqsq.set_ylim(0, 1.5)
                # chiqsq.grid()
                # chiqsq.set_ylabel('\u03C7'r'$^{2}/N_y$', size=15)
                # chiqsq.tick_params(labelsize=12)     

                ind_pres = np.where(pressure >= 5.)
                merid_c2h2.set_title(r'C$_{2}$H$_{2}$'+f" at {int(pressure[ind_pres[0][-1]])} mbar")
                merid_c2h2.plot(latitude, gases[ind_pres[0][-1], :, 2], color=col, label=f"{test_names[i]}")
                merid_c2h2.tick_params(labelsize=12)

                merid_c2h4.set_title(r'C$_{2}$H$_{4}$'+f" at {int(pressure[ind_pres[0][-1]])} mbar")
                merid_c2h4.plot(latitude, gases[ind_pres[0][-1], :, 3], color=col)
                merid_c2h4.tick_params(labelsize=12)

                merid_c2h6.set_title(r'C$_{2}$H$_{6}$'+f" at {int(pressure[ind_pres[0][-1]])} mbar")
                merid_c2h6.plot(latitude, gases[ind_pres[0][-1], :, 4], color=col)
                merid_c2h6.tick_params(labelsize=12)

                ind_pres = np.where(pressure >= 500)
                merid_nh3.set_title(r'NH$_{3}$'+f" at {int(pressure[ind_pres[0][-1]])} mbar")
                merid_nh3.plot(latitude, gases[ind_pres[0][-1], :, 0], color=col)
                merid_nh3.tick_params(labelsize=12)

                # ind_pres = np.where(pressure >= 800)
                # merid_nh3_2.set_title(r'NH$_{3}$'+f" at {int(pressure[ind_pres[0][-1]])} mbar")
                # merid_nh3_2.plot(latitude, gases[ind_pres[0][-1], :, 0], color=col)
                # merid_nh3_2.tick_params(labelsize=12)        

            handles, labels = merid_c2h2.get_legend_handles_labels()  
            
            plt.axes([0.12, 0.1, 0.8, 0.65], frameon=False) 
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel(f"Planetographic Latitude", size=20)
            plt.ylabel(f"Volume Mixing Ratio", size=20)
            plt.legend(handles, labels, bbox_to_anchor=(1.04, 1), loc="upper left",fontsize=12)
            # Save figure in the retrievals outputs directory
            plt.savefig(f"{subdir}parametric_test_gases_hydrocarbons.png", dpi=150, bbox_inches='tight')
            plt.close()

def PlotComparisonParametricGasesHydrocarbonsParallel():

    # If subdirectory does not exist, create it
    dir = '../retrievals/retrieved_figures/ComparisonParametricGasesHydrocarbons_lat80S/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Retrieval outputs directory path
    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/"
    # Array of prior file names
    prior = ['jupiter_v2021']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        retrieval_test = [f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_NH3pt_PH3pt_lat80S_no852_no887_reduce", 
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_NH3pt_PH3_lat80S_no852_no887_reduce", 
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_NH3_PH3pt_lat80S_no852_no887_reduce", 
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_NH3pt_lat80S_no852_no887_reduce", 
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_NH3_lat80S_no852_no887_reduce", 
                        ]
        ntest = len(retrieval_test)

        

        for ilon in [0.5, 50.5 ,100.5, 150.5, 200.5, 250.5, 300.5, 350.5]:
            # Setting figure grid of subplots
            fig = plt.figure(figsize=(12, 12))
            grid = plt.GridSpec(1, 5, wspace=0.5, hspace=0.6)
            sixtysouth_prf       = fig.add_subplot(grid[0,0])
            sixtyfivesouth_prf   = fig.add_subplot(grid[0,1], sharey=sixtysouth_prf)
            seventysouth_prf     = fig.add_subplot(grid[0,2], sharey=sixtysouth_prf)
            seventyfivesouth_prf = fig.add_subplot(grid[0,3], sharey=sixtysouth_prf)
            eightysouth_prf      = fig.add_subplot(grid[0,4], sharey=sixtysouth_prf)
            # chiqsq     = fig.add_subplot(grid[0, :])
            # merid_c2h2 = fig.add_subplot(grid[1, :])
            # merid_c2h4 = fig.add_subplot(grid[2, :], sharex=merid_c2h2)
            # merid_c2h6 = fig.add_subplot(grid[3, :], sharex=merid_c2h2)
            # merid_nh3  = fig.add_subplot(grid[4, :], sharex=merid_c2h2)
            # merid_ph3  = fig.add_subplot(grid[5, :], sharex=merid_c2h2)
            # Loop over each retrieval tests for the current prior file
            for i, itest in enumerate(retrieval_test):
                col = cmap(i/ntest)
                # Read ChiSquare values from log files
                chisquare, longitude, nlat = ReadLogFiles(filepath=f"{fpath}{itest}", over_axis="longitude")
                # Read retrieved profiles from .prf outputs files
                _, gases, longitude, _, pressure, _, _, ngas, gases_id = ReadprfFiles(filepath=f"{fpath}{itest}", over_axis="longitude")
                
                
                ind_lat = np.where(longitude==ilon)
                print(ind_lat[0][0])
                sixtysouth_prf.loglog(gases[:, ind_lat[0][0], 2], pressure, label=f"{itest}"[52:], color=col)
                gas_name, _ = RetrieveGasesNames(gas_name=False, gas_id=gases_id[2])
                sixtysouth_prf.set_title(f"{gas_name}")
                
                sixtyfivesouth_prf.loglog(gases[:, ind_lat[0][0], 3], pressure, color=col)
                gas_name, _ = RetrieveGasesNames(gas_name=False, gas_id=gases_id[3])
                sixtyfivesouth_prf.set_title(label=f"{gas_name}")
                
                seventysouth_prf.loglog(gases[:, ind_lat[0][0], 4], pressure, color=col)
                gas_name, _ = RetrieveGasesNames(gas_name=False, gas_id=gases_id[4])
                seventysouth_prf.set_title(f"{gas_name}")
                
                seventyfivesouth_prf.loglog(gases[:, ind_lat[0][0], 0], pressure, color=col)
                gas_name, _ = RetrieveGasesNames(gas_name=False, gas_id=gases_id[0])
                seventyfivesouth_prf.set_title(f"{gas_name}")
                
                eightysouth_prf.loglog(gases[:, ind_lat[0][0], 1], pressure, color=col)
                gas_name, _ = RetrieveGasesNames(gas_name=False, gas_id=gases_id[1])
                eightysouth_prf.set_title(f"{gas_name}")
            handles, labels = sixtysouth_prf.get_legend_handles_labels()  
            fig.legend(handles, labels, loc='upper right')
            plt.savefig(f"{dir}parametric_test_gases_hydrocarbons_profiles_lon_{longitude[ind_lat[0][0]]}_reduce_cmaps_data.png", dpi=150, bbox_inches='tight')
            plt.close()


        # Setting figure grid of subplots
        fig = plt.figure(figsize=(8, 12))
        grid = plt.GridSpec(6, 5, wspace=0.5, hspace=0.6)
        
        chiqsq     = fig.add_subplot(grid[0, :])
        merid_c2h2 = fig.add_subplot(grid[1, :])
        merid_c2h4 = fig.add_subplot(grid[2, :], sharex=merid_c2h2)
        merid_c2h6 = fig.add_subplot(grid[3, :], sharex=merid_c2h2)
        merid_nh3  = fig.add_subplot(grid[4, :], sharex=merid_c2h2)
        merid_ph3  = fig.add_subplot(grid[5, :], sharex=merid_c2h2)

        # Loop over each retrieval tests for the current prior file
        for i, itest in enumerate(retrieval_test):
            col = cmap(i/ntest)
            # Read ChiSquare values from log files
            chisquare, longitude, nlat = ReadLogFiles(filepath=f"{fpath}{itest}", over_axis="longitude")
            # Read retrieved profiles from .prf outputs files
            _, gases, longitude, _, pressure, _, _, ngas, gases_id = ReadprfFiles(filepath=f"{fpath}{itest}", over_axis="longitude")
            chiqsq.plot(longitude, chisquare, label=f"{itest}"[52:], color=col)
            chiqsq.set_ylim(0, 3)
            chiqsq.grid()
            chiqsq.set_ylabel('\u03C7'r'$^{2}/N_y$', size=15)
            chiqsq.tick_params(labelsize=12)     

            ind_pres = np.where(pressure >= 5.)
            merid_c2h2.set_title(r'C$_{2}$H$_{2}$'+f" at {int(pressure[ind_pres[0][-1]])} mbar")
            merid_c2h2.plot(longitude, gases[ind_pres[0][-1], :, 2], color=col)
            merid_c2h2.tick_params(labelsize=12)

            merid_c2h4.set_title(r'C$_{2}$H$_{4}$'+f" at {int(pressure[ind_pres[0][-1]])} mbar")
            merid_c2h4.plot(longitude, gases[ind_pres[0][-1], :, 3], color=col)
            merid_c2h4.tick_params(labelsize=12)

            merid_c2h6.set_title(r'C$_{2}$H$_{6}$'+f" at {int(pressure[ind_pres[0][-1]])} mbar")
            merid_c2h6.plot(longitude, gases[ind_pres[0][-1], :, 4], color=col)
            merid_c2h6.tick_params(labelsize=12)

            ind_pres = np.where(pressure >= 800)
            merid_nh3.set_title(r'NH$_{3}$'+f" at {int(pressure[ind_pres[0][-1]])} mbar")
            merid_nh3.plot(longitude, gases[ind_pres[0][-1], :, 0], color=col)
            merid_nh3.tick_params(labelsize=12)

            merid_ph3.set_title(r'PH$_{3}$'+f" at {int(pressure[ind_pres[0][-1]])} mbar")
            merid_ph3.plot(longitude, gases[ind_pres[0][-1], :, 1], color=col)
            merid_ph3.set_xlabel("System III West Longitude", size=15)
            merid_ph3.tick_params(labelsize=12)        

        handles, labels = chiqsq.get_legend_handles_labels()  
        fig.legend(handles, labels, loc='upper center') 
        # Save figure in the retrievals outputs directory
        plt.savefig(f"{dir}parametric_test_gases_hydrocarbons_profiles_reduce_cmaps_data.png", dpi=150, bbox_inches='tight')
        plt.close()






####### Aurora plotting and mapping routine #######
def PlotAllForAuroraOverTime():

    print("Plot aurora retrieval results...")
     # If subdirectory does not exist, create it
    dir = '../retrievals/retrieved_figures/aurora_over_time/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Retrieval outputs directory path
    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/retrieval_averaged_region_southern_aurora/"
    # Array of prior file names
    prior = ['jupiter_v2021']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        retrieval_test = ["jupiter_vzonal_lat-80_temp_aurora_RegAv_no852_no887", 
                        "jupiter_v2023_from_zonal_c2h2p_c2h6p_temp_aerosol1-10mu-800mbar-05scale-01_aurora_RegAv_no852_no887",
                        "jupiter_v2023_from_zonal_c2h2p_c2h6p_temp_aerosol1-10mu-800mbar-05scale-01_NH3_aurora_RegAv_no852_no887",
                        "jupiter_v2023_from_zonal_c2h2p_c2h6p_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H6_aurora_RegAv_no852_no887",
                        "jupiter_v2023_from_zonal_c2h2p_c2h6p_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H4-C2H6_aurora_RegAv_no852_no887"
                        ]
        ntest = len(retrieval_test)
        night_labels = [r'May 24$^{th}$', r'May 25$^{th}$-26$^{th}$', r'May 26$^{th}$-27$^{th}$']
        # Colormap definition
        cmap = plt.get_cmap("magma") #magma     
        for itest in retrieval_test:
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{itest}/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            nnight, time, chisquare, radiance, rad_err, rad_fit, wavenumb, \
                temp_prior_mre, temp_errprior_mre, temp_fit_mre, temp_errfit_mre, \
                aer_mre, aer_err,  aer_fit, fit_err, \
                gas_scale, gas_scaleerr,  gas_scalefit, gas_errscalefit,\
                height, pressure, temperature, \
                gases, gases_id, aer_prf, h_prf = ReadAllForAuroraOverTime(filepath=f"{fpath}{itest}", gas_name=['C2H2','C2H6'])

            # Plot Figure of chisq/ny over latitude
            fig = plt.subplots(1, 1, figsize=(12, 6))
            maxchi = np.nanmax(chisquare)
            plt.plot(time, chisquare, lw=2)
            plt.plot(time, chisquare, lw=0, marker='*', markersize=4)
            plt.grid()
            if maxchi > 1:
                plt.ylim(0, ceil(maxchi))
                plt.yticks(np.arange(ceil(maxchi)+1))
            else:
                plt.ylim(0, 1)
                plt.yticks(np.arange(0, 1.01, 0.1))
            plt.tick_params(axis = 'y', labelsize=15)
            plt.tick_params(axis = 'x', labelsize=15)        
            plt.ylabel('\u03C7'r'$^{2}/N_y$', size=20)
            plt.xlabel("Nights", size=20)
            plt.xticks(ticks=np.arange(1,4,step=1), labels=night_labels)
            # Save figure in the retrievals outputs directory
            plt.savefig(f"{subdir}aurora_over_time_chisquare.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{subdir}{itest}_chisquare.eps", dpi=100)
            # Close figure to avoid overlapping between plotting subroutines
            plt.close()

        
            fig, axes = plt.subplots(1, 1, figsize=(8, 10))
            # Read profile data from NEMESIS prior file 
            _, prior_p, prior_temperature, prior_err, _, _, _, _ = ReadTemperatureGasesPriorProfile(f"{fpath}{itest}/core_1/")
            # Plot the prior only for the last itest (because it's the same for all itest)
            axes.plot(prior_temperature, prior_p, lw=2, label=f"prior", color='green')
            axes.fill_betweenx(prior_p, prior_temperature-prior_err, prior_temperature+prior_err, color='green', alpha=0.2)
            for inight in range(nnight):
                col = cmap(inight/nnight)
                # Plot 
                axes.plot(temperature[:, inight], pressure, lw=2, label=f"night {night_labels[inight]}", color = col)
            axes.set_yscale('log')
            axes.set_ylim(0.001, 10000)
            axes.invert_yaxis()
            axes.grid()
            axes.legend(loc="center right", fontsize=15)
            axes.tick_params(labelsize=15)
            # Add a big axis 
            plt.axes([0.1, 0.09, 0.8, 0.8], frameon=False) 
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)        
            plt.xlabel(f"Temperature [K]", size=20)
            plt.ylabel(f"Presssure [mbar]", size=20)
            # Save figure in the retrievals outputs directory
            plt.savefig(f"{subdir}aurora_over_time_temperature_profile_allnight.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{subdir}{itest}_retrieved_temperature_profile_at_{latitude[ilat]}.eps", dpi=100)
            # Close figure to avoid overlapping between plotting subroutines
            plt.close()

            # Setting figure grid of subplots
            fig = plt.figure(figsize=(8, 8))
            gs = fig.add_gridspec(1, 2, width_ratios=(4, 1),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.01, hspace=0.05)
            temp_prf = fig.add_subplot(gs[0,0])
            res_prf = fig.add_subplot(gs[0,1])
            
            # Read profile data from NEMESIS prior file 
            # Plot the prior only for the last itest (because it's the same for all itest)
            temp_prf.plot(temp_prior_mre[:, inight], prior_p, lw=2, label=f"prior", color='green')
            temp_prf.fill_betweenx(prior_p, temp_prior_mre[:, inight]-temp_errprior_mre[:, inight], temp_prior_mre[:, inight]+temp_errprior_mre[:, inight], color='green', alpha=0.1)
            for inight in range(nnight):
                col = cmap(inight/nnight)
                # temperature prior and retrieved profiles 
                temp_prf.plot(temp_fit_mre[:, inight], pressure, lw=2, label=f"night {night_labels[inight]}", color = col)
                temp_prf.fill_betweenx(pressure, temp_fit_mre[:, inight]-temp_errfit_mre[:, inight], temp_fit_mre[:, inight]+temp_errfit_mre[:, inight], color=col, alpha=0.2)
                # residual temperature
                res_prf.plot(temp_fit_mre[:, inight]-temp_prior_mre[:, inight], pressure, lw=2, label=f"night {night_labels[inight]}", color = col)
            temp_prf.set_yscale('log')
            temp_prf.set_ylim(0.001, 10000)
            temp_prf.invert_yaxis()
            temp_prf.set_title("(a)", size=18, fontfamily='sans-serif', loc='left')
            temp_prf.grid()
            temp_prf.legend(loc="center right", fontsize=15)
            temp_prf.tick_params(labelsize=16)
            res_prf.set_title("(b)", size=18, fontfamily='sans-serif', loc='left')
            res_prf.set_yscale('log')
            res_prf.set_ylim(0.001, 10000)
            res_prf.invert_yaxis()
            res_prf.grid()
            res_prf.tick_params(labelsize=16)
            res_prf.set_yticklabels([])
            # Add a big axis 
            plt.axes([0.08, 0.09, 0.8, 0.8], frameon=False) 
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)        
            plt.xlabel(f"Temperature [K]", size=20)
            plt.ylabel(f"Presssure [mbar]", size=20)
            # Save figure in the retrievals outputs directory
            plt.savefig(f"{subdir}aurora_over_time_temperature_profile_allnight_from-mre.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{subdir}{itest}_retrieved_temperature_profile_at_{latitude[ilat]}.eps", dpi=100)
            # Close figure to avoid overlapping between plotting subroutines
            plt.close()


            # Setting figure grid of subplots
            fig = plt.figure(figsize=(8, 10))
            grid = plt.GridSpec(1, 3, wspace=0.5, hspace=0.6)
            c2h2_prf = fig.add_subplot(grid[0,0])
            c2h4_prf = fig.add_subplot(grid[0,1], sharey=c2h2_prf)
            c2h6_prf = fig.add_subplot(grid[0,2], sharey=c2h2_prf)
            # nh3_prf  = fig.add_subplot(grid[0,3], sharey=c2h2_prf)
            for inight in range(nnight):
                # Define the color for each night
                col = cmap(inight/nnight)
                # Read profile data from NEMESIS prior file 
                _, prior_p, _, _, prior_gases, _, _, _ = ReadTemperatureGasesPriorProfile(f"{fpath}{itest}/core_1/")    
                
                if inight==0:
                    c2h2_prf.plot(prior_gases[:, 2], prior_p, color='green', lw=2, label=f"prior profile")
                c2h2_prf.plot(gases[:, inight, 2], pressure, label=f"night {night_labels[inight]}", color=col, lw=2)
                gas_name, _ = RetrieveGasesNames(gas_name=False, gas_id=gases_id[2])
                c2h2_prf.set_title(f"{gas_name}")
                c2h2_prf.grid()
                c2h2_prf.set_xscale('log')
                c2h2_prf.set_ylim(0.001, 1000)
                c2h2_prf.set_yscale('log')
                c2h2_prf.invert_yaxis()
                c2h2_prf.tick_params(labelsize=15)        
            
                if inight==0:
                    c2h4_prf.plot(prior_gases[:, 3], prior_p, color='green', lw=2)
                c2h4_prf.plot(gases[:, inight, 3], pressure, color=col, lw=2)
                gas_name, _ = RetrieveGasesNames(gas_name=False, gas_id=gases_id[3])
                c2h4_prf.set_title(label=f"{gas_name}")
                c2h4_prf.grid()
                c2h4_prf.set_xscale('log')
                c2h4_prf.set_ylim(0.001, 1000)
                c2h4_prf.set_yscale('log')
                c2h4_prf.invert_yaxis()
                c2h4_prf.tick_params(labelsize=15)        
            
                if inight==0:
                    c2h6_prf.plot(prior_gases[:, 4], prior_p, color='green', lw=2)
                c2h6_prf.plot(gases[:, inight, 4], pressure, color=col, lw=2)
                gas_name, _ = RetrieveGasesNames(gas_name=False, gas_id=gases_id[4])
                c2h6_prf.set_title(f"{gas_name}")
                c2h6_prf.grid()
                c2h6_prf.set_xscale('log')
                c2h6_prf.set_ylim(0.001, 1000)
                c2h6_prf.set_yscale('log')
                c2h6_prf.invert_yaxis()
                c2h6_prf.tick_params(labelsize=15)        
            
                # if inight==0: 
                #     nh3_prf.plot(prior_gases[:, 0], prior_p, color='green', lw=2)
                # nh3_prf.plot(gases[:, inight, 0], pressure, color=col, lw=2)
                # gas_name, _ = RetrieveGasesNames(gas_name=False, gas_id=gases_id[0])
                # nh3_prf.set_title(f"{gas_name}")
                # nh3_prf.grid()
                # nh3_prf.set_ylim(0.001, 1000)
                # nh3_prf.set_yscale('log')
                # nh3_prf.invert_yaxis()
                # nh3_prf.tick_params(labelsize=15)        
            
                
                # ph3_prf.plot(prior_gases[:, 1], prior_p, color='green')
                # ph3_prf.plot(gases[:, inight, 1], pressure, color=col)
                # gas_name, _ = RetrieveGasesNames(gas_name=False, gas_id=gases_id[1])
                # ph3_prf.set_title(f"{gas_name}")
                # ph3_prf.grid()
                # ph3_prf.set_ylim(0.01, 1000)
                # ph3_prf.set_yscale('log')
                # ph3_prf.invert_yaxis()
                # ph3_prf.tick_params(labelsize=15)        
            # handles, labels = c2h2_prf.get_legend_handles_labels()  
            # fig.legend(handles, labels, loc='upper right',fontsize=12, ncol=2)
            plt.axes([0.1, 0.08, 0.8, 0.85], frameon=False) 
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel(f"Volume Mixing Ratio", size=20)
            plt.ylabel(f"Presssure [mbar]", size=20)
            plt.savefig(f"{subdir}aurora_over_time_gases_hydrocarbons_profiles.png", dpi=150, bbox_inches='tight')
            plt.close()

            # # Mapping the hydrocarbons scale factor from .mre file
            fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharex=True, sharey=False)
            ax[0].plot(time, gas_scalefit[0, :], lw=2, color="darkmagenta")
            # ax[0].fill_between(time, gas_scalefit[0, :]-gas_errscalefit[0, :], gas_scalefit[0, :]+gas_errscalefit[0, :], color="darkmagenta", alpha=0.2)
            ax[0].set_title("(a) "+r'C$_2$H$_2$', fontfamily='sans-serif', loc='left', fontsize=12)
            ax[0].grid()
            ax[0].tick_params(labelsize=15)
            ax[1].plot(time, gas_scalefit[1, :], lw=2, color="darkblue")
            # ax[1].fill_between(time, gas_scalefit[1, :]-gas_errscalefit[1, :], gas_scalefit[1, :]+gas_errscalefit[1, :], color="darkblue", alpha=0.2)
            ax[1].set_title("(b) "+r'C$_2$H$_6$', fontfamily='sans-serif', loc='left', fontsize=12)
            ax[1].grid()
            ax[1].tick_params(labelsize=15)
            ax[1].set_xticks(ticks=np.arange(1,4,step=1), labels=night_labels)
            plt.axes([0.1, 0.08, 0.8, 0.8], frameon=False) 
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel("Time", size=20)
            plt.ylabel(f"Hydrocarbons scale factor", size=20)
            # Save figure in the retrievals outputs directory
            plt.savefig(f"{subdir}aurora_over_time_hydrocarbons_scale_factor_from-mre.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{subdir}{itest}_hydrocarbons_scale_factor_from-mre.eps", dpi=100)
            # Close figure to avoid overlapping between plotting subroutines
            plt.close()

            # # Mapping the aerosol scale factor from .mre file
            plt.figure(figsize=(8, 4))
            plt.plot(time, aer_fit, lw=2)
            plt.grid()
            plt.tick_params(labelsize=15)        
            plt.xlabel("Time", size=20)
            plt.xticks(ticks=np.arange(1,4,step=1), labels=night_labels)
            plt.ylabel(f"Aerosol scale factor", size=20)
            # Save figure in the retrievals outputs directory
            plt.savefig(f"{subdir}aurora_over_time_aerosol_scale_factor_from-mre.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{subdir}{itest}_aerosol_scale_factor_from-mre.eps", dpi=100)
            # Close figure to avoid overlapping between plotting subroutines
            plt.close()

            prior_aer, prior_alt, prior_ncloud, prior_nlevel = ReadAerosolPriorProfile(f"{fpath}{itest}/core_1/aerosol.ref")
            fig= plt.figure(figsize=(8, 10))
            for inight in range(nnight):
                # Define the color for each night
                col = cmap(inight/nnight)
                if inight==0:
                    plt.plot(prior_aer, prior_alt, lw=2, label=f"aerosol apriori", color='blue')
                plt.plot(aer_prf[:, inight], h_prf[:, inight], lw=2, label=f"night {night_labels[inight]}", color=col)
            plt.legend(loc="upper right", fontsize=15)
            plt.tick_params(labelsize=15)
            plt.grid()
            # Add a big axis 
            # plt.axes([0.1, 0.09, 0.8, 0.8], frameon=False) 
            # plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)        
            plt.xlabel(f"Aerosol []", size=20)
            plt.ylabel(f"Height [km]", size=20)
            # Save figure in the retrievals outputs directory
            plt.savefig(f"{subdir}aurora_over_time_retrieved_aerosol_profile.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{subdir}{itest}_retrieved_aerosol_profile_at_{latitude[ilat]}.eps", dpi=100)
            # Close figure to avoid overlapping between plotting subroutines
            plt.close()

            iplot=0
            for ifilt in range(13):
                if ifilt<6 or ifilt>7:
                    _, wavl, wave, _, _ = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=ifilt)

                    plt.figure(figsize=(8, 4))
                    plt.plot(time, radiance[iplot, :], label=f"{wavl}"+'$\mu$m', lw=2)
                    plt.plot(time, radiance[iplot, :], label=f"{wavl}"+'$\mu$m', lw=0, marker='*', markersize=4)
                    plt.grid()
                    plt.tick_params(labelsize=15)        
                    plt.xlabel("Time", size=20)
                    plt.xticks(ticks=np.arange(1,4,step=1), labels=night_labels)
                    plt.ylabel(f"Radiance ", size=20)
                    # Save figure in the retrievals outputs directory
                    plt.legend()
                    plt.savefig(f"{subdir}{wave}_aurora_over_time_radiance_time_evoltion.png", dpi=150, bbox_inches='tight')
                    #plt.savefig(f"{subdir}{itest}_radiance_time_evoltion.eps", dpi=100)
                    # Close figure to avoid overlapping between plotting subroutines
                    plt.close() 
                    iplot+=1

            fig, ax = plt.subplots(5, 2, figsize=(8, 10), sharex=True, sharey=False)
            iax = 0
            for ifilt in [0,10,5,4,6,7,3,2,1]:
                col = cmap(iplot/11)
                irow = [0,1,1,2,2,3,3,4,4]
                icol = [0,0,1,0,1,0,1,0,1]
                ititle = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(i)', '(j)', '(k)']
                # Remove the frame of the empty subplot
                ax[0][1].set_frame_on(False)
                ax[0][1].tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                if ifilt>5 :
                    _, wavl, wave, _, _ = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=ifilt+2)
                else:
                    _, wavl, wave, _, _ = SetWave(filename=None, wavelength=None, wavenumber=None, ifilt=ifilt)
                            
                ax[irow[iax]][icol[iax]].plot(time, radiance[ifilt, :], lw=2, color='green', label="Observed radiance")
                ax[irow[iax]][icol[iax]].fill_between(time, radiance[ifilt, :]-rad_err[ifilt, :], radiance[ifilt, :]+rad_err[ifilt, :], color='green', alpha=0.2)
                ax[irow[iax]][icol[iax]].plot(time, radiance[ifilt, :], lw=0, marker='*', markersize=4, color='green')
                ax[irow[iax]][icol[iax]].plot(time, rad_fit[ifilt, :], lw=2, color='black', label="Retrieved radiance")
                ax[irow[iax]][icol[iax]].tick_params(axis='y', labelsize=14)
                ax[irow[iax]][icol[iax]].tick_params(axis = 'x', labelsize=12)#, rotation=60)
                ax[irow[iax]][icol[iax]].set_xticks(ticks=np.arange(1,4,step=1), labels=['May\n24$^{th}$', 'May\n25$^{th}$-26$^{th}$', 'May\n26$^{th}$-27$^{th}$'])
                ax[irow[iax]][icol[iax]].grid()
                ax[irow[iax]][icol[iax]].set_title(ititle[iax]+f"    {(wavl)}"+" $\mu$m", fontfamily='sans-serif', loc='left', fontsize=12)
                iax+=1 
            handles, labels = ax[0][0].get_legend_handles_labels()  
            fig.legend(handles, labels,fontsize=12, bbox_to_anchor=[0.875, 0.85])
            plt.axes([0.1, 0.08, 0.8, 0.8], frameon=False) 
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel("Time", size=20)
            
            plt.ylabel('Radiance [nW cm$^{-1}$ sr$^{-1}$]', size=20)
            # Save figure in the retrievals outputs directory
            plt.savefig(f"{subdir}aurora_over_time_all_filters_radiance_time_evoltion.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{subdir}{itest}_radiance_time_evoltion.eps", dpi=100)
            # Close figure to avoid overlapping between plotting subroutines
            plt.close() 
            

def PlotSolarWindActivity():

    print("Plot time evolution of the solar wind activity for the 2018 year ... ")
    # If subdirectory does not exist, create it
    dir = '../retrievals/retrieved_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Retrieval outputs directory path
    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/"
    # Array of prior file names
    prior = ['jupiter_v2021']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        retrieval_test = [
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee005mbar-vmrerr0.8-fsh100.0step10.0-C2H4-C2H6pknee02mbar_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H6_no852_no887",
            f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H6pknee02mbar_no852_no887"
                        ]
        ntest = len(retrieval_test)
        
        # Loop over each retrieval tests for the current prior file
        for i, itest in enumerate(retrieval_test):
            col = cmap(i/ntest)
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{iprior}_selected_meridien_tests/{itest}/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)

    year, month, day, _, _, _, SW_dyna_press, heliocentric_long = ReadSolarWindPredi(filepath="/Users/db496/Documents/Research/Observations/archives/Sinclair_2017_Icarus_Jupiter_auroral_related_stratospheric_heating_chemistry_part1/swmhdjup2018Abc1129.txt")            
    
    time_index = np.where((month == 5.0) & (day > 19.0) & (day < 28.0))
    
    obs_day = []
    for i in range(len(year)):
        obs_day.append(f"{int(year[i])} {int(month[i])} {int(day[i])}")
    obs_day = np.reshape(obs_day, (len(year)))
    obs_day = obs_day[time_index]
    obsdate = []
    [obsdate.append(item) for item in obs_day if item not in obsdate]
    
    # setlimit values
    # Plot day evolution of the solar wind activity
    fig, ax = plt.subplots(1, 1, figsize=(6, 5), sharex=True)
    # solar wind dynamical pressure
    SolarWind = SW_dyna_press[time_index]
    ax.plot(SolarWind, color='crimson', lw=2)
    ax.grid()
    ax.set_xlim(obs_day[0], obs_day[-1])
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("Solar wind \n dynamical pressure \n[nPa]", size=20)
    ax.tick_params(labelsize=20)
    # # heliocentric longitude from opposition
    # HeliocentricLong = heliocentric_long[time_index]
    # ax[1].plot(HeliocentricLong, color='crimson', lw=2)
    # ax[1].grid()
    # ax[1].set_xlim(obs_day[0], obs_day[-1])
    # ax[1].set_ylim(-134, -124)
    # ax[1].set_ylabel("Heliocentric longitude \n from opposition", size=20)
    # ax[1].tick_params(labelsize=20)
    plt.xticks(ticks=np.arange(0,len(obs_day), step=24), labels=list(obsdate), rotation=45, ha="right")
    # plt.axes([0.1, 0.0001, 0.8, 0.8], frameon=False) 
    # plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)            
    plt.xlabel(f"Time ", size=20)
    plt.savefig(f"{subdir}solar_wind_activity_time_variations.png", dpi=150, bbox_inches='tight')
    plt.close()