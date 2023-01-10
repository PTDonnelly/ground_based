import os
import re
from sqlite3 import Time
from turtle import color
import numpy as np
from scipy.special import xlogy
from matplotlib.ticker import LogFormatter, LogFormatterMathtext, LogLocator
from matplotlib import colors
from math import *
import matplotlib.pyplot as plt
import Globals
from Tools.SetWave import SetWave
from Read.ReadPrior import ReadTemperatureGasesPriorProfile, ReadAerosolPriorProfile
from Read.ReadRetrievalOutputFiles import ReadprfFiles, ReadmreFiles, ReadaerFiles, ReadLogFiles, ReadmreParametricTest, ReadAerFromMreFiles, ReadContributionFunctions, ReadAllForAuroraOverTime
from Read.ReadZonalWind import ReadZonalWind

# Colormap definition
cmap = plt.get_cmap("magma")
# Small routine to retrieve the name of the gases used by NEMESIS
def RetrieveGasesNames(gas_id):
    if gas_id == 11: 
        gas_name = r'NH$_{3}$'
    if gas_id == 28:
        gas_name = r'PH$_{3}$'
    if gas_id == 26: 
        gas_name = r'C$_{2}$H$_{2}$'
    if gas_id == 32:
        gas_name = r'C$_{2}$H$_{4}$'
    if gas_id == 27: 
        gas_name = r'C$_{2}$H$_{6}$'
    if gas_id == 30:
        gas_name = r'C$_{4}$H$_{2}$'
    if gas_id == 39: 
        gas_name = r'H$_{2}$'
    if gas_id == 40:
        gas_name = r'He'
    if gas_id == 6: 
        gas_name = r'CH$_{4}$'

    return gas_name

# Plotting subroutines:

def PlotContributionFunction(over_axis="latitude"):

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
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6pknee02mbar_no852_no887"
                        ]
        ntest = len(retrieval_test)
        for itest in retrieval_test:
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{itest}/kk_figures/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            # Read retrieved profiles from .prf outputs files
            kk= ReadContributionFunctions(filepath=f"{fpath}{itest}", over_axis=over_axis)

            plt.plot(kk[:,0])
            plt.show()






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
        retrieval_test = [
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6pknee02mbar_no852_no887"
                        ]
        ntest = len(retrieval_test)
        # Loop over each prior used for retrievals tests
        for itest in retrieval_test:
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{itest}/chisq_ny/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
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
                plt.xlabel("Planetocentric Latitude", size=20)
            # Save figure in the retrievals outputs directory
            plt.savefig(f"{subdir}{itest}_chisquare.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{subdir}{itest}_chisquare.eps", dpi=100)
            # Close figure to avoid overlapping between plotting subroutines
            plt.close()

def PlotChiSquareOverNySuperpose(over_axis):

    print('Plotting NEMESIS ChiSquare over latitude (superpose figure of several tests) ...')
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
        subdir = f"{dir}/{iprior}_temp_aerosol1-10mu_C2H2p-C2H4-C2H6p_NH3-PH3-parametric_lat80S_no852_no887_reduce/chisquare_comparison/"
        # subdir = f"{dir}/{iprior}_temp_aerosol1-10or1mu_nospecies/chisquare_comparison/"
        # subdir = f"{dir}/{iprior}_temp_aerosol1-10mu_scale05or075or1/chisquare_comparison/"
        if not os.path.exists(subdir):
                os.makedirs(subdir)
        # List of retrieval tests for comparison...
        retrieval_test = [f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_NH3pt_PH3pt_lat80S_no852_no887_reduce", 
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_NH3pt_PH3_lat80S_no852_no887_reduce", 
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_NH3_PH3pt_lat80S_no852_no887_reduce", 
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_NH3pt_lat80S_no852_no887_reduce", 
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_NH3_lat80S_no852_no887_reduce", 
                        # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_NH3_lat80S_no852_no887_reduce"
                        ]
        ntest = len(retrieval_test)
        # Create the array to store chisquare over latitude for each test 
        maxchisg = []
        # Plot Figure of chisq/ny over latitude
        fig = plt.subplots(1, 1, figsize=(10, 6))
        # Loop over each prior used for retrievals tests
        for i, itest in enumerate(retrieval_test):
            col = cmap(i/ntest)
            # Read retrieved profiles from .prf outputs files
            chisquare, latitude, nlat = ReadLogFiles(filepath=f"{fpath}{itest}", over_axis=over_axis)
            maxchisg.append(np.nanmax(chisquare))
            plt.plot(latitude, chisquare, lw=2, label=r'T_aer10${\mu}$m_C$_2$H$_2$p-C$_2$H$_4$-C$_2$H$_6$p_'+f"{itest}"[69:-7], color = col)
            # plt.plot(latitude, chisquare, lw=2, label=f"{itest}"[14:32]+"C2H2_C2H6_NH3")
            # plt.plot(latitude, chisquare, lw=2, label=f"{itest}"[14:32])
            # plt.plot(latitude, chisquare, lw=2, label=f"{itest}"[14:])
        maxchi = np.nanmax(maxchisg)
        plt.grid()
        plt.legend()
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
            plt.xlabel("Planetocentric Latitude", size=20)
        # Save figure in the retrievals outputs directory
        plt.savefig(f"{subdir}chisquare_comparison.png", dpi=150, bbox_inches='tight')
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
    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/"
    # Array of prior file names
    prior = ['jupiter_v2021']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        retrieval_test = [
                        f"jupiter_v2021_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H4-C2H6_NH3_GRS_no852_no887",
                        f"jupiter_vzonal_temp_aerosol1-10mu-800mbar-05scale-01_NH3_GRS_no852_no887",
                        f"jupiter_vzonal_temp_aerosol1-10mu-800mbar-05scale-01_NH3p_GRS_no852_no887",
                        f"jupiter_vzonal_temp_aerosol1-10mu-800mbar-05scale-01_NH3_PH3_GRS_no852_no887",
                        f"jupiter_vzonal_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H4-C2H6_NH3_PH3_GRS_no852_no887"
                        ]
        ntest = len(retrieval_test)
        # Loop over each prior used for retrievals tests
        for itest in retrieval_test:
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{itest}/chisq_map/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            # Read retrieved profiles from .prf outputs files
            chisquare, latitude, nlat, longitude, nlon = ReadLogFiles(filepath=f"{fpath}{itest}", over_axis="2D")
            maxchi = np.nanmax(chisquare)
            plt.figure(figsize=(8, 6))
            im = plt.imshow(chisquare[:,:], vmin=0, vmax=ceil(maxchi)-1, cmap="magma",
                            origin='lower', extent=[longitude[0],longitude[-1],latitude[0],latitude[-1]])
            plt.xlabel('System III West Longitude', size=15)
            plt.ylabel('Planetocentric Latitude', size=15)
            plt.tick_params(labelsize=12)
            cbar = plt.colorbar(im, extend='both', fraction=0.025, pad=0.05)#, orientation='horizontal')
            cbar.ax.tick_params(labelsize=12)
            cbar.ax.locator_params(nbins=6)
            cbar.set_label("$\chi^{2}/N_y$", size=15)
            # Save figure showing calibation method 
            plt.savefig(f"{subdir}chisquare_maps.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{dir}{wave}_radiance_maps.eps", dpi=900)
            # Clear figure to avoid overlapping between plotting subroutines
            plt.close()








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
                    plt.xlabel("Planetocentric Latitude", size=20)
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
    prior = ['jupiter_v2021']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        retrieval_test = [f"{iprior}_temp_aerosol1-1mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6pknee02mbar_no852_no887",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6pknee02mbar_no852_no887"]
        ntest = len(retrieval_test)
        # Loop over each retrieval tests for the current prior file
        for itest in retrieval_test:
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{itest}/cross_sections/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            # Read retrieved profiles from .prf outputs files
            temperature, _, latitude, _, pressure, _, _, _, _ = ReadprfFiles(filepath=f"{fpath}{itest}", over_axis=over_axis)
            # Determine extreme values of temperature and levels for mapping
            max = np.nanmax(temperature)
            min = np.nanmin(temperature)
            levels_cmap = np.linspace(90, 191, num=25, endpoint=True)
            levels = np.linspace(90, 190, num=10, endpoint=True)

            # Mapping the temperature cross-section with zind location
            plt.figure(figsize=(8, 6))
            im = plt.contourf(latitude, pressure, temperature, cmap='viridis', levels=levels_cmap)
            plt.contour(latitude, pressure, temperature, levels=levels_cmap, colors="white")
            for iejet in range(0,nejet):
                plt.plot([ejets_c[iejet],ejets_c[iejet]],[1, 1000],color='white',linestyle="dashed")
            for iwjet in range(0,nwjet):
                plt.plot([wjets_c[iwjet],wjets_c[iwjet]],[1, 1000],color='white',linestyle="dotted")
            plt.ylim(0.01, 1000)
            # plt.xlim(-80, 80)
            plt.yscale('log')
            plt.gca().invert_yaxis()
            plt.tick_params(labelsize=15)        
            if over_axis=="longitude":
                plt.xlabel("System III West Longitude", size=20)
            elif over_axis=="latitude":
                plt.xlabel("Planetocentric Latitude", size=20)
            plt.ylabel(f"Presssure [mbar]", size=20)
            cbar = plt.colorbar(im, extend='both', fraction=0.046, pad=0.05, orientation='vertical', format="%.0f")
            cbar.ax.tick_params(labelsize=15)
            cbar.set_label("Retrieved Temperature [K]", fontsize=20)   
            # Save figure in the retrievals outputs directory
            plt.savefig(f"{subdir}{itest}_retrieved_temperature_zonal_wind.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{subdir}{itest}_retrieved_temperature_zonal_wind.eps", dpi=100)
            # Close figure to avoid overlapping between plotting subroutines
            plt.close()


            # Mapping the temperature cross-section alone
            plt.figure(figsize=(8, 6))
            im = plt.contourf(latitude, pressure, temperature, cmap='viridis', levels=levels_cmap)
            # plt.contour(latitude, pressure, temperature, levels=levels, colors="white")
            plt.ylim(0.01, 1000)
            # plt.xlim(-80, 80)
            plt.yscale('log')
            plt.gca().invert_yaxis()
            plt.tick_params(labelsize=15)     
            if over_axis=="longitude":
                plt.xlabel("System III West Longitude", size=20)
            elif over_axis=="latitude":
                plt.xlabel("Planetocentric Latitude", size=20)
            plt.ylabel(f"Presssure [mbar]", size=20)
            cbar = plt.colorbar(im, extend='both', fraction=0.046, pad=0.05, orientation='vertical', format="%.0f")
            cbar.ax.tick_params(labelsize=15)
            cbar.set_label("Retrieved Temperature [K]", fontsize=20)   
            # Save figure in the retrievals outputs directory
            plt.savefig(f"{subdir}{itest}_retrieved_temperature.png", dpi=150, bbox_inches='tight')
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
    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/"
    # Array of prior file names
    prior = ['jupiter_v2021']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        retrieval_test = [
                        f"jupiter_v2021_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H4-C2H6_NH3_GRS_no852_no887",
                        f"jupiter_vzonal_temp_aerosol1-10mu-800mbar-05scale-01_NH3_GRS_no852_no887",
                        f"jupiter_vzonal_temp_aerosol1-10mu-800mbar-05scale-01_NH3p_GRS_no852_no887",
                        f"jupiter_vzonal_temp_aerosol1-10mu-800mbar-05scale-01_NH3_PH3_GRS_no852_no887",
                        f"jupiter_vzonal_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H4-C2H6_NH3_PH3_GRS_no852_no887"
                        ]
        ntest = len(retrieval_test)
        # Loop over each prior used for retrievals tests
        for itest in retrieval_test:
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{itest}/temperature_map/"
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

            print("      ... retrieved temperature meridian cross-section")
            # Mapping the meridian temperature cross-section
            lon_index = (longitude == 157.5)
            tempkeep = temperature[:, :, lon_index]
            plt.figure(figsize=(8, 6))
            im = plt.contourf(latitude, pressure, tempkeep[:, :, 0], cmap='viridis', levels=levels_cmap)
            # plt.contour(latitude, pressure, tempkeep[:, :, 0], levels=levels, colors="white")
            plt.ylim(0.001, 1000)
            # plt.xlim(-80, 80)
            plt.yscale('log')
            plt.gca().invert_yaxis()
            plt.tick_params(labelsize=15)        
            plt.xlabel("Planetocentric Latitude", size=20)
            plt.ylabel(f"Presssure [mbar]", size=20)
            plt.title(f"Great Red Spot structure at {float(longitude[lon_index])}"+"$^{\circ}$W")
            cbar = plt.colorbar(im, extend='both', fraction=0.046, pad=0.05, orientation='vertical', format="%.0f")
            cbar.ax.tick_params(labelsize=15)
            cbar.set_label("Retrieved Temperature [K]", fontsize=20)   
            # Save figure in the retrievals outputs directory
            plt.savefig(f"{subdir}{itest}_meridian_cross_section_at_lon{float(longitude[lon_index])}.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{subdir}{itest}_meridian_cross_section_at_lon{float(longitude[lon_index])}.eps", dpi=100)
            # Close figure to avoid overlapping between plotting subroutines
            plt.close()

            print("      ... retrieved temperature zonal cross-section")
            # Mapping the zonal temperature cross-section
            lat_index = (latitude == -20.5)
            tempkeep = temperature[:, lat_index, :]
            plt.figure(figsize=(8, 6))
            im = plt.contourf(longitude, pressure, tempkeep[:, 0, :], cmap='viridis', levels=levels_cmap)
            # plt.contour(longitude, pressure, tempkeep[:, :, 0], levels=levels, colors="white")
            plt.ylim(0.001, 1000)
            # plt.xlim(-80, 80)
            plt.yscale('log')
            plt.gca().invert_yaxis()
            plt.gca().invert_xaxis()
            plt.tick_params(labelsize=15)        
            plt.xlabel("System III West longitude", size=20)
            plt.ylabel(f"Presssure [mbar]", size=20)
            plt.title(f"Great Red Spot structure at {float(latitude[lat_index])}"+"$^{\circ}$")
            cbar = plt.colorbar(im, extend='both', fraction=0.046, pad=0.05, orientation='vertical', format="%.0f")
            cbar.ax.tick_params(labelsize=15)
            cbar.set_label("Retrieved Temperature [K]", fontsize=20)   
            # Save figure in the retrievals outputs directory
            plt.savefig(f"{subdir}{itest}_zonal_cross_section_at_lat{float(latitude[lat_index])}.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{subdir}{itest}_zonal_cross_section_at_lat{float(latitude[lat_index])}.eps", dpi=100)
            # Close figure to avoid overlapping between plotting subroutines
            plt.close()

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
                ax[irow[iax]][icol[iax]].set_title(ititle[iax]+f"    {int(pressure[ind_pres[0][-1]])} mbar", fontfamily='serif', loc='left', fontsize=12)
                cbar = fig.colorbar(im, ax=ax[irow[iax]][icol[iax]], extend='both', fraction=0.04, pad=0.05, format="%.0f")#, orientation='horizontal')
                cbar.ax.tick_params(labelsize=12)
                cbar.ax.locator_params(nbins=6)
                cbar.set_label("Retrieved Temperature [K]", fontsize=8)
                iax+=1 
            plt.axes([0.15, 0.1, 0.8, 0.8], frameon=False) 
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel("System III West Longitude", size=18)
            plt.ylabel("Planetocentric Latitude", size=18)
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
                ax[irow[iax]][icol[iax]].set_title(ititle[iax]+f"    {int(pressure[ind_pres[0][-1]])} mbar", fontfamily='serif', loc='left', fontsize=12)
                cbar = fig.colorbar(im, ax=ax[irow[iax]][icol[iax]], extend='both', fraction=0.04, pad=0.05)#, orientation='horizontal')
                cbar.ax.tick_params(labelsize=11)
                cbar.ax.locator_params(nbins=7)
                cbar.set_label("Temperature errror (%)", size=10)
                iax+=1 
            plt.axes([0.15, 0.1, 0.8, 0.8], frameon=False) 
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel("System III West Longitude", size=18)
            plt.ylabel("Planetocentric Latitude", size=18)
            # Save figure showing calibation method 
            plt.savefig(f"{subdir}{itest}_all_filters_temperature_residual_maps.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{dir}{wave}_calibration_merid_profiles.eps", dpi=900)
            # Clear figure to avoid overlapping between plotting subroutines
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
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6pknee02mbar_NH3p_fshfix_no852_no887_aprmodif"
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
                        # f"{iprior}_temp_aerosol1-1mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6pknee02mbar_no852_no887",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6pknee02mbar_no852_no887"

                        ]
        ntest = len(retrieval_test)
        # Loop over each retrieval tests for the current prior file
        for itest in retrieval_test:
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{itest}/merid_radiances/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            # Read retrieved profiles from .mre outputs files
            radiance, wavenumb, rad_err, rad_fit, latitude, nlat = ReadmreFiles(filepath=f"{fpath}{itest}", over_axis=over_axis) 
            # Plotting retrieved radiance over wavenumber for each wavenumber
            for ifilt in range(Globals.nfilters):
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
                    plt.xlabel("Planetocentric Latitude", size=15)
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
                    plt.xlabel("Planetocentric Latitude", size=20)
                
                # Save figure in the retrievals outputs directory
                plt.savefig(f"{subdir}{itest}_residual_merid_radiance_at_{wavenumb[ifilt, 1]}.png", dpi=150, bbox_inches='tight')
                #plt.savefig(f"{subdir}{itest}_retrieved_radiance_at_{latitude[ilat]}.eps", dpi=100)
                # Close figure to avoid overlapping between plotting subroutines
                plt.close()

def PlotRetrievedRadianceMeridianSuperpose(over_axis):

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
        # If retrieval test comparison subdirectory does not exist, create it
        # subdir = f"{dir}/{iprior}_temp_aerosol1-10mu_greycloud_70-300mbar/merid_radiance_comparison/"
        # subdir = f"{dir}/{iprior}_temp_aerosol1-10mu_paraH2_greycloud_70-300mbar/merid_radiance_comparison/"
        subdir = f"{dir}/zonal_parametric_hydrocb_NH3/merid_radiance_comparison/"
        # subdir = f"{dir}/{iprior}_temp_aerosol1-10or1mu_nospecies/merid_radiance_comparison/"
        # subdir = f"{dir}/{iprior}_temp_aerosol1-10mu_scale05or075or1/merid_radiance_comparison/"
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

        radiance = np.empty((ntest, Globals.nfilters, 176))
        rad_fit = np.empty((ntest, Globals.nfilters, 176))
        rad_err = np.empty((ntest, Globals.nfilters, 176))
        # Loop over each retrieval tests for the current prior file
        iretrieve = 0
        for itest in retrieval_test:
            # Read retrieved profiles from .mre outputs files
            radiance[iretrieve, :, :], wavenumb, rad_err[iretrieve, :, :], rad_fit[iretrieve, :, :], latitude, nlat = ReadmreFiles(filepath=f"{fpath}{itest}", over_axis=over_axis)
            iretrieve +=1

        for ifilt in range(Globals.nfilters):
            iretrieve = 0
            # Plotting retrieved radiance over wavenumber for each wavenumber
            fig = plt.subplots(1, 1, figsize=(8, 3))
            plt.plot(latitude, radiance[iretrieve, ifilt, :], lw=2, label=f"Obs Radiance at {int(wavenumb[ifilt, 1])}", color='green')
            plt.fill_between(latitude, radiance[iretrieve, ifilt, :]-rad_err[iretrieve, ifilt, :], radiance[iretrieve, ifilt, :]+rad_err[iretrieve, ifilt, :], color='green', alpha=0.2) 
            for i, itest in enumerate(retrieval_test):
                col = cmap(i/ntest)
                plt.plot(latitude, rad_fit[iretrieve, ifilt, :], lw=2, label=f"{itest}"[52:-12], color=col)
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
                plt.xlabel("Planetocentric Latitude", size=15)
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
    # Retrieval outputs directory path
    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/"
    # Array of prior file names
    prior = ['jupiter_v2021']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        retrieval_test = [
                        f"jupiter_v2021_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H4-C2H6_NH3_GRS_no852_no887",
                        f"jupiter_vzonal_temp_aerosol1-10mu-800mbar-05scale-01_NH3_GRS_no852_no887",
                        f"jupiter_vzonal_temp_aerosol1-10mu-800mbar-05scale-01_NH3p_GRS_no852_no887",
                        f"jupiter_vzonal_temp_aerosol1-10mu-800mbar-05scale-01_NH3_PH3_GRS_no852_no887",
                        f"jupiter_vzonal_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H4-C2H6_NH3_PH3_GRS_no852_no887"
                        ]
        ntest = len(retrieval_test)
        # Loop over each retrieval tests for the current prior file
        for itest in retrieval_test:
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{itest}/maps_radiances/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            # Read retrieved profiles from .mre outputs files
            radiance, wavenumb, rad_err, rad_fit, latitude, nlat, longitude, nlon = ReadmreFiles(filepath=f"{fpath}{itest}", over_axis=over_axis)
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
                ax[irow[iax]][icol[iax]].set_title(ititle[iax]+f"    {(wavl)}"+" $\mu$m", fontfamily='serif', loc='left', fontsize=12)
                cbar = fig.colorbar(im, ax=ax[irow[iax]][icol[iax]], extend='both', fraction=0.04, pad=0.05)#, orientation='horizontal')
                cbar.ax.tick_params(labelsize=12)
                cbar.ax.locator_params(nbins=6)
                cbar.set_label("Radiance (nW cm$^{-1}$ sr$^{-1}$)", size=8)
                iax+=1 
            plt.axes([0.15, 0.1, 0.8, 0.8], frameon=False) 
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel("System III West Longitude", size=18)
            plt.ylabel("Planetocentric Latitude", size=18)
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
                ax[irow[iax]][icol[iax]].set_title(ititle[iax]+f"    {(wavl)}"+" $\mu$m", fontfamily='serif', loc='left', fontsize=12)
                cbar = fig.colorbar(im, ax=ax[irow[iax]][icol[iax]], extend='both', fraction=0.04, pad=0.05)#, orientation='horizontal')
                cbar.ax.tick_params(labelsize=12)
                cbar.ax.locator_params(nbins=6)
                cbar.set_label("Retrieved Radiance (nW cm$^{-1}$ sr$^{-1}$)", size=8)
                iax+=1 
            plt.axes([0.15, 0.1, 0.8, 0.8], frameon=False) 
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel("System III West Longitude", size=18)
            plt.ylabel("Planetocentric Latitude", size=18)
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
                ax[irow[iax]][icol[iax]].set_title(ititle[iax]+f"    {(wavl)}"+" $\mu$m", fontfamily='serif', loc='left', fontsize=12)
                cbar = fig.colorbar(im, ax=ax[irow[iax]][icol[iax]], extend='both', fraction=0.04, pad=0.05)#, orientation='horizontal')
                cbar.ax.tick_params(labelsize=11)
                cbar.ax.locator_params(nbins=7)
                cbar.set_label("Radiance errror (%)", size=10)
                iax+=1 
            plt.axes([0.15, 0.1, 0.8, 0.8], frameon=False) 
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel("System III West Longitude", size=18)
            plt.ylabel("Planetocentric Latitude", size=18)
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
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6pknee02mbar_no852_no887"
                        ]
        ntest = len(retrieval_test)
        # Loop over each retrieval tests for the current prior file
        for itest in retrieval_test:
            
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{itest}/cross_sections/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            # Read retrieved profiles from aerosol.prf outputs files
            aerosol, altitude, latitude, nlevel, nlat = ReadaerFiles(filepath=f"{fpath}{itest}", over_axis=over_axis)
            # Determine extreme values of aerosol and levels for mapping
            max = np.nanmax(aerosol)
            min = np.nanmin(aerosol)
            levels_cmap = np.linspace(min, max, num=15, endpoint=True)
            # Plotting retrieved aerosol profile for each latitude
            fig, axes = plt.subplots(1, 1, figsize=(8, 6), sharey=True)
            
            axes = plt.contourf(latitude, altitude[:,0], aerosol, cmap='cividis', levels=levels_cmap) 
            # plt.ylim(1, 1000)
            # plt.xlim(-80, 80)
            # plt.yscale('log')
            # plt.gca().invert_yaxis()
            plt.tick_params(labelsize=15)
            # axes.legend(loc="upper right", fontsize=15)
            # axes.tick_params(labelsize=15)
            # Add a big axis 
            plt.axes([0.1, 0.09, 0.8, 0.8], frameon=False) 
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)        
            if over_axis=="longitude":
                plt.xlabel("System III West Longitude", size=20)
            elif over_axis=="latitude":
                plt.xlabel("Planetocentric Latitude", size=20)
            plt.ylabel(f"Height [km]", size=20)
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
    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/"
    # Array of prior file names
    prior = ['jupiter_v2021']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        retrieval_test = [
                        # f"jupiter_v2021_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H4-C2H6_NH3_GRS_no852_no887",
                        f"jupiter_vzonal_temp_aerosol1-10mu-800mbar-05scale-01_NH3_GRS_no852_no887",
                        f"jupiter_vzonal_temp_aerosol1-10mu-800mbar-05scale-01_NH3p_GRS_no852_no887",
                        # f"jupiter_vzonal_temp_aerosol1-10mu-800mbar-05scale-01_NH3_PH3_GRS_no852_no887",
                        # f"jupiter_vzonal_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H4-C2H6_NH3_PH3_GRS_no852_no887"
                        ]
        ntest = len(retrieval_test)
        # Loop over each prior used for retrievals tests
        for itest in retrieval_test:
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{itest}/aerosol_map/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            # Read retrieved profiles from .prf outputs files
            aerosol, height, nlevel, latitude, nlat, longitude, nlon, ncoor = ReadaerFiles(filepath=f"{fpath}{itest}", over_axis=over_axis)
            aer_mre, aer_err, aer_fit, fit_err, lat_mre, nlat, long_mre, nlon = ReadAerFromMreFiles(filepath=f"{fpath}{itest}", over_axis=over_axis)

            
            # Determine extreme values of temperature and levels for mapping
            aermax = np.nanmax(aerosol[:, :, :])
            aermin = np.nanmin(aerosol[:, :, :])
            levels_cmap = np.linspace(aermin, aermax, num=20, endpoint=True)

            # Mapping the meridian aerosol cross-section
            lon_index = (longitude == 157.5)
            gaskeep = aerosol[:, :, lon_index]
            plt.figure(figsize=(8, 6))
            im = plt.contourf(latitude, height[:, 0, 0], gaskeep[:, :, 0], cmap='viridis', levels=levels_cmap)
            # plt.contour(latitude, height, gaskeep[:, :, 0], levels=levels, colors="white")
            # plt.ylim(0.001, 1000)
            # plt.xlim(-80, 80)
            # plt.yscale('log')
            # plt.gca().invert_yaxis()
            plt.tick_params(labelsize=15)        
            plt.xlabel("Planetocentric Latitude", size=20)
            plt.ylabel(f"Height [km]", size=20)
            plt.title(f"Great Red Spot structure at {float(longitude[lon_index])}"+"$^{\circ}$W")
            cbar = plt.colorbar(im, extend='both', fraction=0.046, pad=0.05, orientation='vertical')
            cbar.ax.tick_params(labelsize=15)
            # Save figure in the retrievals outputs directory
            plt.savefig(f"{subdir}{itest}_aerosol_meridian_cross_section_at_lon{float(longitude[lon_index])}.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{subdir}{itest}_meridian_cross_section_at_lon{float(longitude[lon_index])}.eps", dpi=100)
            # Close figure to avoid overlapping between plotting subroutines
            plt.close()

            # Mapping the zonal aerosol cross-section
            lat_index = (latitude == -20.5)
            gaskeep = aerosol[:, lat_index, :]
            plt.figure(figsize=(8, 6))
            im = plt.contourf(longitude, height[:, 0, 0], gaskeep[:, 0, :], cmap='viridis', levels=levels_cmap)
            # plt.contour(longitude, height, gaskeep[:, :, 0], levels=levels, colors="white")
            # plt.ylim(0.001, 1000)
            # plt.xlim(-80, 80)
            # plt.yscale('log')
            # plt.gca().invert_yaxis()
            plt.gca().invert_xaxis()
            plt.tick_params(labelsize=15)        
            plt.xlabel("System III West longitude", size=20)
            plt.ylabel(f"Height [km]", size=20)
            plt.title(f"Great Red Spot structure at {float(latitude[lat_index])}"+"$^{\circ}$")
            cbar = plt.colorbar(im, extend='both', fraction=0.046, pad=0.05, orientation='vertical')
            cbar.ax.tick_params(labelsize=15)
            # Save figure in the retrievals outputs directory
            plt.savefig(f"{subdir}{itest}_aerosol_zonal_cross_section_at_lat{float(latitude[lat_index])}.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{subdir}{itest}_aerosol_zonal_cross_section_at_lat{float(latitude[lat_index])}.eps", dpi=100)
            # Close figure to avoid overlapping between plotting subroutines
            plt.close()

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
            plt.ylabel(f"Planetocentric Latitude", size=20)
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
            im = plt.imshow(aer_fit[:, ::-1], cmap='cividis', vmin=aer_fitmin, vmax=aer_fitmax, # levels=levels_cmap,
                                                             origin='lower', extent=[long_mre[-1],long_mre[0],lat_mre[0],lat_mre[-1]])
            # plt.contour(longitude, height, aerkeep[:, :, 0], levels=levels, colors="white")
            # plt.ylim(0.001, 1000)
            # plt.xlim(-80, 80)
            # plt.yscale('log')
            # plt.gca().invert_yaxis()
            plt.gca().invert_xaxis()
            plt.tick_params(labelsize=15)        
            plt.xlabel("System III West longitude", size=20)
            plt.ylabel(f"Planetocentric Latitude", size=20)
            cbar = plt.colorbar(im, extend='both', fraction=0.025, pad=0.05, orientation='vertical')
            cbar.ax.tick_params(labelsize=15)
            cbar.set_label(f"Aerosol scale factor", fontsize=12)
            # Save figure in the retrievals outputs directory
            plt.savefig(f"{subdir}{itest}_aerosol_scale_factor_from-mre.png", dpi=150, bbox_inches='tight')
            #plt.savefig(f"{subdir}{itest}_aerosol_scale_factor_from-mre.eps", dpi=100)
            # Close figure to avoid overlapping between plotting subroutines
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
                    # Plot prior profile of the current gas
                    plt.plot(prior_gases[:, igas], prior_p, color='black', label=f"Prior profile of {RetrieveGasesNames(gases_id[igas])}")
                    # Plot the retrieved profile of the current gas
                    plt.plot(gases[:, ilat, igas], pressure, label=f"Retrieved profile of {RetrieveGasesNames(gases_id[igas])}")
                    plt.grid()
                    plt.ylim(0.01, 1000)
                    plt.yscale('log')
                    plt.gca().invert_yaxis()
                    plt.tick_params(labelsize=15)        
                    plt.xlabel(f"Volume Mixing Ratio", size=20)
                    plt.ylabel(f"Presssure [mbar]", size=20)
                    plt.legend(fontsize=20) 
                    # Save figure in the retrievals outputs directory
                    plt.savefig(f"{subdir}{itest}_retrieved_gas_{RetrieveGasesNames(gases_id[igas])}_profile_at_{latitude[ilat]}.png", dpi=150, bbox_inches='tight')
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
                    # Plot prior profile of the current gas
                    plt.plot(prior_gases[:, igas], prior_p, color='black', label=f"Prior profile of {RetrieveGasesNames(gases_id[igas])}")
                    # Plot the retrieved profile of the current gas
                    plt.plot(gases[:, ilat, igas], pressure, label=f"Retrieved profile of {RetrieveGasesNames(gases_id[igas])}")
            plt.grid()
            # plt.ylim(0.01, 1000)
            plt.yscale('log')
            plt.gca().invert_yaxis()
            plt.tick_params(labelsize=15)        
            plt.xlabel(f"Volume Mixing Ratio", size=20)
            plt.ylabel(f"Presssure [mbar]", size=20)
            plt.legend(fontsize=20) 
            # Save figure in the retrievals outputs directory
            plt.savefig(f"{subdir}{itest}_retrieved_gas_{RetrieveGasesNames(gases_id[igas])}_profile_at_{latitude[ilat]}.png", dpi=150, bbox_inches='tight')
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
        retrieval_test = [
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6pknee02mbar_no852_no887"
                        ]
        ntest = len(retrieval_test)
        # Loop over each retrieval tests for the current prior file
        for itest in retrieval_test:
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{itest}/cross_sections/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            # Read retrieved profiles from .prf outputs files
            _, gases, latitude, _, pressure, _, _, ngas, gases_id = ReadprfFiles(filepath=f"{fpath}{itest}", over_axis=over_axis)
            # Mapping the gases cross-section with zind location
            for igas in range(ngas):   
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
                    plt.xlabel("Planetocentric Latitude", size=20)
                plt.ylabel(f"Presssure [mbar]", size=20)
                cbar = plt.colorbar(im, extend='both', fraction=0.046, pad=0.05, orientation='vertical')
                cbar.ax.tick_params(labelsize=15)
                cbar.set_label(f"Retrieved {RetrieveGasesNames(gases_id[igas])}", fontsize=20)   
                # Save figure in the retrievals outputs directory
                plt.savefig(f"{subdir}{itest}_retrieved_gas_{RetrieveGasesNames(gases_id[igas])}.png", dpi=150, bbox_inches='tight')
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
    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/"
    # Array of prior file names
    prior = ['jupiter_v2021']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        retrieval_test = [
                        f"jupiter_v2021_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H4-C2H6_NH3_GRS_no852_no887",
                        f"jupiter_vzonal_temp_aerosol1-10mu-800mbar-05scale-01_NH3_GRS_no852_no887",
                        f"jupiter_vzonal_temp_aerosol1-10mu-800mbar-05scale-01_NH3p_GRS_no852_no887",
                        f"jupiter_vzonal_temp_aerosol1-10mu-800mbar-05scale-01_NH3_PH3_GRS_no852_no887",
                        f"jupiter_vzonal_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H4-C2H6_NH3_PH3_GRS_no852_no887"
                        ]
        ntest = len(retrieval_test)
        # Loop over each prior used for retrievals tests
        for itest in retrieval_test:
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{itest}/gases_map/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            # Read retrieved profiles from .prf outputs files
            temperature, gases, latitude, longitude, height, pressure, ncoor, nlevel, nlat, nlon, ngas, gases_id = ReadprfFiles(filepath=f"{fpath}{itest}", over_axis=over_axis)

            for igas in [0, 1, 2, 3, 4]:
                # Determine extreme values of temperature and levels for mapping
                gmax = np.nanmax(gases[:, :, :, igas])
                gmin = np.nanmin(gases[:, :, :, igas])
                levels_cmap = np.linspace(gmin, gmax, num=20, endpoint=True)

                # Mapping the meridian gases cross-section
                lon_index = (longitude == 157.5)
                gaskeep = gases[:, :, lon_index, igas]
                plt.figure(figsize=(8, 6))
                im = plt.contourf(latitude, pressure, gaskeep[:, :, 0], cmap='viridis', levels=levels_cmap)
                # plt.contour(latitude, pressure, gaskeep[:, :, 0], levels=levels, colors="white")
                plt.ylim(500, 1000) if igas < 2 else plt.ylim(0.001,1)
                # plt.xlim(-80, 80)
                plt.yscale('log')
                plt.gca().invert_yaxis()
                plt.tick_params(labelsize=15)        
                plt.xlabel("Planetocentric Latitude", size=20)
                plt.ylabel(f"Presssure [mbar]", size=20)
                plt.title(f"Great Red Spot structure at {float(longitude[lon_index])}"+"$^{\circ}$W")
                cbar = plt.colorbar(im, extend='both', fraction=0.046, pad=0.05, orientation='vertical')
                cbar.ax.tick_params(labelsize=15)
                cbar.set_label("Volume Mixing Ratio", fontsize=20)   
                # Save figure in the retrievals outputs directory
                plt.savefig(f"{subdir}{itest}_gas_{RetrieveGasesNames(gases_id[igas])}_meridian_cross_section_at_lon{float(longitude[lon_index])}.png", dpi=150, bbox_inches='tight')
                #plt.savefig(f"{subdir}{itest}_meridian_cross_section_at_lon{float(longitude[lon_index])}.eps", dpi=100)
                # Close figure to avoid overlapping between plotting subroutines
                plt.close()

                # Mapping the zonal gases cross-section
                lat_index = (latitude == -20.5)
                gaskeep = gases[:, lat_index, :, igas]
                plt.figure(figsize=(8, 6))
                im = plt.contourf(longitude, pressure, gaskeep[0, :, :], cmap='viridis', levels=levels_cmap)
                # plt.contour(longitude, pressure, gaskeep[:, :, 0], levels=levels, colors="white")
                plt.ylim(500, 1000) if igas < 2 else plt.ylim(0.001,1)
                # plt.xlim(-80, 80)
                plt.yscale('log')
                plt.gca().invert_yaxis()
                plt.gca().invert_xaxis()
                plt.tick_params(labelsize=15)        
                plt.xlabel("System III West longitude", size=20)
                plt.ylabel(f"Presssure [mbar]", size=20)
                plt.title(f"Great Red Spot structure at {float(latitude[lat_index])}"+"$^{\circ}$")
                cbar = plt.colorbar(im, extend='both', fraction=0.046, pad=0.05, orientation='vertical')
                cbar.ax.tick_params(labelsize=15)
                cbar.set_label("Volume Mixing Ratio", fontsize=20)   
                # Save figure in the retrievals outputs directory
                plt.savefig(f"{subdir}{itest}_gas_{RetrieveGasesNames(gases_id[igas])}_zonal_cross_section_at_lat{float(latitude[lat_index])}.png", dpi=150, bbox_inches='tight')
                #plt.savefig(f"{subdir}{itest}_zonal_cross_section_at_lat{float(latitude[lat_index])}.eps", dpi=100)
                # Close figure to avoid overlapping between plotting subroutines
                plt.close()

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
                    ax[irow[iax]][icol[iax]].set_title(ititle[iax]+f"    {ptarget[ipressure]} mbar    {RetrieveGasesNames(gases_id[igas])}", fontfamily='serif', loc='left', fontsize=12)
                    cbar = fig.colorbar(im, ax=ax[irow[iax]][icol[iax]], extend='both', fraction=0.04, pad=0.05, format='%.2e')#, orientation='horizontal')
                    cbar.ax.tick_params(labelsize=12)
                    cbar.ax.locator_params(nbins=6)
                    cbar.set_label("Volume Mixing Ratio", fontsize=8)
                    iax+=1 
                plt.axes([0.15, 0.1, 0.8, 0.8], frameon=False) 
                plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
                plt.xlabel("System III West Longitude", size=18)
                plt.ylabel("Planetocentric Latitude", size=18)
                # Save figure showing calibation method 
                plt.savefig(f"{subdir}{itest}_gas_{RetrieveGasesNames(gases_id[igas])}_maps_at_11_pressure_levels.png", dpi=150, bbox_inches='tight')
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
                gmin = np.nanmin(gases[ind_pres[0][-1], :, :, igas])
                levels_cmap = np.linspace(gmin, gmax, num=20, endpoint=True)
                # subplot showing the regional radiance maps
                im = plt.imshow(gases[ind_pres[0][-1], :, :, igas], cmap='plasma', vmin=gmin, vmax=gmax, # levels=levels_cmap,
                                                            origin='lower', extent=[longitude[0],longitude[-1],latitude[0],latitude[-1]])
                plt.tick_params(labelsize=14)
                cbar = fig.colorbar(im, extend='both', fraction=0.04, pad=0.05, format='%.2e')#, orientation='horizontal')
                cbar.ax.tick_params(labelsize=12)
                cbar.ax.locator_params(nbins=6)
                cbar.set_label(f"Volume Mixing Ratio {RetrieveGasesNames(gases_id[igas])} at {ptarget[ipressure]} mbar", fontsize=12)
                plt.xlabel("System III West Longitude", size=18)
                plt.ylabel("Planetocentric Latitude", size=18)
                # Save figure showing calibation method 
                plt.savefig(f"{subdir}{itest}_gas_{RetrieveGasesNames(gases_id[igas])}_maps_at_pressure_{ptarget[ipressure]}.png", dpi=150, bbox_inches='tight')
                #plt.savefig(f"{subdir}{itest}_gases_maps_at_11_pressure_levels.eps", dpi=900)
                # Clear figure to avoid overlapping between plotting subroutines
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
        retrieval_test = [#f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_NH3_no852_no887_aprmodif",
                        # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_NH3p_no852_no887_aprmodif",
                        # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_knee02mbar_NH3p_no852_no887_aprmodif",
                        # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H4-C2H6_NH3_no852_no887",
                        # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H4-C2H6_NH3p_fshfix_no852_no887_aprmodif",
                        # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H4-C2H6pknee02mbar_NH3p_fshfix_no852_no887_aprmodif",
                        # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6_NH3p_fshfix_no852_no887_aprmodif",
                        # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6pknee02mbar_NH3_no852_no887",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6pknee02mbar_no852_no887"
                        # f"{iprior}_temp_aerosol1-1mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6pknee02mbar_no852_no887"
                        # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2pknee008mbar-C2H4-C2H6pknee02mbar_NH3p_fshfix_no852_no887_aprmodif"
                        #f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2p-C2H4-C2H6p_no852_no887_aprmodif"
                        ]
        ntest = len(retrieval_test)

        for ilat in [-0.5, -25.5, -60.5, -65.5 ,-70.5, -75.5, -80.5, -85.5]:
            # Setting figure grid of subplots
            fig = plt.figure(figsize=(12, 10))
            grid = plt.GridSpec(1, 4, wspace=0.5, hspace=0.6)
            c2h2_prf = fig.add_subplot(grid[0,0])
            c2h4_prf = fig.add_subplot(grid[0,1], sharey=c2h2_prf)
            c2h6_prf = fig.add_subplot(grid[0,2], sharey=c2h2_prf)
            nh3_prf  = fig.add_subplot(grid[0,3], sharey=c2h2_prf)
            # ph3_prf  = fig.add_subplot(grid[0,4], sharey=c2h2_prf)
            # Loop over each retrieval tests for the current prior file
            for i, itest in enumerate(retrieval_test):
                # If retrieval test subdirectory does not exist, create it
                subdir = f"{dir}{itest}/ComparisonParametricGasesHydrocarbons/"
                if not os.path.exists(subdir):
                    os.makedirs(subdir)
                col = cmap(i/ntest)
                # Read ChiSquare values from log files
                chisquare, latitude, nlat = ReadLogFiles(filepath=f"{fpath}{itest}", over_axis="latitude")
                # Read profile data from NEMESIS prior file 
                _, prior_p, _, _, prior_gases, _, _, _ = ReadTemperatureGasesPriorProfile(f"{fpath}{itest}/core_1/")    
                # Read retrieved profiles from .prf outputs files
                _, gases, latitude, _, pressure, _, _, ngas, gases_id = ReadprfFiles(filepath=f"{fpath}{itest}", over_axis="latitude")
                

                ind_lat = np.where(latitude==ilat)
                print(ind_lat[0][0])
                if i==0:
                    c2h2_prf.plot(prior_gases[:, 2], prior_p, color='blue', lw=2, label=f"prior profile at {latitude[ind_lat[0][0]]}")
                c2h2_prf.plot(gases[:, ind_lat[0][0], 2], pressure, label=f"{itest}"[52:], color=col, lw=2)
                c2h2_prf.set_title(f"{RetrieveGasesNames(gases_id[2])}")
                c2h2_prf.grid()
                c2h2_prf.set_ylim(0.001, 1000)
                c2h2_prf.set_yscale('log')
                c2h2_prf.invert_yaxis()
                c2h2_prf.tick_params(labelsize=15)        
            
                if i==0:
                    c2h4_prf.plot(prior_gases[:, 3], prior_p, color='blue', lw=2)
                c2h4_prf.plot(gases[:, ind_lat[0][0], 3], pressure, color=col, lw=2)
                c2h4_prf.set_title(label=f"{RetrieveGasesNames(gases_id[3])}")
                c2h4_prf.grid()
                c2h4_prf.set_ylim(0.001, 1000)
                c2h4_prf.set_yscale('log')
                c2h4_prf.invert_yaxis()
                c2h4_prf.tick_params(labelsize=15)        
            
                if i==0:
                    c2h6_prf.plot(prior_gases[:, 4], prior_p, color='blue', lw=2)
                c2h6_prf.plot(gases[:, ind_lat[0][0], 4], pressure, color=col, lw=2)
                c2h6_prf.set_title(f"{RetrieveGasesNames(gases_id[4])}")
                c2h6_prf.grid()
                c2h6_prf.set_ylim(0.001, 1000)
                c2h6_prf.set_yscale('log')
                c2h6_prf.invert_yaxis()
                c2h6_prf.tick_params(labelsize=15)        
            
                if i==0: 
                    nh3_prf.plot(prior_gases[:, 0], prior_p, color='blue', lw=2)
                nh3_prf.plot(gases[:, ind_lat[0][0], 0], pressure, color=col, lw=2)
                nh3_prf.set_title(f"{RetrieveGasesNames(gases_id[0])}")
                nh3_prf.grid()
                nh3_prf.set_ylim(0.001, 1000)
                nh3_prf.set_yscale('log')
                nh3_prf.invert_yaxis()
                nh3_prf.tick_params(labelsize=15)        
            
                
                # ph3_prf.plot(prior_gases[:, 1], prior_p, color='blue')
                # ph3_prf.plot(gases[:, ind_lat[0][0], 1], pressure, color=col)
                # ph3_prf.set_title(f"{RetrieveGasesNames(gases_id[1])}")
                # ph3_prf.grid()
                # ph3_prf.set_ylim(0.01, 1000)
                # ph3_prf.set_yscale('log')
                # ph3_prf.invert_yaxis()
                # ph3_prf.tick_params(labelsize=15)        
            handles, labels = c2h2_prf.get_legend_handles_labels()  
            fig.legend(handles, labels, loc='upper right',fontsize=12)
            plt.axes([0.1, 0.08, 0.8, 0.85], frameon=False) 
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel(f"Volume Mixing Ratio", size=20)
            plt.ylabel(f"Presssure [mbar]", size=20)

            plt.savefig(f"{subdir}parametric_test_gases_hydrocarbons_profiles_lat_{latitude[ind_lat[0][0]]}_reduce_cmaps_data.png", dpi=150, bbox_inches='tight')
            plt.close()

        # Setting figure grid of subplots
        fig = plt.figure(figsize=(8, 10))
        grid = plt.GridSpec(4, 4, wspace=0.5, hspace=0.6)
        
        chiqsq     = fig.add_subplot(grid[0, :])
        merid_c2h2 = fig.add_subplot(grid[1, :])
        merid_c2h4 = fig.add_subplot(grid[2, :], sharex=merid_c2h2)
        merid_c2h6 = fig.add_subplot(grid[3, :], sharex=merid_c2h2)
        # merid_nh3  = fig.add_subplot(grid[4, :], sharex=merid_c2h2)
        # merid_nh3_2 = fig.add_subplot(grid[5, :], sharex=merid_c2h2)



        # Loop over each retrieval tests for the current prior file
        for i, itest in enumerate(retrieval_test):
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{itest}/ComparisonParametricGasesHydrocarbons/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            col = cmap(i/ntest)
            # Read ChiSquare values from log files
            chisquare, latitude, nlat = ReadLogFiles(filepath=f"{fpath}{itest}", over_axis="latitude")
            # Read retrieved profiles from .prf outputs files
            _, gases, latitude, _, pressure, _, _, ngas, gases_id = ReadprfFiles(filepath=f"{fpath}{itest}", over_axis="latitude")

            chiqsq.plot(latitude, chisquare, label=f"{itest}"[52:], color=col)
            chiqsq.set_ylim(0, 1.5)
            chiqsq.grid()
            chiqsq.set_ylabel('\u03C7'r'$^{2}/N_y$', size=15)
            chiqsq.tick_params(labelsize=12)     

            ind_pres = np.where(pressure >= 5.)
            merid_c2h2.set_title(r'C$_{2}$H$_{2}$'+f" at {int(pressure[ind_pres[0][-1]])} mbar")
            merid_c2h2.plot(latitude, gases[ind_pres[0][-1], :, 2], color=col)
            merid_c2h2.tick_params(labelsize=12)

            merid_c2h4.set_title(r'C$_{2}$H$_{4}$'+f" at {int(pressure[ind_pres[0][-1]])} mbar")
            merid_c2h4.plot(latitude, gases[ind_pres[0][-1], :, 3], color=col)
            merid_c2h4.tick_params(labelsize=12)

            merid_c2h6.set_title(r'C$_{2}$H$_{6}$'+f" at {int(pressure[ind_pres[0][-1]])} mbar")
            merid_c2h6.plot(latitude, gases[ind_pres[0][-1], :, 4], color=col)
            merid_c2h6.tick_params(labelsize=12)

            # ind_pres = np.where(pressure >= 500)
            # merid_nh3.set_title(r'NH$_{3}$'+f" at {int(pressure[ind_pres[0][-1]])} mbar")
            # merid_nh3.plot(latitude, gases[ind_pres[0][-1], :, 0], color=col)
            # merid_nh3.tick_params(labelsize=12)

            # ind_pres = np.where(pressure >= 800)
            # merid_nh3_2.set_title(r'NH$_{3}$'+f" at {int(pressure[ind_pres[0][-1]])} mbar")
            # merid_nh3_2.plot(latitude, gases[ind_pres[0][-1], :, 0], color=col)
            # merid_nh3_2.tick_params(labelsize=12)        

        handles, labels = chiqsq.get_legend_handles_labels()  
        fig.legend(handles, labels, loc='upper center',fontsize=12)
        plt.axes([0.12, 0.1, 0.8, 0.65], frameon=False) 
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.xlabel(f"Planetocentric Latitude", size=20)
        plt.ylabel(f"Volume Mixing Ratio", size=20) 
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
                sixtysouth_prf.set_title(f"{RetrieveGasesNames(gases_id[2])}")
                sixtyfivesouth_prf.loglog(gases[:, ind_lat[0][0], 3], pressure, color=col)
                sixtyfivesouth_prf.set_title(label=f"{RetrieveGasesNames(gases_id[3])}")
                seventysouth_prf.loglog(gases[:, ind_lat[0][0], 4], pressure, color=col)
                seventysouth_prf.set_title(f"{RetrieveGasesNames(gases_id[4])}")
                seventyfivesouth_prf.loglog(gases[:, ind_lat[0][0], 0], pressure, color=col)
                seventyfivesouth_prf.set_title(f"{RetrieveGasesNames(gases_id[0])}")
                eightysouth_prf.loglog(gases[:, ind_lat[0][0], 1], pressure, color=col)
                eightysouth_prf.set_title(f"{RetrieveGasesNames(gases_id[1])}")
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
    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/"
    # Array of prior file names
    prior = ['jupiter_v2021']#, 'jupiter_v2016']
    # Loop over each prior used for retrievals tests
    for iprior in prior:
        retrieval_test = ["jupiter_vzonal_lat-80_temp_aurora_RegAv_no852_no887", 
                        "jupiter_aerosol_vzonal_lat-80_temp_aurora_RegAv_no852_no887",
                        "jupiter_aerosol_vzonal_temp_C2H2-C2H4-C2H6_aurora_RegAv_no852_no887",
                        "jupiter_aerosol_vzonal_temp_C2H2-C2H4-C2H6_NH3_aurora_RegAv_no852_no887",
                        "jupiter_vzonal_aerosol_v1mu_temp_1mu_800mbar_05scale_01_C2H2-C2H4-C2H6_aurora_RegAv_no852_no887",
                        "jupiter_vzonal_aerosol_v1mu_temp_C2H2-C2H4-C2H6_aurora_RegAv_no852_no887",
                        "jupiter_vzonal_aerosol_v1mu_temp_C2H2-C2H4-C2H6_NH3_aurora_RegAv_no852_no887",
                        "jupiter_aerosol_vzonal_temp_aerosol1-10mu-800mbar-05scale_C2H2-C2H4-C2H6_aurora_RegAv_no852_no887",
                        "jupiter_v2021_aerosol_vzonal_temp_C2H2-C2H4-C2H6_aurora_RegAv_no852_no887",
                        "jupiter_aerosol_vzonal_temp_C2H2-C2H4-C2H6_aurora_RegAv_no852_no887_top_Terr20K",
                        "jupiter_aerosol_vzonal_temp_aurora_RegAv_no852_no887_top_Terr20K",
                        "jupiter_aerosol_vzonal1mu_temp_aerosol1-1mu-800mbar-05scale-01_C2H2-C2H4-C2H6_aurora_RegAv_no852_no887",
                        "jupiter_aerosol_vzonal1mu_temp_C2H2-C2H4-C2H6_aurora_RegAv_no852_no887",
                        "jupiter_aerosol_vzonal1mu_temp_aurora_RegAv_no852_no887"
                        ]
        ntest = len(retrieval_test)
        night_labels = ['May 24th', 'May 25th-26th', 'May 26th-27th']
        for itest in retrieval_test:
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{itest}/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            nnight, time, chisquare, radiance, rad_err, rad_fit, wavenumb, temp_prior_mre, temp_errprior_mre, temp_fit_mre, temp_errfit_mre, aer_mre, aer_err,  aer_fit, fit_err, height, pressure, temperature, gases, gases_id, aer_prf, h_prf= ReadAllForAuroraOverTime(filepath=f"{fpath}{itest}")

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

            fig, axes = plt.subplots(1, 1, figsize=(8, 10))
            # Read profile data from NEMESIS prior file 
            # Plot the prior only for the last itest (because it's the same for all itest)
            axes.plot(temp_prior_mre[:, inight], prior_p, lw=2, label=f"prior", color='green')
            axes.fill_betweenx(prior_p, temp_prior_mre[:, inight]-temp_errprior_mre[:, inight], temp_prior_mre[:, inight]+temp_errprior_mre[:, inight], color='green', alpha=0.1)
            for inight in range(nnight):
                col = cmap(inight/nnight)
                # Plot 
                axes.plot(temp_fit_mre[:, inight], pressure, lw=2, label=f"night {night_labels[inight]}", color = col)
                axes.fill_betweenx(pressure, temp_fit_mre[:, inight]-temp_errfit_mre[:, inight], temp_fit_mre[:, inight]+temp_errfit_mre[:, inight], color=col, alpha=0.2)
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
                c2h2_prf.set_title(f"{RetrieveGasesNames(gases_id[2])}")
                c2h2_prf.grid()
                c2h2_prf.set_ylim(0.001, 1000)
                c2h2_prf.set_yscale('log')
                c2h2_prf.invert_yaxis()
                c2h2_prf.tick_params(labelsize=15)        
            
                if inight==0:
                    c2h4_prf.plot(prior_gases[:, 3], prior_p, color='green', lw=2)
                c2h4_prf.plot(gases[:, inight, 3], pressure, color=col, lw=2)
                c2h4_prf.set_title(label=f"{RetrieveGasesNames(gases_id[3])}")
                c2h4_prf.grid()
                c2h4_prf.set_ylim(0.001, 1000)
                c2h4_prf.set_yscale('log')
                c2h4_prf.invert_yaxis()
                c2h4_prf.tick_params(labelsize=15)        
            
                if inight==0:
                    c2h6_prf.plot(prior_gases[:, 4], prior_p, color='green', lw=2)
                c2h6_prf.plot(gases[:, inight, 4], pressure, color=col, lw=2)
                c2h6_prf.set_title(f"{RetrieveGasesNames(gases_id[4])}")
                c2h6_prf.grid()
                c2h6_prf.set_ylim(0.001, 1000)
                c2h6_prf.set_yscale('log')
                c2h6_prf.invert_yaxis()
                c2h6_prf.tick_params(labelsize=15)        
            
                # if inight==0: 
                #     nh3_prf.plot(prior_gases[:, 0], prior_p, color='green', lw=2)
                # nh3_prf.plot(gases[:, inight, 0], pressure, color=col, lw=2)
                # nh3_prf.set_title(f"{RetrieveGasesNames(gases_id[0])}")
                # nh3_prf.grid()
                # nh3_prf.set_ylim(0.001, 1000)
                # nh3_prf.set_yscale('log')
                # nh3_prf.invert_yaxis()
                # nh3_prf.tick_params(labelsize=15)        
            
                
                # ph3_prf.plot(prior_gases[:, 1], prior_p, color='green')
                # ph3_prf.plot(gases[:, inight, 1], pressure, color=col)
                # ph3_prf.set_title(f"{RetrieveGasesNames(gases_id[1])}")
                # ph3_prf.grid()
                # ph3_prf.set_ylim(0.01, 1000)
                # ph3_prf.set_yscale('log')
                # ph3_prf.invert_yaxis()
                # ph3_prf.tick_params(labelsize=15)        
            handles, labels = c2h2_prf.get_legend_handles_labels()  
            fig.legend(handles, labels, loc='upper right',fontsize=12, ncol=2)
            plt.axes([0.1, 0.08, 0.8, 0.85], frameon=False) 
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
            plt.xlabel(f"Volume Mixing Ratio", size=20)
            plt.ylabel(f"Presssure [mbar]", size=20)
            plt.savefig(f"{subdir}aurora_over_time_gases_hydrocarbons_ammonia_profiles.png", dpi=150, bbox_inches='tight')
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
                ax[irow[iax]][icol[iax]].set_xticks(ticks=np.arange(1,4,step=1), labels=['May\n24th', 'May\n25th-26th', 'May\n26th-27th'])
                ax[irow[iax]][icol[iax]].grid()
                ax[irow[iax]][icol[iax]].set_title(ititle[iax]+f"    {(wavl)}"+" $\mu$m", fontfamily='serif', loc='left', fontsize=12)
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
            

            
                    
