import os
import numpy as np
import matplotlib.pyplot as plt
import Globals
from Read.ReadPrior import ReadTemperaturePriorProfile, ReadAerosolPriorProfile
from Read.ReadRetrievalOutputFiles import ReadprfFiles, ReadmreFiles, ReadaerFiles

def PlotRetrievedTemperature():

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
        retrieval_test = [f"{iprior}_temp_aerosol1-1mu-800mbar-05scale-01_retrieval",
                        # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_retrieval"]#,
                        f"{iprior}_temp_aerosol1-1mu-800mbar-05scale-01_C2H2-C2H6_NH3_retrieval",
                        f"{iprior}_temp_aerosol1-5mu-800mbar-05scale-01_C2H2-C2H6_NH3_retrieval",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H6_NH3_retrieval", 
                        f"{iprior}_temp_aerosol2_10-1mu_08bar_05scale_04-01bar_C2H2_C2H6_NH3_retrieval",
                        f"{iprior}_temp_aerosol2_10-5mu_08bar_05scale_04-01bar_C2H2_C2H6_NH3_retrieval",
                        f"{iprior}_temp_aerosol2_5-1mu_08bar_05scale_04-01bar_C2H2_C2H6_NH3_retrieval"]
        ntest = len(retrieval_test)
        # Loop over each prior used for retrievals tests
        for itest in retrieval_test:
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{itest}/meridians/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            # # Read profile data from NEMESIS prior file 
            # _, prior_p, prior_temperature, prior_error, _, _, nlevel, _ = ReadTemperaturePriorProfile(f"{fpath}{itest}/core_1/")
            # Read retrieved profiles from .prf outputs files
            temperature, _, latitude, _, pressure, nlat, nlevel, _ = ReadprfFiles(f"{fpath}{itest}")
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
                plt.xlabel(f"Planetocentric Latitude", size=20)
                # Save figure in the retrievals outputs directory
                plt.savefig(f"{subdir}{itest}_retrieved_temperature_{pressure[ilev]}mbar.png", dpi=300)
                #plt.savefig(f"{subdir}{itest}_retrieved_temperature_{pressure[ilev]}mbar.eps", dpi=300)
                # Close figure to avoid overlapping between plotting subroutines
                plt.close()

def PlotRetrievedTemperatureProfile():

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
        retrieval_test = [f"{iprior}_temp_aerosol1-1mu-800mbar-05scale-01_retrieval",
                        # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_retrieval"]#,
                        f"{iprior}_temp_aerosol1-1mu-800mbar-05scale-01_C2H2-C2H6_NH3_retrieval",
                        f"{iprior}_temp_aerosol1-5mu-800mbar-05scale-01_C2H2-C2H6_NH3_retrieval",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H6_NH3_retrieval", 
                        f"{iprior}_temp_aerosol2_10-1mu_08bar_05scale_04-01bar_C2H2_C2H6_NH3_retrieval",
                        f"{iprior}_temp_aerosol2_10-5mu_08bar_05scale_04-01bar_C2H2_C2H6_NH3_retrieval",
                        f"{iprior}_temp_aerosol2_5-1mu_08bar_05scale_04-01bar_C2H2_C2H6_NH3_retrieval"]
        ntest = len(retrieval_test)
        # Loop over each retrieval tests for the current prior file
        for itest in retrieval_test:
            # Read profile data from NEMESIS prior file 
            _, prior_p, prior_temperature, prior_err, _, _, _, _ = ReadTemperaturePriorProfile(f"{fpath}{itest}/core_1/")
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{itest}/profiles/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            # Read retrieved profiles from .prf outputs files
            temperature, gases, latitude, height, pressure, nlat, nlevel, ngas = ReadprfFiles(f"{fpath}{itest}")
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
                plt.savefig(f"{subdir}{itest}_retrieved_temperature_profile_at_{latitude[ilat]}.png", dpi=300)
                #plt.savefig(f"{subdir}{itest}_retrieved_temperature_profile_at_{latitude[ilat]}.eps", dpi=300)
                # Close figure to avoid overlapping between plotting subroutines
                plt.close()

def PlotRetrievedRadiance():

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
        retrieval_test = [f"{iprior}_temp_aerosol1-1mu-800mbar-05scale-01_retrieval",
                        # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_retrieval"]#,
                        f"{iprior}_temp_aerosol1-1mu-800mbar-05scale-01_C2H2-C2H6_NH3_retrieval",
                        f"{iprior}_temp_aerosol1-5mu-800mbar-05scale-01_C2H2-C2H6_NH3_retrieval",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H6_NH3_retrieval", 
                        f"{iprior}_temp_aerosol2_10-1mu_08bar_05scale_04-01bar_C2H2_C2H6_NH3_retrieval",
                        f"{iprior}_temp_aerosol2_10-5mu_08bar_05scale_04-01bar_C2H2_C2H6_NH3_retrieval",
                        f"{iprior}_temp_aerosol2_5-1mu_08bar_05scale_04-01bar_C2H2_C2H6_NH3_retrieval"]
        ntest = len(retrieval_test)
        # Loop over each retrieval tests for the current prior file
        for itest in retrieval_test:
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{itest}/radiances/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            # Read retrieved profiles from .mre outputs files
            radiance, wavenumb, rad_err, rad_fit, latitude, nlat = ReadmreFiles(f"{fpath}{itest}") 
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
                plt.savefig(f"{subdir}{itest}_retrieved_radiance_at_{latitude[ilat]}.png", dpi=300)
                #plt.savefig(f"{subdir}{itest}_retrieved_radiance_at_{latitude[ilat]}.eps", dpi=300)
                # Close figure to avoid overlapping between plotting subroutines
                plt.close()

def PlotRetrievedRadianceMeridian():

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
        retrieval_test = [f"{iprior}_temp_aerosol1-1mu-800mbar-05scale-01_retrieval",
                        # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_retrieval"]#,
                        f"{iprior}_temp_aerosol1-1mu-800mbar-05scale-01_C2H2-C2H6_NH3_retrieval",
                        f"{iprior}_temp_aerosol1-5mu-800mbar-05scale-01_C2H2-C2H6_NH3_retrieval",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H6_NH3_retrieval", 
                        f"{iprior}_temp_aerosol2_10-1mu_08bar_05scale_04-01bar_C2H2_C2H6_NH3_retrieval",
                        f"{iprior}_temp_aerosol2_10-5mu_08bar_05scale_04-01bar_C2H2_C2H6_NH3_retrieval",
                        f"{iprior}_temp_aerosol2_5-1mu_08bar_05scale_04-01bar_C2H2_C2H6_NH3_retrieval"]
        ntest = len(retrieval_test)
        # Loop over each retrieval tests for the current prior file
        for itest in retrieval_test:
            # If retrieval test subdirectory does not exist, create it
            subdir = f"{dir}{itest}/merid_radiances/"
            if not os.path.exists(subdir):
                os.makedirs(subdir)
            # Read retrieved profiles from .mre outputs files
            radiance, wavenumb, rad_err, rad_fit, latitude, nlat = ReadmreFiles(f"{fpath}{itest}") 
            # Plotting retrieved radiance over wavenumber for each wavenumber
            for ifilt in range(Globals.nfilters):
                fig, axes = plt.subplots(1, 1, figsize=(10, 7), sharex=True, sharey=True)        
                axes.plot(latitude, radiance[ifilt, :], lw=2, label=f"Obs Radiance at {wavenumb[ifilt, 1]}", color='orange')
                axes.fill_between(latitude, radiance[ifilt, :]-rad_err[ifilt, :], radiance[ifilt, :]+rad_err[ifilt, :], color='orange', alpha=0.2)
                axes.plot(latitude, rad_fit[ifilt, :], lw=2, label=f"Retrieved Radiance")
                axes.grid()
                axes.legend(loc="upper right", fontsize=15)
                axes.tick_params(labelsize=15)
                # Add a big axis 
                plt.axes([0.1, 0.09, 0.8, 0.8], frameon=False) 
                plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)        
                plt.xlabel("Latitude", size=20)
                plt.ylabel("Radiance [nW cm$^{-2}$ sr$^{-1}$ cm]", size=20)
                # Save figure in the retrievals outputs directory
                plt.savefig(f"{subdir}{itest}_retrieved_merid_radiance_at_{wavenumb[ifilt, 1]}.png", dpi=300)
                #plt.savefig(f"{subdir}{itest}_retrieved_radiance_at_{latitude[ilat]}.eps", dpi=300)
                # Close figure to avoid overlapping between plotting subroutines
                plt.close()
               
def PlotRetrievedAerosolProfile():

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
        retrieval_test = [f"{iprior}_temp_aerosol1-1mu-800mbar-05scale-01_retrieval",
                        # f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_retrieval"]#,
                        f"{iprior}_temp_aerosol1-1mu-800mbar-05scale-01_C2H2-C2H6_NH3_retrieval",
                        f"{iprior}_temp_aerosol1-5mu-800mbar-05scale-01_C2H2-C2H6_NH3_retrieval",
                        f"{iprior}_temp_aerosol1-10mu-800mbar-05scale-01_C2H2-C2H6_NH3_retrieval", 
                        f"{iprior}_temp_aerosol2_10-1mu_08bar_05scale_04-01bar_C2H2_C2H6_NH3_retrieval",
                        f"{iprior}_temp_aerosol2_10-5mu_08bar_05scale_04-01bar_C2H2_C2H6_NH3_retrieval",
                        f"{iprior}_temp_aerosol2_5-1mu_08bar_05scale_04-01bar_C2H2_C2H6_NH3_retrieval"]
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
            aerosol, altitude, latitude, nlevel, nlat = ReadaerFiles(f"{fpath}{itest}")
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
                plt.savefig(f"{subdir}{itest}_retrieved_aerosol_profile_at_{latitude[ilat]}.png", dpi=300)
                #plt.savefig(f"{subdir}{itest}_retrieved_aerosol_profile_at_{latitude[ilat]}.eps", dpi=300)
                # Close figure to avoid overlapping between plotting subroutines
                plt.close()
            