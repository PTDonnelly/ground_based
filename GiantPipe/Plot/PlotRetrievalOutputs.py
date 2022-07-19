import os
import numpy as np
import matplotlib.pyplot as plt
from Read.ReadPrior import ReadTemperaturePriorProfile
from Read.ReadRetrievalOutputFiles import ReadprfFiles

def PlotRetrievedTemperature():

    print('Plotting NEMESIS retrieved temperature over latitude...')
    # If subdirectory does not exist, create it
    dir = '../retrievals/retrieved_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/"
    retrieval_test = ["jupiter_v2021_temp_aerosol1-1mu-800mbar-05scale-01_C2H2-C2H6_NH3_retrieval"]
    ntest = len(retrieval_test)

    for itest in retrieval_test:
        temperature, _, latitude, _, pressure, _, nlevel, _ = ReadprfFiles(f"{fpath}{itest}")

        pressure_level = 0.57424E-02 
        ind = nlevel - np.searchsorted(pressure[::-1], pressure_level)

        fig, axes = plt.subplots(1, 1, figsize=(10, 6), sharex=True, sharey=True)        
        axes.plot(latitude, temperature[ind, :].T, lw=3)
        axes.grid()
        #axes.legend(loc="upper center", fontsize=10, handletextpad=0, handlelength=0, markerscale=0)
        axes.tick_params(labelsize=15)
        # Add a big axis 
        plt.axes([0.1, 0.09, 0.9, 0.8], frameon=False) 
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)        
        plt.ylabel(f"Retrieved Temperature at {int(pressure_level)} mbar", size=20)
        plt.xlabel(f"Planetocentric Latitude", size=20)
        # Save figure in the retrievals outputs directory
        plt.savefig(f"{dir}{itest}_retrieved_temperature_{pressure_level}mbar.png", dpi=300)
        plt.savefig(f"{dir}{itest}_retrieved_temperature_{pressure_level}mbar.eps", dpi=300)
        # Clear figure to avoid overlapping between plotting subroutines
        plt.clf()

def PlotRetrievedTemperatureProfile():

    print('Plotting NEMESIS retrieved temperature profiles...')
    # If subdirectory does not exist, create it
    dir = '../retrievals/retrieved_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    fpath = "/Users/db496/Documents/Research/Observations/NEMESIS_outputs/"
    prior = ['jupiter_v2021', 'jupiter_v2016']
    retrieval_test = [f"{prior[0]}_temp_aerosol1-1mu-800mbar-05scale-01_C2H2-C2H6_NH3_retrieval"]
    ntest = len(retrieval_test)

    # Read profile data from NEMESIS prior file 
    _, prior_p, prior_temperature, _, _, _, _ = ReadTemperaturePriorProfile(f"{fpath}{prior[0]}.prf")

    for itest in retrieval_test:
        temperature, gases, latitude, height, pressure, nlat, nlevel, ngas = ReadprfFiles(f"{fpath}{itest}")
        for ilat in range(nlat):
            fig, axes = plt.subplots(1, 1, figsize=(7, 10), sharex=True, sharey=True)        
            axes.plot(temperature[:, ilat], pressure, lw=3, label=f"Retrieved Temperature at {latitude[ilat]}")
            axes.plot(prior_temperature, prior_p, lw=3, label=f"{prior[0]}")
            axes.set_yscale('log')
            axes.invert_yaxis()
            axes.grid()
            axes.legend(loc="upper right", fontsize=10)
            axes.tick_params(labelsize=15)
            # Add a big axis 
            plt.axes([0.1, 0.09, 0.9, 0.8], frameon=False) 
            plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)        
            plt.xlabel(f"Retrieved Temperature [K]", size=20)
            plt.ylabel(f"Presssure [atm]", size=20)
            # Save figure in the retrievals outputs directory
            plt.savefig(f"{dir}{itest}_retrieved_temperature_profile_at_{latitude[ilat]}.png", dpi=300)
            plt.savefig(f"{dir}{itest}_retrieved_temperature_profile_at_{latitude[ilat]}.eps", dpi=300)
            # Clear figure to avoid overlapping between plotting subroutines
            plt.clf()

