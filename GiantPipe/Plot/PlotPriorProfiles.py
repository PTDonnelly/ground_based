import os
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from Read.ReadPrior import ReadTemperatureGasesPriorProfile, ReadAerosolPriorProfile

def PlotTemperaturePriorProfiles():

    print('Plotting NEMESIS Temperature and Gases prior profiles...')
    # If subdirectory does not exist, create it
    dir = '../retrievals/prior_profiles/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    fpath = '/Users/db496/Documents/Research/Observations/NEMESIS_outputs/'
    filename = ['jupiter_v2021', 'jupiter_v2016']
    for fname in filename:
        # Read profile data from NEMESIS prior file 
        _, pressure, temperature, gas, gasname, nlevel, ngas = ReadTemperatureGasesPriorProfile(f"{fpath}{fname}.prf")

        # Create a figure
        fig, axes = plt.subplots(1, ngas+1, figsize=(12,6), sharey=True)
        # Temperature profile subplot
        axes[0].plot(temperature, pressure, color='black', lw=0, marker='.', markersize=2)
        axes[0].set_yscale('log')
        axes[0].invert_yaxis()
        axes[0].set_xlabel("Temperature [K]", size=15)
        axes[0].tick_params(labelsize=10)
        # Gases profiles subplots
        for igas in range(ngas):
            axes[igas+1].plot(gas[:, igas], pressure, lw=0, marker='.', markersize=2, label=f"{gasname[igas]}")
            axes[igas+1].legend(loc="upper center", fontsize=10, handletextpad=0, handlelength=0, markerscale=0)
            axes[igas+1].tick_params(labelsize=10)
        # Add a big axis 
        plt.axes([0.1, 0.09, 0.9, 0.8], frameon=False) 
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)        
        plt.ylabel("Pressure [mbar]", size=15)
        plt.xlabel("Volume Mixing Ratio", size=15)
        plt.title(f"{fname}", size=15)
        # Save figure in the retrievals outputs directory
        plt.savefig(f"{dir}{fname}_prior_profiles.png", dpi=150, bbox_inches='tight')
        plt.savefig(f"{dir}{fname}_prior_profiles.eps", dpi=150, bbox_inches='tight')
        # Clear figure to avoid overlapping between plotting subroutines
        plt.clf()
        
def PlotAerosolPriorProfiles():

    print('Plotting NEMESIS Aerosol prior profiles...')
    # If subdirectory does not exist, create it
    dir = '../retrievals/prior_profiles/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    filename = ['1mu_800mbar_05scale_01', '5mu_800mbar_05scale_01', '10mu_800mbar_05scale_01']
    nfiles = len(filename)
    # Create a composite figure
    fig, axes = plt.subplots(1, nfiles, figsize=(7,8), sharex=True, sharey=True)
    for ifiles in range(nfiles):
        # Read profile data from NEMESIS prior file
        aerosol, altitude, _, _ = ReadAerosolPriorProfile(f"../../NEMESIS_outputs/{filename[ifiles]}.prf")
        # Subplot of the current file
        axes[ifiles].plot(aerosol, altitude, lw=0, marker='.', markersize=2, label=f"{filename[ifiles]}")
        axes[ifiles].legend(loc="upper center", fontsize=10, handletextpad=0, handlelength=0, markerscale=0)
        axes[ifiles].tick_params(labelsize=15)
        # Empty aerosol array to avoid overlapping during the loop process
        aerosol = []
    # Add a big axis 
    plt.axes([0.1, 0.09, 0.9, 0.8], frameon=False) 
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)        
    plt.ylabel("Height [km]", size=20)
    plt.xlabel("?", size=20)
    # Save figure in the retrievals outputs directory
    plt.savefig(f"{dir}aerosol_prior_profiles.png", dpi=150, bbox_inches='tight')
    plt.savefig(f"{dir}aerosol_prior_profiles.eps", dpi=150, bbox_inches='tight')
    # Clear figure to avoid overlapping between plotting subroutines
    plt.clf()

    # Create a figure per file
    for fname in filename:
        # Read profile data from NEMESIS prior file
        aerosol, altitude, ncloud, _ = ReadAerosolPriorProfile(f"../../NEMESIS_outputs/{fname}.prf")
        fig, axes = plt.subplots(1, ncloud, figsize=(7,10), sharey=True)
        if ncloud > 1:
            for icloud in range(ncloud):
                axes[icloud].plot(aerosol, altitude, lw=0, marker='.', markersize=2)
                axes[icloud].tick_params(labelsize=15)
        else:
            axes.plot(aerosol, altitude, lw=0, marker='.', markersize=2)
            axes.tick_params(labelsize=15)
        # Add a big axis 
        plt.axes([0.1, 0.09, 0.9, 0.8], frameon=False) 
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)        
        plt.ylabel("Height [km]", size=20)
        plt.xlabel("?", size=20)
        plt.title(f"{fname}", size=20)
        # Save figure in the retrievals outputs directory
        plt.savefig(f"{dir}{fname}_prior_profiles.png", dpi=150, bbox_inches='tight')
        plt.savefig(f"{dir}{fname}_prior_profiles.eps", dpi=150, bbox_inches='tight')
        # Clear figure to avoid overlapping between plotting subroutines
        plt.clf()
        


