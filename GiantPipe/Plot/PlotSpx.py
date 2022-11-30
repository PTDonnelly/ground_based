import os
import numpy as np
import matplotlib.pyplot as plt
import Globals
from Read.ReadSpxFiles import ReadSpxFiles


def PlotSpxMerid(dataset):
    """ Plot radiances contains into .spx files for checking """

    # Set the path to the .spx files created by GiantPipe
    dir = f'/Users/db496/Documents/Research/Observations/ground_based_structure_branch/outputs/{dataset}/spxfiles/'

    # If subdirectory does not exist, create it
    subdir = f'{dir}/spx_figures/'
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    print('Plotting .spx files before NEMESIS retrieval...')
    
        # Read the spectral information and store in dedicated arrays
    radiance, rad_err, wavenumb, latgrid = ReadSpxFiles(dir)
    
    # Loop over each filter to plot the meridian profile of radiance
    for ifilt in range(Globals.nfilters):
        fig, axes = plt.subplots(1, 1, figsize=(10, 7), sharex=True, sharey=True)        
        axes.plot(latgrid, radiance[ifilt, :], lw=2, label=f"spxfile Radiance at {wavenumb[ifilt]}", color='orange')
        axes.fill_between(latgrid, radiance[ifilt, :]-rad_err[ifilt, :], radiance[ifilt, :]+rad_err[ifilt, :], color='orange', alpha=0.2)
        axes.grid()
        axes.legend(loc="upper right", fontsize=15)
        axes.tick_params(labelsize=15)
        # Add a big axis 
        plt.axes([0.1, 0.09, 0.8, 0.8], frameon=False) 
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)        
        plt.xlabel("Latitude", size=20)
        plt.ylabel("Radiance [nW cm$^{-2}$ sr$^{-1}$ cm]", size=20)
        # Save figure in the retrievals outputs directory
        plt.savefig(f"{subdir}spx_merid_radiance_at_{wavenumb[ifilt]}.png", dpi=150, bbox_inches='tight')
        # Close figure to avoid overlapping between plotting subroutines
        plt.close()