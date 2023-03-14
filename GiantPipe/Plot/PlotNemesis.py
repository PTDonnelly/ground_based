import glob
from math import log
import matplotlib.pyplot as plt
import natsort
import numpy as np
import os
from pathlib import Path
from typing import List
from Read.ReadNemesisOutputs import ReadNemesisOutputs as nem

class PlotNemesis:

    def __init__():
        return
    
    @staticmethod
    def radiance_to_brightness_temperature(wavenumbers, radiance):
        # Define constants
        h = 6.626e-34
        c = 2.9979e8
        k =1.3806e-23
        # Balance units
        v = [wave * 100 for wave in wavenumbers] # cm-1 -> m-1
        spec = [(r * 1e-9) * 1e4 / 100 for r in radiance]  # W/cm2/sr/cm-1 -> W/m2/sr/m-1
        # Construct equation
        c1 = 2 * h * c**2
        c2 = h * c / k
        a = [c1 * v[i]**3 / spec[i] for i, _ in enumerate(wavenumbers)]
        return [c2 * v[i] / (log(a[i] + 1)) for i, _ in enumerate(wavenumbers)]


    @classmethod
    def plot_single_spectrum(cls, icore: int, cores: List[str], mre: dict) -> None:
        # Create nests lists for containing data
        wavenumber = [[] for _, _ in enumerate(cores)]
        measured_radiance = [[] for _, _ in enumerate(cores)]
        measured_radiance_error = [[] for _, _ in enumerate(cores)]
        retrieved_radiance = [[] for _, _ in enumerate(cores)]
        latitudes = [[] for _, _ in enumerate(cores)]

        # Store data
        wavenumber[icore] = [float(value) for value in mre['spectrum'][1]]
        measured_radiance[icore] = [float(value) for value in mre['spectrum'][2]]
        measured_radiance_error[icore] = [float(value) for value in mre['spectrum'][3]]
        retrieved_radiance[icore] = [float(value) for value in mre['spectrum'][5]]
        latitudes[icore] = float(mre['latitude'])

        # Convert from radiance to brightness temperature
        measured_TB = cls.radiance_to_brightness_temperature(wavenumber[icore], measured_radiance[icore])
        retrieved_TB = cls.radiance_to_brightness_temperature(wavenumber[icore], retrieved_radiance[icore])

        figname = f"{dir}{latitudes[icore]}.png"
        _, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(6, 6), dpi=300)
        plt.suptitle(f"Latitude = {latitudes[icore]}")

        ax1.plot(wavenumber[icore], measured_radiance[icore], color='black', label='Observed')
        ax1.plot(wavenumber[icore], retrieved_radiance[icore], color='red', label='Retrieved')
        fill_neg = [radiance - error for radiance, error in zip(measured_radiance[icore], measured_radiance_error[icore])]
        fill_pos = [radiance + error for radiance, error in zip(measured_radiance[icore], measured_radiance_error[icore])]
        ax1.fill_between(wavenumber[icore], fill_neg, fill_pos, color='black', alpha=0.5)
        ax1.grid(axis='both', markevery=1, color='k', ls=':', lw=0.5)
        ax1.set_ylabel(r"Radiance (nw ...)")
        ax1.legend(loc='upper right')
        ax1.tick_params(axis='both', length=1, pad=1, labelsize=6)

        residual = [retrieved - measured for measured, retrieved in zip(measured_radiance[icore], retrieved_radiance[icore])]
        ax2.plot(wavenumber[icore], residual, color='black')
        
        ax2.fill_between(wavenumber[icore], [-1*err for err in measured_radiance_error[icore]], measured_radiance_error[icore], color='black', alpha=0.5)
        ax2.grid(axis='both', markevery=1, color='k', ls=':', lw=0.5)
        ax2.set_ylabel(r"$\Delta$ Radiance (nw ...)")
        ax2.tick_params(axis='both', length=1, pad=1, labelsize=6)

        ax3.plot(wavenumber[icore], measured_TB, color='black')
        ax3.plot(wavenumber[icore], retrieved_TB, color='red')
        ax3.grid(axis='both', markevery=1, color='k', ls=':', lw=0.5)
        ax3.set_ylabel(r"T$_{B}$ (K)")
        ax3.tick_params(axis='both', length=1, pad=1, labelsize=6)

        residual = [retrieved - measured for measured, retrieved in zip(measured_TB, retrieved_TB)]
        ax4.plot(wavenumber[icore], residual, color='black')
        ax4.grid(axis='both', markevery=1, color='k', ls=':', lw=0.5)
        ax4.set_ylabel(r"$\Delta$ T$_{B}$ (K)")
        ax4.set_xlabel(r"Wavenumber (cm$^{-1}$)")
        ax4.tick_params(axis='both', length=1, pad=1, labelsize=6)

        plt.xlabel(r"Wavenumber (cm$^{-1}$)")
        plt.xticks(wavenumber[icore])
        
        plt.savefig(figname, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    @classmethod
    def plot_ctl_spectrum(cls, icore: int, cores: List[str], mre: dict) -> None:
        return

    @classmethod
    def plot_spectrum_with_latitude(cls):
        experiment = "/Users/ptdonnelly/Documents/Research/projects/nemesis_centre_to_limb/retrievals/experiment_2_maximum_emission_angle/"
        tests = ["limb_50_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/"
                "limb_55_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
                "limb_60_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
                "limb_65_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
                "limb_70_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
                "limb_75_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
                "limb_80_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
                "limb_85_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
                "limb_90_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/"
                ]

        for test in tests:
            # If subdirectory does not exist, create it
            dir = f'{experiment}spectra/{test}'
            if not os.path.exists(dir):
                os.makedirs(dir)

            cores = natsort.natsorted(glob.glob(f"{experiment}{test}core_*"))

            for icore, core in enumerate(cores):
                # Read .mre file
                mre = nem.read_mre(f"{core}/nemesis.mre")

                if mre['ngeom'] == 1:
                    cls.plot_single_spectrum(icore, cores, mre)
                else:
                    cls.plot_ctl_spectrum(icore, cores, mre)



        return
