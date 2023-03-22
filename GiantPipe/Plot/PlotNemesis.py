import glob
from math import log
import matplotlib.pyplot as plt
import natsort
import numpy as np
import numpy.typing as npt
import os
from pathlib import Path
from typing import List, Tuple
import snoop
from Read.ReadNemesisOutputs import ReadNemesisOutputs as nem

class PlotNemesis:

    def __init__():
        return
    
    @staticmethod
    def make_new_dir(dir: str) -> None:
        if not os.path.exists(dir):
            os.makedirs(dir)
        return dir
    
    @staticmethod
    def radiance_to_brightness_temperature(wavenumbers_cm: npt.ArrayLike, radiance_cm: npt.ArrayLike) -> npt.ArrayLike:
        
        # Define constants
        h = 6.626e-34
        c = 2.9979e8
        k =1.3806e-23

        # Balance units
        wavenumbers_m = np.multiply(wavenumbers_cm, 100)
        radiance_conversion = (1e-9 * 1e4) / 100
        radiance_m = np.multiply(radiance_cm, radiance_conversion)
        
        # Construct equation
        constant_1 = 2 * h * np.power(c, 2)
        constant_2 = h * c / k
        wavenumbers_m_cubed = np.power(wavenumbers_m, 3)
        wavenumber_m_cubed_over_radiance_m = np.divide(wavenumbers_m_cubed, radiance_m)
        a = np.multiply(constant_1, wavenumber_m_cubed_over_radiance_m)
        log_a = np.log(a + 1)
        wavenumber_m_over_log_a = np.divide(wavenumbers_m, log_a)
        return np.multiply(wavenumber_m_over_log_a, constant_2)

    @classmethod
    def plot_single_spectrum(cls, dir: str, mre: dict, spx: dict) -> None:

        # Read spectrum (remove the extra surrounding list from ngeom = 1)
        spectrum = mre['spectrum'].pop()
        wavenumber = [float(value) for value in spectrum[1]]
        measured_radiance = [float(value) for value in spectrum[2]]
        measured_radiance_error = [float(value) for value in spectrum[3]]
        retrieved_radiance = [float(value) for value in spectrum[5]]
        latitude = float(mre['latitude'])

        # Convert from radiance to brightness temperature
        measured_TB = cls.radiance_to_brightness_temperature(wavenumber, measured_radiance)
        retrieved_TB = cls.radiance_to_brightness_temperature(wavenumber, retrieved_radiance)

        figname = f"{dir}{latitude}.png"
        _, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(6, 6), dpi=300)
        plt.suptitle(f"Latitude = {latitude}")

        ax1.plot(wavenumber, measured_radiance, color='black', label='Observed')
        ax1.plot(wavenumber, retrieved_radiance, color='red', label='Retrieved')
        fill_neg = [radiance - error for radiance, error in zip(measured_radiance, measured_radiance_error)]
        fill_pos = [radiance + error for radiance, error in zip(measured_radiance, measured_radiance_error)]
        ax1.fill_between(wavenumber, fill_neg, fill_pos, color='black', alpha=0.5)
        ax1.grid(axis='both', markevery=1, color='k', ls=':', lw=0.5)
        ax1.grid(axis='both', markevery=1, color='k', ls=':', lw=0.5)
        ax1.grid(axis='both', markevery=1, color='k', ls=':', lw=0.5)
        ax1.grid(axis='both', markevery=1, color='k', ls=':', lw=0.5)
        ax1.grid(axis='both', markevery=1, color='k', ls=':', lw=0.5)
        ax1.grid(axis='both', markevery=1, color='k', ls=':', lw=0.5)
        ax1.set_ylabel(r"Radiance (nw ...)")
        ax1.legend(loc='upper right')
        ax1.tick_params(axis='both', length=1, pad=1, labelsize=6)

        residual = [retrieved - measured for measured, retrieved in zip(measured_radiance, retrieved_radiance)]
        ax2.plot(wavenumber, residual, color='black')
        
        ax2.fill_between(wavenumber, [-1*err for err in measured_radiance_error], measured_radiance_error, color='black', alpha=0.5)
        ax2.grid(axis='both', markevery=1, color='k', ls=':', lw=0.5)
        ax2.set_ylabel(r"$\Delta$ Radiance (nw ...)")
        ax2.tick_params(axis='both', length=1, pad=1, labelsize=6)

        ax3.plot(wavenumber, measured_TB, color='black')
        ax3.plot(wavenumber, retrieved_TB, color='red')
        ax3.grid(axis='both', markevery=1, color='k', ls=':', lw=0.5)
        ax3.set_ylabel(r"T$_{B}$ (K)")
        ax3.tick_params(axis='both', length=1, pad=1, labelsize=6)

        residual = [retrieved - measured for measured, retrieved in zip(measured_TB, retrieved_TB)]
        ax4.plot(wavenumber, residual, color='black')
        ax4.grid(axis='both', markevery=1, color='k', ls=':', lw=0.5)
        ax4.set_ylabel(r"$\Delta$ T$_{B}$ (K)")
        ax4.set_xlabel(r"Wavenumber (cm$^{-1}$)")
        ax4.tick_params(axis='both', length=1, pad=1, labelsize=6)

        plt.xlabel(r"Wavenumber (cm$^{-1}$)")
        plt.xticks(wavenumber)
        plt.savefig(figname, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    @classmethod
    def plot_ctl_spectrum(cls, dir: str, mre: dict, spx: dict) -> None:
        
        # Read spectra from mrefile, reshape nested lists from (ngeom, nconv) to (nconv, ngeom) and broadcast as numpy array
        ngeom, ny = mre['ngeom'], mre['ny']
        wavenumber = np.asarray([value for value in zip(*[mre['spectrum'][igeom][1] for igeom in range(ngeom)])], dtype=float)
        measured_radiance = np.asarray([value for value in zip(*[mre['spectrum'][igeom][2] for igeom in range(ngeom)])], dtype=float)
        measurement_error = np.asarray([value for value in zip(*[mre['spectrum'][igeom][3] for igeom in range(ngeom)])], dtype=float)
        retrieved_radiance = np.asarray([value for value in zip(*[mre['spectrum'][igeom][5] for igeom in range(ngeom)])], dtype=float)
        latitude = float(mre['latitude'])
        
        # Read geometry information from spxfile
        emission_angles = np.asarray([spx['angles'][igeom][3] for igeom in range(ngeom)], dtype=float)
        nconvs = np.asarray([spx['nconv'][igeom] for igeom in range(ngeom)], dtype=int)

        # Convert from radiance to brightness temperature
        measured_TB = cls.radiance_to_brightness_temperature(wavenumber, measured_radiance)
        retrieved_TB = cls.radiance_to_brightness_temperature(wavenumber, retrieved_radiance)

        # Define common plotting parameters
        fontsize = 7
        linewidth = 1
        n_plots_per_filter = 3
        nconv = int(ny / ngeom)
        savepng, savepdf = True, False
        
        # Set up figure
        figname = f"{dir}{latitude}.png"
        _, axes = plt.subplots(nrows=nconv, ncols=3, sharex=True, figsize=(12, nconv), dpi=300)
        plt.suptitle(f"Latitude = {latitude}")
        
        # Loop over each axis object
        for i, ax in enumerate(np.transpose(axes.flat)):
            iconv = int(i / n_plots_per_filter)
            icol = int(i % n_plots_per_filter)
            title = wavenumber[iconv][0]
            xmin, xmax = min(emission_angles), max(emission_angles)+0.5
            xticks = np.flipud(np.arange(xmax, xmin, -10))
            xlabel = r'Emission Angle ($^{\circ}$)'

            # Plot each column in figure (i % n_plots_per_filter = 0, 1, or 2)
            if (icol == 0):
                # Plot CTL profile of radiance
                ax.fill_between(emission_angles, measured_radiance[iconv, :]-measurement_error[iconv, :], measured_radiance[iconv, :]+measurement_error[iconv, :], color='black', alpha=0.3)
                ax.plot(emission_angles, measured_radiance[iconv, :], color='black', lw=1.5, label='Observed')
                ax.plot(emission_angles, retrieved_radiance[iconv, :], color='red', lw=1.5, label='Retrieved')
                # Define axis parameters
                ymin, ymax = np.min((measured_radiance[iconv, :], retrieved_radiance[iconv, :])), 1.1*(np.max((measured_radiance[iconv, :], retrieved_radiance[iconv, :])))
                ylabel = "R"
                tx, ty = xmin + 0.05*(xmax-xmin), ymin + 0.3*(ymax-ymin)
                # Add filter wavenumber
                ax.text(tx, ty, int(title), fontsize=fontsize+1, fontweight='bold', bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round'))
            elif (icol == 1):
                # Plot CTL profile of residual radiance (fit minus measurement)
                ax.fill_between(emission_angles, -measurement_error[iconv, :], +measurement_error[iconv, :], color='black', alpha=0.3)
                ax.plot(emission_angles, retrieved_radiance[iconv, :] - measured_radiance[iconv, :], color='black', lw=1.5, label='Observed')
                # Define axis parameters
                ymin, ymax = -1.5*(np.max(measurement_error[iconv, :])), 1.5*(np.max(measurement_error[iconv, :]))
                ylabel = r"$\Delta$ R"
                # Add zero line
                ax.plot((xmin, xmax), (0, 0), lw=0.75, color='black', zorder=1)
            elif (icol == 2):
                # Plot CTL profile of brightness temperature
                ax.plot(emission_angles, retrieved_TB[iconv] - measured_TB[iconv], color='black', lw=1.5, label='Observed')
                # Define axis parameters
                ymin, ymax = -2, 2
                yticks = np.arange(ymin, ymax+0.1, 1)
                ylabel = r"$\Delta$ T$_{B}$"
                # Add zero line
                ax.plot((xmin, xmax), (0, 0), lw=0.75, color='black', zorder=1)

            # Clean up axes
            ax.grid(axis='both', markevery=1, color='k', ls=':', lw=0.5*linewidth)
            ax.set_xlim((xmin, xmax))
            ax.set_ylim((ymin, ymax))
            ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=1)
            ax.set_xticks(xticks)
            if (icol == 2):
                ax.set_yticks(yticks)
            if (iconv == nconv-1):
                ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=1)
            ax.tick_params(axis='both', length=1, pad=1, labelsize=fontsize)

        # Finish and close plot
        plt.subplots_adjust(hspace=0.1, wspace=0.2)  
        plt.savefig(figname, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    @classmethod
    def plot_superposed_ctl_spectrum(cls, dir: str, mre_data: dict, spx_data: dict) -> None:
    
        # Define common plotting parameters
        fontsize = 7
        linewidth = 1
        n_plots_per_filter = 3
        ntests = len(mre_data)
        ny = float(mre_data[0]['ny'])
        ngeom = int(mre_data[0]['ngeom'])
        nconv = int(ny / ngeom)
        savepng, savepdf = True, False
        
        # Set up figure
        latitude = float(mre_data[0]['latitude'])
        figname = f"{dir}{latitude}.png"
        _, axes = plt.subplots(nrows=nconv, ncols=3, sharex=True, figsize=(12, nconv), dpi=300)
        cmap = plt.get_cmap('cividis')
        plt.suptitle(f"Latitude = {latitude}\n[0, -1p, 27s, 26s]")

        # Read spectra from mrefile, reshape nested lists from (ngeom, nconv) to (nconv, ngeom) and broadcast as numpy array
        wavenumber, measured_radiance, measurement_error, retrieved_radiance, emission_angles, measured_TB, retrieved_TB = [[] for _ in range(7)]
        for itest, (mre, spx) in enumerate(zip(mre_data, spx_data)):
            ngeom, ny = mre['ngeom'], mre['ny']
            wavenumber.append(np.asarray([value for value in zip(*[mre['spectrum'][igeom][1] for igeom in range(ngeom)])], dtype=float))
            measured_radiance.append(np.asarray([value for value in zip(*[mre['spectrum'][igeom][2] for igeom in range(ngeom)])], dtype=float))
            measurement_error.append(np.asarray([value for value in zip(*[mre['spectrum'][igeom][3] for igeom in range(ngeom)])], dtype=float))
            retrieved_radiance.append(np.asarray([value for value in zip(*[mre['spectrum'][igeom][5] for igeom in range(ngeom)])], dtype=float))
            latitude = float(mre['latitude'])
        
            # Read geometry information from spxfile
            emission_angles.append(np.asarray([spx['angles'][igeom][3] for igeom in range(ngeom)], dtype=float))
            nconvs = np.asarray([spx['nconv'][igeom] for igeom in range(ngeom)], dtype=int)

            # Convert from radiance to brightness temperature
            measured_TB.append(cls.radiance_to_brightness_temperature(wavenumber[itest], measured_radiance[itest]))
            retrieved_TB.append(cls.radiance_to_brightness_temperature(wavenumber[itest], retrieved_radiance[itest]))

        # Loop over each axis object
        for i, ax in enumerate(np.transpose(axes.flat)):                         
            iconv = int(i / n_plots_per_filter)
            icol = int(i % n_plots_per_filter)
            xmin = min([min(angles) for angles in emission_angles])
            xmax = max([max(angles) for angles in emission_angles])
            xticks = np.flipud(np.arange(xmax, xmin, -10))
            xlabel = r'Emission Angle ($^{\circ}$)'

            # Plot each column in figure (i % n_plots_per_filter = 0, 1, or 2)
            if (icol == 0):
                for itest in range(ntests):
                    # Plot CTL profile of radiance
                    ax.fill_between(emission_angles[itest], measured_radiance[itest][iconv, :]-measurement_error[itest][iconv, :], measured_radiance[itest][iconv, :]+measurement_error[itest][iconv, :], color='black', alpha=0.3)
                    ax.plot(emission_angles[itest], measured_radiance[itest][iconv, :], color='black', lw=1.5, label='Observed')
                    col = cmap(itest / ntests)
                    ax.plot(emission_angles[itest], retrieved_radiance[itest][iconv, :], color=col, lw=1.5, label='Retrieved')
                # Define axis parameters
                values = [(meas[iconv, :], retr[iconv, :]) for (meas, retr) in zip(measured_radiance, retrieved_radiance)]
                ymin, ymax = 0.95*np.min([np.min(v) for v in values]), 1.05*np.max([np.max(v) for v in values])
                ylabel = "R"
                tx, ty = xmin + 0.05*(xmax-xmin), ymin + 0.3*(ymax-ymin)
                wave = wavenumber[0][iconv][0]
                # Add filter wavenumber
                ax.text(tx, ty, int(wave), fontsize=fontsize+1, fontweight='bold', bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round'))
            elif (icol == 1):
                for itest in range(ntests):
                    # Plot CTL profile of residual radiance (fit minus measurement)
                    ax.fill_between(emission_angles[itest], -measurement_error[itest][iconv, :], +measurement_error[itest][iconv, :], color='black', alpha=0.3)
                    col = cmap(itest / ntests)
                    ax.plot(emission_angles[itest], retrieved_radiance[itest][iconv, :] - measured_radiance[itest][iconv, :], color=col, lw=1.5)
                # Define axis parameters
                values = [err[iconv, :] for err in measurement_error]
                ymin, ymax = -1.5*np.max([np.max(v) for v in values]), 1.5*np.max([np.max(v) for v in values])
                ylabel = r"$\Delta$ R"
                # Add zero line
                ax.plot((xmin, xmax), (0, 0), lw=0.75, color='black', zorder=1)
            elif (icol == 2):
                for itest in range(ntests):
                    col = cmap(itest / ntests)
                    # Plot CTL profile of brightness temperature
                    ax.plot(emission_angles[itest], retrieved_TB[itest][iconv] - measured_TB[itest][iconv], color=col, lw=1.5)
                # Define axis parameters
                ymin, ymax = -2, 2
                yticks = np.arange(ymin, ymax+0.1, 1)
                ylabel = r"$\Delta$ T$_{B}$"
                # Add zero line
                ax.plot((xmin, xmax), (0, 0), lw=0.75, color='black', zorder=1)

            # Clean up axes
            ax.grid(axis='both', markevery=1, color='k', ls=':', lw=0.5*linewidth)
            ax.set_xlim((xmin, xmax))
            ax.set_ylim((ymin, ymax))
            ax.set_ylabel(ylabel, fontsize=fontsize, labelpad=1)
            ax.set_xticks(xticks)
            if (icol == 2):
                ax.set_yticks(yticks)
            if (iconv == nconv-1):
                ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=1)
            ax.tick_params(axis='both', length=1, pad=1, labelsize=fontsize)

        # Finish and close plot
        plt.subplots_adjust(hspace=0.1, wspace=0.2)  
        plt.savefig(figname, dpi=300, bbox_inches='tight')
        plt.close()
        return

    @classmethod
    def plot_superposed_temperatures(cls, dir: str, mre_data: dict, spx_data: dict) -> None:
        
        # Define common plotting parameters
        fontsize = 7
        linewidth = 1
        ntests = len(mre_data)
        ny = float(mre_data[0]['ny'])
        ngeom = int(mre_data[0]['ngeom'])
        nconv = int(ny / ngeom)
        savepng, savepdf = True, False
        
        # Set up figure
        latitude = float(mre_data[0]['latitude'])
        figname = f"{dir}{latitude}.png"
        _, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(5, 5), gridspec_kw={'width_ratios': [2, 1]}, dpi=300)
        cmap = plt.get_cmap('cividis')
        plt.suptitle(f"Latitude = {latitude}\n[0, -1p, 27s, 26s]", fontsize=fontsize)

        # Read temperature profile from mrefile, reshape nested lists from (npro, nparams) to (nparams, npro) and broadcast as numpy array
        temperature_prior, temperature_prior_err, temperature_retrieved, temperature_retrieved_error = [[] for _ in range(4)]
        pressure = np.arange(120)
        for mre in mre_data:
            profiles = np.asarray([value for value in zip(*mre['profiles'][0])], dtype=float)
            temperature_prior.append(profiles[0])
            temperature_prior_err.append(profiles[1])
            temperature_retrieved.append(profiles[2])
            temperature_retrieved_error.append(profiles[3])
            latitude = float(mre['latitude'])

        # Loop over each axis object
        for i, ax in enumerate(np.transpose(axes.flat)):                         
            ymin, ymax = 0, 120
            ylabel = 'Pressure (atm)'
            if (i == 0):
                # Plot a priori temperature profile
                ax.fill_betweenx(pressure, temperature_prior[0]-temperature_prior_err[0], temperature_prior[0]+temperature_prior_err[0], color='black', alpha=0.3)
                ax.plot(temperature_prior[0], pressure, color='black', label='Prior')
                # Loop over each test
                for itest in range(ntests):
                    col = cmap(itest / ntests)
                    ax.plot(temperature_retrieved[itest], pressure, color=col, label='Retrieved')
                # Define axis parameters
                xmin, xmax = 90, 390
                xticks = np.arange(xmin, xmax+1, 60)
                xlabel = "Temperature (K)"
            elif (i == 1):
                # Plot residual a priori temperature profile
                ax.fill_betweenx(pressure, -temperature_prior_err[0], temperature_prior_err[0], color='black', alpha=0.3)
                ax.plot([0]*len(temperature_prior[0]), pressure, color='black', label='Prior')
                # Loop over each test
                for itest in range(ntests):
                    col = cmap(itest / ntests)
                    ax.plot(temperature_retrieved[itest]-temperature_prior[0], pressure, color=col, label='Retrieved')
                # Define axis parameters
                xmin, xmax = -30, 30
                xticks = np.arange(xmin, xmax+1, 15)
                xlabel = r"$\Delta$Temperature (K)"
            
            # Clean up axes
            ax.grid(axis='both', markevery=1, color='k', ls=':', lw=0.5*linewidth)
            ax.set_xlim((xmin, xmax))
            ax.set_ylim((ymin, ymax))
            ax.set_xlabel(xlabel, fontsize=fontsize, labelpad=1)
            ax.set_xticks(xticks)
            if (i == 0):
                ax.set_xlabel(ylabel, fontsize=fontsize, labelpad=1)
            ax.tick_params(axis='both', length=1, pad=1, labelsize=fontsize)

        # Finish and close plot
        plt.subplots_adjust(hspace=0.1, wspace=0.2)  
        plt.savefig(figname, dpi=300, bbox_inches='tight')
        plt.close()
        return

    @staticmethod
    def get_retrievals() -> Tuple[str, list]:
        # experiment = "/Users/ptdonnelly/Documents/Research/projects/nemesis_centre_to_limb/retrievals/experiment_1_initial_tests/"
        # tests = [
        #         "ctl_flat5_jupiter2021_nh3cloud_0_1p/",
        #         "ctl_flat5_jupiter2021_nh3cloud_0_1p_27s/",
        #         "ctl_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/"
        #         ]
        
        # experiment = "/Users/ptdonnelly/Documents/Research/projects/nemesis_centre_to_limb/retrievals/experiment_2_maximum_emission_angle/"
        # tests = [
        #         "limb_50_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/"
        #         "limb_55_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         "limb_60_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         "limb_65_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         "limb_70_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         "limb_75_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         "limb_80_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         "limb_85_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         "limb_90_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/"
        #         ]
        
        experiment = "/Users/ptdonnelly/Documents/Research/projects/nemesis_centre_to_limb/retrievals/experiment_3_characterise_ctl_profile_bins_maxmu/"
        tests = [
                "bins_maxmu_0_75_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
                "bins_maxmu_5_75_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
                "bins_maxmu_10_75_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
                "bins_maxmu_15_75_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
                "bins_maxmu_20_75_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
                "bins_maxmu_25_75_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
                "bins_maxmu_30_75_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
                "bins_maxmu_35_75_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
                "bins_maxmu_40_75_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
                "bins_maxmu_45_75_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
                # "bins_maxmu_50_75_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
                # "bins_maxmu_55_75_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
                # "bins_maxmu_60_75_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
                "bins_maxmu_65_75_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
                "bins_maxmu_70_75_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/"
                ]
        
        # experiment = "/Users/ptdonnelly/Documents/Research/projects/nemesis_centre_to_limb/retrievals/experiment_4_characterise_ctl_profile_bins_fixedwidth/"
        # tests = [
        #         "bins_fixedwidth_5_10_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         "bins_fixedwidth_10_15_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         "bins_fixedwidth_15_20_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         "bins_fixedwidth_20_25_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         "bins_fixedwidth_25_30_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         "bins_fixedwidth_30_35_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         # "bins_fixedwidth_35_40_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         "bins_fixedwidth_40_45_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         "bins_fixedwidth_45_50_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         "bins_fixedwidth_50_55_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         "bins_fixedwidth_55_60_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         "bins_fixedwidth_60_65_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         "bins_fixedwidth_65_70_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         "bins_fixedwidth_70_75_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/"
        #         ]
        
        return experiment, tests

    @classmethod
    def plot_spectrum_with_latitude(cls) -> None:
        """Plots the measured and retrieved spectra from a single test in a given experiment."""

        # Get filepaths of retrieval outputs
        experiment, tests = cls.get_retrievals()

        # Loop over each test within the experiment
        for test in tests:
            
            # Get subdirectory, if it does not exist, create it - format argument as you like
            dirname = "spectra_tests/"
            dir = cls.make_new_dir(f"{experiment}{dirname}{test}")
                   
            # Point to individual retrieval outputs
            cores = natsort.natsorted(glob.glob(f"{experiment}{test}core*"))
            
            # Loop over each core
            for core in cores:
                
                # Read .mre file
                mre = nem.read_mre(f"{core}/nemesis.mre")

                # Read .spx file
                spx = nem.read_spx(f"{core}/nemesis.spx")
                
                # Plot spectra
                if mre['ngeom'] == 1:
                    cls.plot_single_spectrum(dir, mre, spx)
                else:
                    cls.plot_ctl_spectrum(dir, mre, spx)
        return
    
    @classmethod
    def plot_superposed_results_with_latitude(cls) -> None:
        """Plots the measured and retrieved spectra from multiple tests in a given experiment
        on the same axes for comparison."""

        # Get filepaths of retrieval outputs
        experiment, tests = cls.get_retrievals()

        # Get subdirectory, if it does not exist, create it
        mode = 'temperature'
        dirname = f"{mode}_experiment/"
        dir = cls.make_new_dir(f"{experiment}{dirname}")

        # Loop over each core
        core_numbers = np.arange(1, 147, 1)
        for core_number in core_numbers:
            # Find this core in multiple tests within this experiment
            mre_data, spx_data = [], []
            for test in tests:
                
                # Point to individual retrieval outputs
                core = glob.glob(f"{experiment}{test}core_{core_number}").pop()
                    
                # Read .mre file
                mre_data.append(nem.read_mre(f"{core}/nemesis.mre"))

                # Read .spx file
                spx_data.append(nem.read_spx(f"{core}/nemesis.spx"))
                
            if mode == 'spectra':
                if mre_data[0]['ngeom'] == 1:
                    cls.plot_single_spectrum(dir, mre_data, spx_data)
                else:
                    cls.plot_superposed_ctl_spectrum(dir, mre_data, spx_data)
            elif mode =='temperature':
                cls.plot_superposed_temperatures(dir, mre_data, spx_data)
            # exit()
        return
    