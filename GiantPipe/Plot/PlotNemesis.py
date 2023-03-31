import glob
from math import log
import matplotlib.pyplot as plt
# import matplotlib.colors as mc
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
    def plot_xsc(cls, dir: str, xsc_data: dict) -> None:
        # Define common plotting parameters
        fontsize = 7
        linewidth = 1
        ntests = len(xsc_data)
        
        # Set up figure
        figname = f"{dir}aerosol_scattering_parameters.png"
        _, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(4, 2), dpi=300)
        cmap = plt.get_cmap('jet')
        plt.suptitle(f"Aerosol Cross-section", fontsize=fontsize)

        # Read temperature profile from mrefile, reshape nested lists from (npro, nparams) to (nparams, npro) and broadcast as numpy array
        wavenumber, cross_section, single_scattering_albedo = [], [], []
        for xsc in xsc_data:
            # Read spectrum (remove the extra surrounding list from ngeom = 1)
            wavenumber.append(np.asarray([float(value) for value in xsc['wavenumber']], dtype=float))
            cross_section.append(np.asarray([float(value) for value in xsc['cross_section']], dtype=float))
            single_scattering_albedo.append(np.asarray([float(value) for value in xsc['single_scattering_albedo']], dtype=float))

        # Loop over each axis object
        for i, ax in enumerate(np.transpose(axes.flat)):      
            if (i == 0):
                for itest in range(ntests):
                    ax.plot(wavenumber[itest], cross_section[itest], color='black', lw=0, marker='.', markersize='4')
            if (i == 1):
                for itest in range(ntests):
                    ax.plot(wavenumber[itest], single_scattering_albedo[itest], color=cmap(itest/ntests), label=f"Test {itest+1}")
            ax.grid(axis='both', markevery=20, color='k', ls=':', lw=0.5)
            # ax.set_ylabel(ylabel)
            ax.tick_params(axis='both', length=1, pad=1, labelsize=6)

        plt.xlabel(r"Wavenumber (cm$^{-1}$)")
        plt.xticks(wavenumber[0])
        plt.savefig(figname, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    @classmethod
    def plot_spectrum(cls, dir: str, mre_data: dict) -> None:
     
        # Define common plotting parameters
        fontsize = 7
        linewidth = 1
        ntests = len(mre_data)
        
        # Set up figure
        latitude = float(mre_data[0]['latitude'])
        figname = f"{dir}{latitude}.png"
        _, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(6, 6), dpi=300)
        cmap = plt.get_cmap('jet')
        plt.suptitle(f"Latitude = {latitude}", fontsize=fontsize)

        # Read temperature profile from mrefile, reshape nested lists from (npro, nparams) to (nparams, npro) and broadcast as numpy array
        wavenumber, measured_radiance, measurement_error, retrieved_radiance, measured_TB, retrieved_TB = [[] for _ in range(6)]
        for i, mre in enumerate(mre_data):
            # Read spectrum (remove the extra surrounding list from ngeom = 1)
            spectrum = mre['spectrum'].pop()
            wavenumber.append(np.asarray([float(value) for value in spectrum[1]], dtype=float))
            measured_radiance.append(np.asarray([float(value) for value in spectrum[2]], dtype=float))
            measurement_error.append(np.asarray([float(value) for value in spectrum[3]], dtype=float))
            retrieved_radiance.append(np.asarray([float(value) for value in spectrum[5]], dtype=float))

            # Convert from radiance to brightness temperature
            measured_TB.append(cls.radiance_to_brightness_temperature(wavenumber[i], measured_radiance[i]))
            retrieved_TB.append(cls.radiance_to_brightness_temperature(wavenumber[i], retrieved_radiance[i]))

        # Loop over each axis object
        for i, ax in enumerate(np.transpose(axes.flat)):      
            if (i == 0):
                for itest in range(ntests):
                    ax.fill_between(wavenumber[itest], measured_radiance[itest]-measurement_error[itest], measured_radiance[itest]+measurement_error[itest], color='black', alpha=0.25)
                    ax.plot(wavenumber[itest], measured_radiance[itest], color='black', lw=0, marker='.', markersize='4')
                    ax.plot(wavenumber[itest], retrieved_radiance[itest], color=cmap(itest/ntests), label=f"Test {itest+1}")
                    # print(retrieved_radiance[itest])
                ylabel = "Radiance (nw ...)"
                ax.legend(loc='upper right')
            elif (i == 1):
                for itest in range(ntests):
                    ax.fill_between(wavenumber[itest], [-1 * err for err in measurement_error[itest]], measurement_error[itest], color='black', alpha=0.25)
                    ax.plot(wavenumber[itest], retrieved_radiance[itest]-measured_radiance[itest], color=cmap(itest/ntests))
                ylabel = r"$\Delta$ Radiance (nw ...)"
            elif (i == 2):
                for itest in range(ntests):
                    ax.plot(wavenumber[itest], measured_TB[itest], color='black', lw=0, marker='.', markersize='4')
                    ax.plot(wavenumber[itest], retrieved_TB[itest], color=cmap(itest/ntests))
                ylabel = r"T$_{B}$ (K)"
            elif (i == 3):
                for itest in range(ntests):
                    ax.plot(wavenumber[itest], retrieved_TB[itest]-measured_TB[itest], color=cmap(itest/ntests))
                
                ylabel = r"$\Delta$ T$_{B}$ (K)"
                ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")

            ax.grid(axis='both', markevery=1, color='k', ls=':', lw=0.5)
            ax.set_ylabel(ylabel)
            ax.tick_params(axis='both', length=1, pad=1, labelsize=6)

        plt.xlabel(r"Wavenumber (cm$^{-1}$)")
        plt.xticks(wavenumber[0])
        plt.savefig(figname, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    @classmethod
    def plot_temperature(cls, dir: str, mre_data: dict) -> None:
        
        # Define common plotting parameters
        fontsize = 7
        linewidth = 1
        ntests = len(mre_data)
        
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
                for itest in range(ntests):
                    ax.plot(temperature_retrieved[itest], pressure, color=cmap(itest / ntests), label=f"Test {itest}")
                xmin, xmax = 90, 390
                xticks = np.arange(xmin, xmax+1, 60)
                xlabel = "Temperature (K)"
                ax.legend(loc='upper right')
            elif (i == 1):
                # Plot residual a priori temperature profile
                ax.fill_betweenx(pressure, -temperature_prior_err[0], temperature_prior_err[0], color='black', alpha=0.3)
                ax.plot([0]*len(temperature_prior[0]), pressure, color='black', label=f"Test {itest}")
                for itest in range(ntests):
                    ax.plot(temperature_retrieved[itest]-temperature_prior[0], pressure, color=cmap(itest / ntests), label='Retrieved')
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
        
    @classmethod
    def plot_ctl_spectrum(cls, dir: str, mre_data: dict, spx_data: dict) -> None:
    
        # Define common plotting parameters
        fontsize = 7
        linewidth = 1
        n_plots_per_filter = 3
        ntests = len(mre_data)
        ny = float(mre_data[0]['ny'])
        ngeom = int(mre_data[0]['ngeom'])
        nconv = int(ny / ngeom)
        
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
                # Plot CTL profile of radiance
                ax.fill_between(emission_angles[0], measured_radiance[0][iconv, :]-measurement_error[0][iconv, :], measured_radiance[0][iconv, :]+measurement_error[0][iconv, :], color='black', alpha=0.3)
                ax.plot(emission_angles[0], measured_radiance[0][iconv, :], color='black', lw=1.5, label='Observed')
                for itest in range(ntests):
                    col = cmap(itest / ntests)
                    ax.plot(emission_angles[itest], retrieved_radiance[itest][iconv, :], color=col, lw=1.5, label=f"Test {itest}")
                values = [(meas[iconv, :], retr[iconv, :]) for (meas, retr) in zip(measured_radiance, retrieved_radiance)]
                ymin, ymax = 0.95*np.min([np.min(v) for v in values]), 1.05*np.max([np.max(v) for v in values])
                ylabel = "R"
                tx, ty = xmin + 0.05*(xmax-xmin), ymin + 0.3*(ymax-ymin)
                wave = wavenumber[0][iconv][0]
                ax.text(tx, ty, int(wave), fontsize=fontsize+1, fontweight='bold', bbox=dict(facecolor='white', edgecolor='grey', boxstyle='round'))
            elif (icol == 1):
                # Plot CTL profile of residual radiance (fit minus measurement)
                ax.fill_between(emission_angles[0], -measurement_error[0][iconv, :], +measurement_error[0][iconv, :], color='black', alpha=0.3)
                for itest in range(ntests):
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
        
        # experiment = "/Users/ptdonnelly/Documents/Research/projects/nemesis_centre_to_limb/retrievals/experiment_3_characterise_ctl_profile_bins_maxmu/"
        # tests = [
        #         "bins_maxmu_0_75_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         "bins_maxmu_5_75_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         "bins_maxmu_10_75_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         "bins_maxmu_15_75_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         "bins_maxmu_20_75_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         "bins_maxmu_25_75_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         "bins_maxmu_30_75_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         "bins_maxmu_35_75_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         "bins_maxmu_40_75_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         "bins_maxmu_45_75_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         # "bins_maxmu_50_75_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         # "bins_maxmu_55_75_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         # "bins_maxmu_60_75_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         "bins_maxmu_65_75_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         "bins_maxmu_70_75_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/"
        #         ]
        
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

        # experiment = "/Users/ptdonnelly/Documents/Research/projects/nemesis_centre_to_limb/retrievals/experiment_5_full_ctl_parametric_gas_retrieval/"
        # tests = [
        #         "bins_maxmu_0_75_flat5_jupiter2021_nh3cloud_0_1p_11s_27s_26s/"]#,
        #         "bins_maxmu_0_75_flat5_jupiter2021_nh3cloud_0_1p_11pt_27p_26p/"]#,
        #         "bins_maxmu_0_75_flat5_jupiter2021_nh3cloud_0_1p_11pt_27p_26s/",
        #         "bins_maxmu_0_75_flat5_jupiter2021_nh3cloud_0_1p_11pt_27s_26p/",
        #         "bins_maxmu_0_75_flat5_jupiter2021_nh3cloud_0_1p_11pt_27s_26s/",
        #         "bins_maxmu_0_75_flat5_jupiter2021_nh3cloud_0_1p_11s_27p_26p/",
        #         "bins_maxmu_0_75_flat5_jupiter2021_nh3cloud_0_1p_11s_27p_26s/"
        #         # "bins_maxmu_0_75_flat5_jupiter2021_nh3cloud_0_1p_11s_27s_26p/"
        #         ]

        # experiment = "/Users/ptdonnelly/Documents/Research/projects/nemesis_centre_to_limb/retrievals/experiment_6_full_ctl_continuous_gas_retrieval/"
        # tests = [
        #         "bins_maxmu_0_75_flat5_jupiter2021_nh3cloud_0_1p_11s_27_26/",
        #         # "bins_maxmu_0_75_flat5_jupiter2021_nh3cloud_0_1p_11s_27_26s/",
        #         # "bins_maxmu_0_75_flat5_jupiter2021_nh3cloud_0_1p_11s_27s_26/",
        #         # "bins_maxmu_0_75_flat5_jupiter2021_nh3cloud_0_1p_11s_27s_26s/"
        #         ]

        # experiment = "/Users/ptdonnelly/Documents/Research/projects/nemesis_centre_to_limb/retrievals/experiment_7_meridian_vs_limb/"
        # tests = [
        #         "merid_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
        #         "limb_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/"
        #         ]

        experiment = "/Users/ptdonnelly/Documents/Research/projects/nemesis_centre_to_limb/retrievals/experiment_8_meridian_vs_limb_aerosol_testing/"
        tests = [
                "limb_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/",
                "merid_flat5_jupiter2021_nh3cloud_0_1p_27s_26s/"
                ]

        # experiment = "/Users/ptdonnelly/Documents/Research/projects/nemesis_centre_to_limb/retrievals/experiment_9_meridian_vs_limb_nh3_testing/"
        # tests = [
        #         "limb_flat5_jupiter2021_nh3cloud_0_1p_11/",
        #         "limb_flat5_jupiter2021_nh3cloud_0_1p_11pt/",
        #         "limb_flat5_jupiter2021_nh3cloud_0_1p_11s/"
        #         ]
        
        return experiment, tests

    @classmethod
    def plot_results_global(cls) -> None:
        """Reads and plots the contents of NEMESIS output files for a global retrieval
        mre: contains the measured and retrieved spectra, and retrieved profiles
        spx: contains the measured spectrum (useful for reconstructing geometries)
        """

        # Get filepaths of retrieval outputs
        experiment, tests = cls.get_retrievals()

        # Get subdirectory, if it does not exist, create it - format argument as you like
        spectra_dir = cls.make_new_dir(f"{experiment}spectra_experiment/")
        temperature_dir = cls.make_new_dir(f"{experiment}temperature_experiment/")

        # Create fixed grid on which to plot retrievals
        latitudes = np.arange(-89.5, 90, 1)

        for latitude in latitudes:

            print(f"Plotting: {latitude}")
            
            # Create empty lists to store each test at this latitude
            mre_data, spx_data = [], []

            # Loop over each test within the experiment
            for test in tests:
                
                # Point to location of retrieval outputs
                cores = natsort.natsorted(glob.glob(f"{experiment}{test}core*"))
                
                # Verify if latitude exists in this test, return filepath to core
                core = nem.check_core(cores, latitude)
                if not core:
                    pass
                else:
                    # Read NEMESIS outputs
                    mre_data.append(nem.read_mre(f"{core}/nemesis.mre"))
                    spx_data.append(nem.read_spx(f"{core}/nemesis.spx"))

            # Plot results
            if not mre_data:
                pass
            else:
                cls.plot_spectrum(spectra_dir, mre_data)
                # cls.plot_ctl_spectrum(spectra_dir, mre_data, spx_data)
                cls.plot_temperature(temperature_dir, mre_data)

                # # Create final plots for figure
                # FinalPlot.plot_spectrum_meridian_vs_limb(f"{spectra_dir}_final", mre_data)
                # FinalPlot.plot_temperatures_meridian_vs_limb(f"{temperature_dir}_final", mre_data)
        return
    
    @classmethod
    def plot_results(cls) -> None:
        """Reads and plots the contents of NEMESIS output files for a global retrieval
        mre: contains the measured and retrieved spectra, and retrieved profiles
        spx: contains the measured spectrum (useful for reconstructing geometries)
        """

        # Get filepaths of retrieval outputs
        experiment, tests = cls.get_retrievals()

        # Loop over each test within the experiment
        for test in tests:

            # Get subdirectory, if it does not exist, create it - format argument as you like
            spectra_dir = cls.make_new_dir(f"{experiment}spectra_test/{test}")
            temperature_dir = cls.make_new_dir(f"{experiment}temperature_test/{test}")
            xsc_dir = cls.make_new_dir(f"{experiment}xsc_test/{test}")
            
            # Point to location of retrieval outputs
            cores = natsort.natsorted(glob.glob(f"{experiment}{test}core*"))
            
            # Create empty lists to store each test at this latitude
            mre_data, spx_data, xsc_data = [], [], []

            # Read NEMESIS outputs
            for core in cores:
                mre_data.append(nem.read_mre(f"{core}/nemesis.mre"))
                spx_data.append(nem.read_spx(f"{core}/nemesis.spx"))
                xsc_data.append(nem.read_xsc(f"{core}/nemesis.xsc"))
            # Plot results
            if not mre_data:
                pass
            else:
                # cls.plot_spectrum(spectra_dir, mre_data)
                # # cls.plot_ctl_spectrum(spectra_dir, mre_data, spx_data)
                # cls.plot_temperature(temperature_dir, mre_data)
                cls.plot_xsc(xsc_dir, xsc_data)

                # # Create final plots for figure
                # FinalPlot.plot_spectrum_meridian_vs_limb(f"{spectra_dir}_final", mre_data)
                # FinalPlot.plot_temperatures_meridian_vs_limb(f"{temperature_dir}_final", mre_data)
        return
    
    @classmethod
    def plot_contribution_function(cls) -> None:
        """Reads and plots the contents of NEMESIS output files
        mre: contains the measured and retrieved spectra, and retrieved profiles
        spx: contains the measured spectrum (useful for reconstructing geometries)
        """

        # Get filepaths of retrieval outputs
        experiment, tests = cls.get_retrievals()

        # Get subdirectory, if it does not exist, create it - format argument as you like
        kk_dir = cls.make_new_dir(f"{experiment}contribution_function/")
        
        # Create an empty list to store each test
        mre_data, kk_data = [], []

        # Loop over each test within the experiment
        for test in tests:
            
            # Read NEMESIS outputs
            mre_data.append(nem.read_mre(f"{experiment}{test}core_1/nemesis.mre"))
            kk_data.append(nem.read_kk(f"{experiment}{test}core_1/nemesis.cov"))

        # Plot contribution function
        # FinalPlot.plot_1d_contribution_function(f"{kk_dir}", mre_data, kk_data)
        # FinalPlot.plot_2d_contribution_function(f"{kk_dir}", mre_data, kk_data)

class FinalPlot:

    def __init__():
        return
    
    @classmethod
    def plot_spectrum_meridian_vs_limb(cls, dir: str, mre_data: dict) -> None:
     
        # Define common plotting parameters
        fontsize = 7
        linewidth = 1
        ntests = len(mre_data)
        
        # Set up figure
        latitude = float(mre_data[0]['latitude'])
        figname = f"{dir}{latitude}.png"
        _, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(6, 6), dpi=300)
        test_labels = ['Meridian', 'Limb']
        test_colors = ['royalblue', 'crimson']
        plt.suptitle(f"Latitude = {latitude}\n[0, -1p, 27s, 26s]", fontsize=fontsize)

        # Read temperature profile from mrefile, reshape nested lists from (npro, nparams) to (nparams, npro) and broadcast as numpy array
        wavenumber, measured_radiance, measurement_error, retrieved_radiance, measured_TB, retrieved_TB = [[] for _ in range(6)]
        for i, mre in enumerate(mre_data):
            # Read spectrum (remove the extra surrounding list from ngeom = 1)
            spectrum = mre['spectrum'].pop()
            wavenumber.append(np.asarray([float(value) for value in spectrum[1]], dtype=float))
            measured_radiance.append(np.asarray([float(value) for value in spectrum[2]], dtype=float))
            measurement_error.append(np.asarray([float(value) for value in spectrum[3]], dtype=float))
            retrieved_radiance.append(np.asarray([float(value) for value in spectrum[5]], dtype=float))

            # Convert from radiance to brightness temperature
            measured_TB.append(PlotNemesis.radiance_to_brightness_temperature(wavenumber[i], measured_radiance[i]))
            retrieved_TB.append(PlotNemesis.radiance_to_brightness_temperature(wavenumber[i], retrieved_radiance[i]))

        # Loop over each axis object
        for i, ax in enumerate(np.transpose(axes.flat)):      
            if (i == 0):
                # Plot radiance
                for itest in range(ntests):
                    ax.fill_between(wavenumber[itest], measured_radiance[itest]-measurement_error[itest], measured_radiance[itest]+measurement_error[itest], color=test_colors[itest], alpha=0.25)
                    ax.plot(wavenumber[itest], measured_radiance[itest], color=test_colors[itest], lw=0, marker='.', markersize='4')
                    ax.plot(wavenumber[itest], retrieved_radiance[itest], color=test_colors[itest], label=test_labels[itest])
                ylabel = "Radiance (nw ...)"
                ax.legend(loc='upper right')
            elif (i == 1):
                # Plot residual radiance
                for itest in range(ntests):
                    ax.fill_between(wavenumber[itest], [-1 * err for err in measurement_error[itest]], measurement_error[itest], color=test_colors[itest], alpha=0.25)
                    ax.plot(wavenumber[itest], retrieved_radiance[itest]-measured_radiance[itest], color=test_colors[itest])
                ylabel = r"$\Delta$ Radiance (nw ...)"
            elif (i == 2):
                # Plot brightness temperature
                for itest in range(ntests):
                    ax.plot(wavenumber[itest], measured_TB[itest], color=test_colors[itest], lw=0, marker='.', markersize='4')
                    ax.plot(wavenumber[itest], retrieved_TB[itest], color=test_colors[itest])
                ylabel = r"T$_{B}$ (K)"
            elif (i == 3):
                # Plot residual brightness temperature
                for itest in range(ntests):
                    ax.plot(wavenumber[itest], retrieved_TB[itest]-measured_TB[itest], color=test_colors[itest])
                
                ylabel = r"$\Delta$ T$_{B}$ (K)"
                ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")

            ax.grid(axis='both', markevery=1, color='k', ls=':', lw=0.5)
            ax.set_ylabel(ylabel)
            ax.tick_params(axis='both', length=1, pad=1, labelsize=6)

        plt.xlabel(r"Wavenumber (cm$^{-1}$)")
        plt.xticks(wavenumber[0])
        plt.savefig(figname, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    @classmethod
    def plot_temperatures_meridian_vs_limb(cls, dir: str, mre_data: dict) -> None:
        
        # Define common plotting parameters
        fontsize = 7
        linewidth = 1
        ntests = len(mre_data)
        
        # Set up figure
        latitude = float(mre_data[0]['latitude'])
        figname = f"{dir}{latitude}.png"
        _, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(5, 5), gridspec_kw={'width_ratios': [2, 1]}, dpi=300)
        test_labels = ['Meridian', 'Limb']
        test_colors = ['royalblue', 'crimson']
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
                # Plot a temperature profiles
                ax.fill_betweenx(pressure, temperature_prior[0]-temperature_prior_err[0], temperature_prior[0]+temperature_prior_err[0], color='black', alpha=0.25)
                ax.plot(temperature_prior[0], pressure, color='black', label='Prior')
                for itest in range(ntests):
                    col = test_colors[itest]
                    ax.plot(temperature_retrieved[itest], pressure, color=col, label=test_labels[itest])
                xmin, xmax = 90, 390
                xticks = np.arange(xmin, xmax+1, 60)
                xlabel = "Temperature (K)"
                ax.legend(loc='upper right')
            elif (i == 1):
                # Plot residual temperature profiles
                ax.fill_betweenx(pressure, -temperature_prior_err[0], temperature_prior_err[0], color='black', alpha=0.25)
                ax.plot([0]*len(temperature_prior[0]), pressure, color='black', label='Prior')
                for itest in range(ntests):
                    col = test_colors[itest]
                    ax.plot(temperature_retrieved[itest]-temperature_prior[0], pressure, color=col, label=test_labels[itest])
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
    
    @classmethod
    def plot_1d_contribution_function(cls, dir: str, mre_data: dict, kk_data: dict) -> None:
        
        # Define common plotting parameters
        fontsize = 9
        linewidth = 1
        npro, ny = kk_data[0]['npro'], kk_data[0]['ny']
        pressure = np.arange(npro)

        # Read outputs
        weighting_function = [[[float(value) for value in values] for values in zip(*kk['kk'])] for kk in kk_data]
        wavenumber = [[[float(value) for value in values[1]] for values in mre['spectrum']] for mre in mre_data]
        labels = wavenumber[0].pop()
        
        # Set up figure
        latitude = float(0.5) # mre_data[0]['latitude'])
        figname = f"{dir}{latitude}.png"
        _, axes = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True, figsize=(4, 4), dpi=300)
        cmap = plt.get_cmap('turbo')
        plt.suptitle(f"Contribution Weighting Functions at the Equator", fontsize=fontsize)
        titles = ['Meridian', 'Limb']

        # Loop over each axis object
        for i, ax in enumerate(np.transpose(axes.flat)):      
            xmin, xmax = 0, 1
            ymin, ymax = 0, 120
            xlabel = r"Normalised $\frac{dR}{dx}$"
            ylabel = "Pressure (atm)"

            for iy in range(ny):
                if (i == 0):
                    data = weighting_function[i][iy][:npro]
                    normalised_data  = [value / max(data) for value in data]
                elif (i == 1):
                    data = weighting_function[i][ny-iy-1][:npro]
                    normalised_data  = [value / max(data) for value in data]
                ax.plot(normalised_data, pressure, color=cmap(iy / ny), label=int(labels[iy]))
            
            ax.grid(axis='both', markevery=1, color='k', ls=':', lw=0.5)
            ax.set_title(titles[i], fontsize=fontsize, pad=2)
            ax.set_xlim((xmin, xmax))
            ax.set_ylim((ymin, ymax))
            ax.set_xlabel(xlabel, fontsize=fontsize-1, labelpad=2)
            if (i == 0):
                ax.set_ylabel(ylabel, fontsize=fontsize-1, labelpad=2)
            elif (i == 1):
                ax.legend(loc='upper right', fontsize=fontsize-2)
            ax.tick_params(axis='both', length=1, pad=1, labelsize=fontsize-1)
        
        # Finish and close plot
        plt.subplots_adjust(hspace=0.1, wspace=0.2)  
        plt.savefig(figname, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    @classmethod
    def plot_2d_contribution_function(cls, dir: str, mre_data: dict, kk_data: dict) -> None:

        # Define common plotting parameters
        fontsize = 7
        linewidth = 1
        npro, ny = kk_data[0]['npro'], kk_data[0]['ny']
        pressure = np.arange(npro)

        # Read outputs
        weighting_function = np.asarray([[[float(value) for value in values] for values in kk['kk']] for kk in kk_data].pop(), dtype=float)
        wavenumber = [[[float(value) for value in values[1]] for values in mre['spectrum']] for mre in mre_data]
        titles = np.asarray(wavenumber[0].pop(), dtype=int)
        nconv = len(titles)

         # Set up figure
        latitude = float(0.5) # mre_data[0]['latitude'])
        figname = f"{dir}{latitude}.png"
        _, axes = plt.subplots(nrows=2, ncols=4, sharex=True, sharey=True, figsize=(5, 4), dpi=300)
        cmap = plt.get_cmap('gist_yarg', lut=10)
        plt.suptitle(f"Contribution Weighting Functions at the Equator", fontsize=fontsize)

        # Loop over each axis object
        for i, ax in enumerate(np.transpose(axes.flat)):      
            xmin, xmax = 0, 75
            ymin, ymax = 0, 120
            xlabel = "Emission Angle (deg)"
            ylabel = "Pressure (atm)"

            ngeom = int(ny / nconv)
            normalised_data = np.empty((npro, ngeom))
            for igeom in range(ngeom):
                find = i + (igeom * nconv)
                data = weighting_function[:npro, find]
                normalised_data[:, igeom] = data / np.max(data)
            ax.imshow(normalised_data, cmap=cmap, vmin=0, vmax=1)
            ax.plot([75]*normalised_data[:, 0], pressure, color='royalblue', lw=0.75, label='Meridian')
            ax.plot([75]*normalised_data[:, -1], pressure, color='crimson', lw=0.75, label='Limb')

            # Clean up axes
            ax.grid(axis='both', markevery=1, color='k', ls=':', lw=0.5)
            ax.set_title(titles[i], fontsize=fontsize, pad=2)
            ax.set_xlim((xmin, xmax))
            ax.set_ylim((ymin, ymax))
            if (i == 3):
                ax.legend(loc='upper right', fontsize=fontsize-1)
            if (i // 4 != 0):
                ax.set_xlabel(xlabel, fontsize=fontsize-1, labelpad=2)
            if (i % 4 == 0):
                ax.set_ylabel(ylabel, fontsize=fontsize-1, labelpad=2)
            ax.tick_params(axis='both', length=1, pad=1, labelsize=fontsize-1)
        
        # Finish and close plot
        plt.subplots_adjust(hspace=0.1, wspace=0.1)  
        plt.savefig(figname, dpi=300, bbox_inches='tight')
        plt.close()
        return