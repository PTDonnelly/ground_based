import os
import numpy as np
import Globals
from snoop import snoop

def WriteMeridSpx(dataset, mode, spectrals):
    """Create spectral input for NEMESIS using central meridian profiles.
       Populate .spxfile with radiances, measurement errors, and geometries."""

    print('Creating meridian spectra...')

    def create_merid_spx(f, lats, LCMs, mus, rads, rad_errs, waves):
        """Write spxfile for meridional binning method """

        # Write first line of texfile with relevant formatting
        clat = lats[0]
        LCM  = np.mean(LCMs)
        nmu  = 1
        f.write("{0:12.5f}  {1:12.5f}  {2:12.5f}  {3:12.5f}\n".format(0, clat, LCM, nmu))

        # Calculate NCONV spectral points (NCONV = no. of wavenumbers per geometry = 1 for merid binning)
        nconv = len(waves)
        # I can't remember what NAV is... check the NEMESIS manual
        nav   = 1
        # The "angles line": this defines a "geometry" that holds a "spectrum"
        clat           = lats[0]
        clon           = LCMs[0]
        solar_ang      = 0
        emission_angle = min(mus)
        azimuth_angle  = 0
        wgeom          = 1 
        # The spectrum lines: these are the spectral points that occur at a given "geometry"
        waves.reverse()
        rads.reverse()
        rad_errs.reverse()
        # Write output to texfile with relevant formatting
        f.write("{0:d}\n".format(nconv))
        f.write("{0:d}\n".format(nav))
        f.write("{0:12.5f}  {1:12.5f}  {2:12.5f}  {3:12.5f}  {4:12.5f}  {5:12.5f}\n".format(clat, clon, solar_ang, emission_angle, azimuth_angle, wgeom))
        for spec in zip(waves, rads, rad_errs):
            f.write("{0:10.4f}  {1:15.6e}  {2:15.6e}\n".format(spec[0], spec[1], spec[2]))

    # If subdirectory does not exist, create it
    dir = f'../outputs/{dataset}/spxfiles_merid_new/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Loop over latitudes to create one .spxfile per latitude
    for ilat in range(Globals.nlatbins):
        # Extract variables and throw NaNs
        lats     = [spectrals[ilat, ifilt, 0] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ilat, ifilt, 0]) == False]
        LCMs     = [spectrals[ilat, ifilt, 1] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ilat, ifilt, 1]) == False]
        mus      = [spectrals[ilat, ifilt, 2] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ilat, ifilt, 2]) == False]
        rads     = [spectrals[ilat, ifilt, 3] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ilat, ifilt, 3]) == False]
        rad_errs = [spectrals[ilat, ifilt, 4] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ilat, ifilt, 4]) == False]
        waves    = [spectrals[ilat, ifilt, 5] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ilat, ifilt, 5]) == False]
        # Only write spxfile for latitudes with spectral information
        if lats:
            # Open textfile
            with open(f"{dir}lat_{lats[0]}.txt", 'w') as f:
                create_merid_spx(f, lats, LCMs, mus, rads, rad_errs, waves)
            # Open spxfile
            with open(f"{dir}lat_{lats[0]}.spx", 'w') as f:
                create_merid_spx(f, lats, LCMs, mus, rads, rad_errs, waves)

def WriteParaSpx(dataset, mode, spectrals):
    """Create spectral input for NEMESIS using parallel profiles at a specific latitude range.
       Populate .spxfile with radiances, measurement errors, and geometries."""

    print('Creating parallel spectra...')

    def create_para_spx(f, LCPs, lons, mus, rads, rad_errs, waves):
        """Write spxfile for parallel binning method """
        # Write first line of texfile with relevant formatting
        LCP = np.mean(LCPs)
        clon  = lons[0]
        nmu  = len(mus)
        f.write("{0:12.5f}  {1:12.5f}  {2:12.5f}  {3:12.5f}\n".format(0, LCP, clon, nmu))

        # Loop over NGEOM geometries (no. of geometries = no. of emission angle points)
        for igeom, mu in enumerate(mus):
            # Calculate NCONV spectral points (NCONV = no. of wavenumbers per geometry = 1 for merid binning)
            nconv = 1
            # I can't remember what NAV is... check the NEMESIS manual
            nav   = 1
            # The "angles line": this defines a "geometry" that holds a "spectrum"
            clat           = LCPs[igeom]
            clon           = lons[igeom]
            solar_ang      = 0
            emission_angle = mu
            azimuth_angle  = 0
            wgeom          = 1 
            # The spectrum lines: these are the spectral points that occur at a given "geometry"
            wave    = waves[igeom]
            rad     = rads[igeom]
            rad_err = rad_errs[igeom]

            # Write output to texfile with relevant formatting
            f.write("{0:d}\n".format(nconv))
            f.write("{0:d}\n".format(nav))
            f.write("{0:12.5f}  {1:12.5f}  {2:12.5f}  {3:12.5f}  {4:12.5f}  {5:12.5f}\n".format(clat, clon, solar_ang, emission_angle, azimuth_angle, wgeom))
            f.write("{0:10.4f}  {1:15.6e}  {2:15.6e}\n".format(wave, rad, rad_err))        
    
    # If subdirectory does not exist, create it
    dir = f'../outputs/{dataset}/spxfiles_para_{Globals.LCP}_no852_no887/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    # Loop over longitudes to create one .spxfile per longitude
    for ilon in range(Globals.nlonbins):
        # Extract variables and throw NaNs
        LCPs     = [spectrals[ilon, ifilt, 0] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ilon, ifilt, 0]) == False]
        lons     = [spectrals[ilon, ifilt, 1] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ilon, ifilt, 1]) == False]
        mus      = [spectrals[ilon, ifilt, 2] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ilon, ifilt, 2]) == False]
        rads     = [spectrals[ilon, ifilt, 3] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ilon, ifilt, 3]) == False]
        rad_errs = [spectrals[ilon, ifilt, 4] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ilon, ifilt, 4]) == False]
        waves    = [spectrals[ilon, ifilt, 5] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ilon, ifilt, 5]) == False]
        # Only write spxfile for longitudes with spectral information
        if lons:
            # Open textfile
            with open(f"{dir}lon_{lons[0]}.txt", 'w') as f:
                create_para_spx(f, LCPs, lons, mus, rads, rad_errs, waves)
            # Open spxfile
            with open(f"{dir}lon_{lons[0]}.spx", 'w') as f:
                create_para_spx(f, LCPs, lons, mus, rads, rad_errs, waves)

def WriteCentreToLimbSpx(dataset, mode, spectrals):
    """Create spectral input for NEMESIS using centre-to-limb profiles.
       Populate .spxfile with radiances, measurement errors, and geometries."""
    
    print('Creating spectra...')
    
    def create_ctl_spx(f, lats, LCMs, mus, rads, rad_errs, waves):
        """Write spxfile for meridional binning method """

        # Write first line of texfile with relevant formatting
        clat = lats[0]
        LCM  = np.mean(LCMs)
        nmu  = len(mus)
        f.write("{0:12.5f}  {1:12.5f}  {2:12.5f}  {3:d}\n".format(0, clat, LCM, nmu))

        # Loop over NGEOM geometries (no. of geometries = no. of emission angle points)
        for imu, mu in enumerate(mus):
            # Calculate NCONV spectral points (NCONV = no. of wavenumbers per geometry = 1 for merid binning)
            nconv = len(waves[imu])
            # I can't remember what NAV is... check the NEMESIS manual
            nav   = 1
            # The "angles line": this defines a "geometry" that holds a "spectrum"
            clat           = lats[imu]
            clon           = LCMs[imu]
            solar_ang      = 0
            emission_angle = mu
            azimuth_angle  = 0
            wgeom          = 1 
            # Write output to texfile with relevant formatting
            f.write("{0:d}\n".format(nconv))
            f.write("{0:d}\n".format(nav))
            f.write("{0:12.5f}  {1:12.5f}  {2:12.5f}  {3:12.5f}  {4:12.5f}  {5:12.5f}\n".format(clat, clon, solar_ang, emission_angle, azimuth_angle, wgeom))
            # The spectrum lines: these are the spectral points that occur at a given "geometry"
            wave, rad, rad_err = waves[imu], rads[imu], rad_errs[imu]
            for iconv, spec in enumerate(zip(wave, rad, rad_err)):
                f.write("{0:10.4f}  {1:15.6e}  {2:15.6e}\n".format(spec[0], spec[1], spec[2]))

    # If subdirectory does not exist, create it
    dir = f'../outputs/{dataset}/spxfiles_ctl/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Loop over latitudes to create one .spxfile per latitude
    for ilat in range(Globals.nlatbins):
        lats, LCMs, mus, rads, rad_errs, waves = [[] for _ in range(6)]
        for imu in range(Globals.nmubins):
            # Extract variables and throw NaNs
            lat     = [spectrals[ifilt, ilat, imu, 0] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ifilt, ilat, imu, 0]) == False]
            LCM     = [spectrals[ifilt, ilat, imu, 1] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ifilt, ilat, imu, 1]) == False]
            mu      = [spectrals[ifilt, ilat, imu, 2] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ifilt, ilat, imu, 2]) == False]
            rad     = [spectrals[ifilt, ilat, imu, 3] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ifilt, ilat, imu, 3]) == False]
            rad_err = [spectrals[ifilt, ilat, imu, 4] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ifilt, ilat, imu, 4]) == False]
            wave    = [spectrals[ifilt, ilat, imu, 5] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ifilt, ilat, imu, 5]) == False]
            if lat:
                lats.append(lat[0])
                LCMs.append(LCM[0])
                mus.append(mu[0])
                rads.append(rad)
                rad_errs.append(rad_err)
                waves.append(wave)
        # Only write spxfile for latitudes with spectral information
        if lats:
            # Open textfile
            with open(f"{dir}lat_{lats[0]}.txt", 'w') as f:
                create_ctl_spx(f, lats, LCMs, mus, rads, rad_errs, waves)
            # Open spxfile
            with open(f"{dir}lat_{lats[0]}.spx", 'w') as f:
                create_ctl_spx(f, lats, LCMs, mus, rads, rad_errs, waves)

def WriteLimbSpx(dataset, mode, spectrals):
    """Create spectral input for NEMESIS using centre-to-limb profiles.
       Populate .spxfile with radiances, measurement errors, and geometries.
       Differs from WriteCentreToLimbSpx() because it selects only the highest emission angles"""
    
    print('Creating spectra...')
    
    def create_ctl_spx(f, lats, LCMs, mus, rads, rad_errs, waves):
        """Write spxfile for meridional binning method """

        # Write first line of texfile with relevant formatting
        clat = lats[0]
        LCM  = np.mean(LCMs)
        nmu  = len(mus)
        f.write("{0:12.5f}  {1:12.5f}  {2:12.5f}  {3:d}\n".format(0, clat, LCM, nmu))

        # Loop over NGEOM geometries (no. of geometries = no. of emission angle points)
        for imu, mu in enumerate(mus):
            # Calculate NCONV spectral points (NCONV = no. of wavenumbers per geometry = 1 for merid binning)
            nconv = len(waves[imu])
            # I can't remember what NAV is... check the NEMESIS manual
            nav   = 1
            # The "angles line": this defines a "geometry" that holds a "spectrum"
            clat           = lats[imu]
            clon           = LCMs[imu]
            solar_ang      = 0
            emission_angle = mu
            azimuth_angle  = 0
            wgeom          = 1 
            # Write output to texfile with relevant formatting
            f.write("{0:d}\n".format(nconv))
            f.write("{0:d}\n".format(nav))
            f.write("{0:12.5f}  {1:12.5f}  {2:12.5f}  {3:12.5f}  {4:12.5f}  {5:12.5f}\n".format(clat, clon, solar_ang, emission_angle, azimuth_angle, wgeom))
            # The spectrum lines: these are the spectral points that occur at a given "geometry"
            wave, rad, rad_err = waves[imu], rads[imu], rad_errs[imu]
            for iconv, spec in enumerate(zip(wave, rad, rad_err)):
                f.write("{0:10.4f}  {1:15.6e}  {2:15.6e}\n".format(spec[0], spec[1], spec[2]))

    # If subdirectory does not exist, create it
    mu1, mu2 = 0, 90
    dir = f'../outputs/{dataset}/spxfiles_bins_lowmu_{mu1}_{mu2}/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    mubins = np.arange(mu1, mu2, 1)
    nmubins = len(mubins)

    # Loop over latitudes to create one .spxfile per latitude
    for ilat in range(Globals.nlatbins):
        lats, LCMs, mus, rads, rad_errs, waves = [[] for _ in range(6)]
        for imu in range(nmubins):
            imu += mubins[0]

            # Extract variables and throw NaNs
            lat     = [spectrals[ifilt, ilat, imu, 0] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ifilt, ilat, imu, 0]) == False]
            LCM     = [spectrals[ifilt, ilat, imu, 1] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ifilt, ilat, imu, 1]) == False]
            mu      = [spectrals[ifilt, ilat, imu, 2] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ifilt, ilat, imu, 2]) == False]
            rad     = [spectrals[ifilt, ilat, imu, 3] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ifilt, ilat, imu, 3]) == False]
            rad_err = [spectrals[ifilt, ilat, imu, 4] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ifilt, ilat, imu, 4]) == False]
            wave    = [spectrals[ifilt, ilat, imu, 5] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ifilt, ilat, imu, 5]) == False]
            if lat:
                lats.append(lat[0])
                LCMs.append(LCM[0])
                mus.append(mu[0])
                rads.append(rad)
                rad_errs.append(rad_err)
                waves.append(wave)
        # Only write spxfile for latitudes with spectral information
        if lats:
            # Open textfile
            with open(f"{dir}lat_{lats[0]}.txt", 'w') as f:
                create_ctl_spx(f, lats, LCMs, mus, rads, rad_errs, waves)
            # Open spxfile
            with open(f"{dir}lat_{lats[0]}.spx", 'w') as f:
                create_ctl_spx(f, lats, LCMs, mus, rads, rad_errs, waves)

def WriteLimbAverageSpx(dataset, mode, spectrals):
    """Create spectral input for NEMESIS using centre-to-limb profiles.
       Populate .spxfile with radiances, measurement errors, and geometries.
       Differs from WriteCentreToLimbSpx() because it selects only the highest emission angles"""
    
    print('Creating spectra...')
    
    def create_ctl_spx(f, lats, LCMs, mus, rads, rad_errs, waves):
        """Write spxfile for meridional binning method """
        
        # Reshape the lists and calculate the mean over the bins
        rads_avg = [np.nanmean(rad) for rad in zip(*rads)]
        rad_errs_avg = [np.nanmean(rad_err) for rad_err in zip(*rad_errs)]
        waves_avg = [np.nanmean(wave) for wave in zip(*waves)]       

        # Write first line of texfile with relevant formatting
        clat = lats[0]
        LCM  = np.mean(LCMs)
        nmu  = 1
        f.write("{0:12.5f}  {1:12.5f}  {2:12.5f}  {3:d}\n".format(0, clat, LCM, nmu))
 
        # Calculate NCONV spectral points (NCONV = no. of wavenumbers per geometry = 1 for merid binning)
        nconv = len(waves_avg)
        # I can't remember what NAV is... check the NEMESIS manual
        nav   = 1
        # The "angles line": this defines a "geometry" that holds a "spectrum"
        clat           = lats[0]
        clon           = LCMs[0]
        solar_ang      = 0
        emission_angle = mus[0]
        azimuth_angle  = 0
        wgeom          = 1 
        # Write output to texfile with relevant formatting
        f.write("{0:d}\n".format(nconv))
        f.write("{0:d}\n".format(nav))
        f.write("{0:12.5f}  {1:12.5f}  {2:12.5f}  {3:12.5f}  {4:12.5f}  {5:12.5f}\n".format(clat, clon, solar_ang, emission_angle, azimuth_angle, wgeom))
        # The spectrum lines: these are the spectral points that occur at a given "geometry"
        wave, rad, rad_err = waves_avg, rads_avg, rad_errs_avg
        # Write output to texfile with relevant formatting
        for iconv, spec in enumerate(zip(wave, rad, rad_err)):
            f.write("{0:10.4f}  {1:15.6e}  {2:15.6e}\n".format(spec[0], spec[1], spec[2]))

    # If subdirectory does not exist, create it
    mu1, mu2 = 65, 75
    dir = f'../outputs/{dataset}/spxfiles_limb_avg/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    mubins = np.arange(mu1, mu2, 1)

    # Loop over latitudes to create one .spxfile per latitude
    for ilat in range(Globals.nlatbins):
        lats, LCMs, mus, rads, rad_errs, waves = [[] for _ in range(6)]
        for imu, _ in enumerate(mubins):
            imu += mubins[0]

            # Extract variables and throw NaNs
            lat     = [spectrals[ifilt, ilat, imu, 0] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ifilt, ilat, imu, 0]) == False]
            LCM     = [spectrals[ifilt, ilat, imu, 1] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ifilt, ilat, imu, 1]) == False]
            mu      = [spectrals[ifilt, ilat, imu, 2] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ifilt, ilat, imu, 2]) == False]
            rad     = [spectrals[ifilt, ilat, imu, 3] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ifilt, ilat, imu, 3]) == False]
            rad_err = [spectrals[ifilt, ilat, imu, 4] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ifilt, ilat, imu, 4]) == False]
            wave    = [spectrals[ifilt, ilat, imu, 5] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ifilt, ilat, imu, 5]) == False]

            if lat:
                lats.append(lat[0])
                LCMs.append(LCM[0])
                mus.append(np.mean((mu1, mu2)))
                rads.append(rad)
                rad_errs.append(rad_err)
                waves.append(wave)
        # Only write spxfile for latitudes with spectral information
        if lats:
            # Open textfile
            with open(f"{dir}lat_{lats[0]}.txt", 'w') as f:
                create_ctl_spx(f, lats, LCMs, mus, rads, rad_errs, waves)
            # Open spxfile
            with open(f"{dir}lat_{lats[0]}.spx", 'w') as f:
                create_ctl_spx(f, lats, LCMs, mus, rads, rad_errs, waves)

def WriteRegionalSpx(dataset, mode, spectrals):
    """Create spectral input for NEMESIS using regional binning scheme.
       Populate .spxfile with radiances, measurement errors, and geometries."""

    print('Creating regional spectra...')

    def create_regional_spx(f, lats, lons, mus, rads, rad_errs, waves):
        """Write spxfile for regional binning method """
        # Write first line of texfile with relevant formatting
        clat = lats[0]
        clon = lons[0]
        nmu  = len(mus)
        f.write("{0:12.5f}  {1:12.5f}  {2:12.5f}  {3:12.5f}\n".format(0, clat, clon, nmu))

        # Loop over NGEOM geometries (no. of geometries = no. of emission angle points)
        for igeom, mu in enumerate(mus):
            # Calculate NCONV spectral points (NCONV = no. of wavenumbers per geometry = 1 for merid binning)
            nconv = 1
            # I can't remember what NAV is... check the NEMESIS manual
            nav   = 1
            # The "angles line": this defines a "geometry" that holds a "spectrum"
            clat           = lats[igeom]
            clon           = lons[igeom]
            solar_ang      = 0
            emission_angle = mu
            azimuth_angle  = 0
            wgeom          = 1 
            # The spectrum lines: these are the spectral points that occur at a given "geometry"
            wave    = waves[igeom]
            rad     = rads[igeom]
            rad_err = rad_errs[igeom]

            # Write output to texfile with relevant formatting
            f.write("{0:d}\n".format(nconv))
            f.write("{0:d}\n".format(nav))
            f.write("{0:12.5f}  {1:12.5f}  {2:12.5f}  {3:12.5f}  {4:12.5f}  {5:12.5f}\n".format(clat, clon, solar_ang, emission_angle, azimuth_angle, wgeom))
            f.write("{0:10.4f}  {1:15.6e}  {2:15.6e}\n".format(wave, rad, rad_err))        
    
    # If subdirectory does not exist, create it
    dir = f'../outputs/{dataset}/spxfiles_lat{Globals.lat_target}_lon{Globals.lon_target}_no852_no887/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    # Loop over latitudes and logitudes to create one .spxfile per pixel into the 2D selected maps
    for ilat in range(Globals.nlatbins):
        for ilon in range(Globals.nlonbins):
            # print(np.shape(LCPs))
            # Extract variables and throw NaNs
            lats     = [spectrals[ilat, ilon, ifilt, 0] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ilat, ilon, ifilt, 0]) == False]
            lons     = [spectrals[ilat, ilon, ifilt, 1] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ilat, ilon, ifilt, 1]) == False]
            mus      = [spectrals[ilat, ilon, ifilt, 2] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ilat, ilon, ifilt, 2]) == False]
            rads     = [spectrals[ilat, ilon, ifilt, 3] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ilat, ilon, ifilt, 3]) == False]
            rad_errs = [spectrals[ilat, ilon, ifilt, 4] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ilat, ilon, ifilt, 4]) == False]
            waves    = [spectrals[ilat, ilon, ifilt, 5] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ilat, ilon, ifilt, 5]) == False]
            # Only write spxfile for longitudes and latitudes with spectral information
            if lats:
                if lons:
                    # Open textfile
                    with open(f"{dir}lat_{lats[0]}_lon_{lons[0]}.txt", 'w') as f:
                        create_regional_spx(f, lats, lons, mus, rads, rad_errs, waves)
                    # Open spxfile
                    with open(f"{dir}lat_{lats[0]}_lon_{lons[0]}.spx", 'w') as f:
                        create_regional_spx(f, lats, lons, mus, rads, rad_errs, waves)

def WriteRegionalAverageSpx(dataset, mode, spectrals, per_night, Nnight):
    """Create spectral input for NEMESIS using average regional binning scheme.
       Populate .spxfile with radiances, measurement errors, and geometries."""

    print('Creating regional average spectra...')

    def create_av_reg_spx(f, lats, lons, mus, rads, rad_errs, waves):
        """Write spxfile for regional average binning method """
        # Write first line of texfile with relevant formatting
        clat = lats[0]
        clon = lons[0]
        nmu  = len(mus)
        f.write("{0:12.5f}  {1:12.5f}  {2:12.5f}  {3:12.5f}\n".format(0, clat, clon, nmu))

        # Loop over NGEOM geometries (no. of geometries = no. of emission angle points)
        for igeom, mu in enumerate(mus):
            # Calculate NCONV spectral points (NCONV = no. of wavenumbers per geometry = 1 for merid binning)
            nconv = 1
            # I can't remember what NAV is... check the NEMESIS manual
            nav   = 1
            # The "angles line": this defines a "geometry" that holds a "spectrum"
            clat           = lats[igeom]
            clon           = lons[igeom]
            solar_ang      = 0
            emission_angle = mu
            azimuth_angle  = 0
            wgeom          = 1 
            # The spectrum lines: these are the spectral points that occur at a given "geometry"
            wave    = waves[igeom]
            rad     = rads[igeom]
            rad_err = rad_errs[igeom]

            # Write output to texfile with relevant formatting
            f.write("{0:d}\n".format(nconv))
            f.write("{0:d}\n".format(nav))
            f.write("{0:12.5f}  {1:12.5f}  {2:12.5f}  {3:12.5f}  {4:12.5f}  {5:12.5f}\n".format(clat, clon, solar_ang, emission_angle, azimuth_angle, wgeom))
            f.write("{0:10.4f}  {1:15.6e}  {2:15.6e}\n".format(wave, rad, rad_err))        
    
    # If subdirectory does not exist, create it
    dir = f'../outputs/{dataset}/spxfiles_lat{Globals.lat_target}_lon{Globals.lon_target}_RegionalAverage_no852_no887/'
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    if per_night==True:
        for inight in range(Nnight):
        # Loop over latitudes and logitudes to create one .spxfile per pixel into the 2D selected maps
            for ilat in range(Globals.nlatbins):
                for ilon in range(Globals.nlonbins):
                    # Extract variables and throw NaNs
                    lats     = [spectrals[inight, ilat, ilon, ifilt, 0] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[inight, ilat, ilon, ifilt, 0]) == False]
                    lons     = [spectrals[inight, ilat, ilon, ifilt, 1] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[inight, ilat, ilon, ifilt, 1]) == False]
                    mus      = [spectrals[inight, ilat, ilon, ifilt, 2] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[inight, ilat, ilon, ifilt, 2]) == False]
                    rads     = [spectrals[inight, ilat, ilon, ifilt, 3] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[inight, ilat, ilon, ifilt, 3]) == False]
                    rad_errs = [spectrals[inight, ilat, ilon, ifilt, 4] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[inight, ilat, ilon, ifilt, 4]) == False]
                    waves    = [spectrals[inight, ilat, ilon, ifilt, 5] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[inight, ilat, ilon, ifilt, 5]) == False]

                    # Only write spxfile for longitudes and latitudes with spectral information
                    if lats:
                        if lons:
                            # Open textfile
                            with open(f"{dir}lat_{lats[0]}_lon_{lons[0]}_night{inight}.txt", 'w') as f:
                                create_av_reg_spx(f, lats, lons, mus, rads, rad_errs, waves)
                            # Open spxfile
                            with open(f"{dir}lat_{lats[0]}_lon_{lons[0]}_night{inight}.spx", 'w') as f:
                                create_av_reg_spx(f, lats, lons, mus, rads, rad_errs, waves)