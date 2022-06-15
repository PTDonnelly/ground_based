import os
import numpy as np
import Globals

def WriteSpx(spectrals):
    """Create spectral input for NEMESIS. Take calculated profiles and 
    populate .spxfile with radiances, measurement errors, and geometries."""
    
    def create_merid_spx(f, lats, LCMs, mus, rads, rad_errs, waves):
        """Write spxfile for meridional binning method """

        # Write first line of texfile with relevant formatting
        clat = lats[0]
        LCM  = np.mean(LCMs)
        nmu  = len(mus)
        f.write("{0:12.5f}  {1:12.5f}  {2:12.5f}  {3:12.5f}\n".format(0, clat, LCM, nmu))

        # Loop over NGEOM geometries (no. of geometries = no. of emission angle points)
        for igeom, mu in enumerate(mus):
            # Calculate NCONV spectral points (NCONV = no. of wavenumbers per geometry = 1 for merid binning)
            nconv = 1
            # I can't remember what NAV is... check the NEMESIS manual
            nav   = 1
            # The "angles line": this defines a "geometry" that holds a "spectrum"
            clat           = lats[igeom]
            clon           = LCMs[igeom]
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
            f.write("{0:12.5f}  {1:12.5f}  {2:12.5f}  {3:12.5f}  {4:12.5f}  {4:12.5f}\n".format(clat, clon, solar_ang, emission_angle, azimuth_angle, wgeom))
            f.write("{0:10.4f}  {1:15.6e}  {2:15.6e}\n".format(wave, rad, rad_err))

    # If subdirectory does not exist, create it
    dir = '../outputs/spxfiles/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    print('Creating spectra...')
    # Loop over latitudes to create one .spxfile per latitude
    for ilat in range(Globals.nlatbins):
        # Extract variables and throw NaNs
        lats     = [spectrals[ilat, ifilt, 0] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ilat, ifilt, 0]) == False]
        LCMs     = [spectrals[ilat, ifilt, 1] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ilat, ifilt, 1]) == False]
        mus      = [spectrals[ilat, ifilt, 2] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ilat, ifilt, 2]) == False]
        rads     = [spectrals[ilat, ifilt, 3] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ilat, ifilt, 3]) == False]
        rad_errs = [spectrals[ilat, ifilt, 4] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ilat, ifilt, 4]) == False]
        waves = [spectrals[ilat, ifilt, 5] for ifilt in range(Globals.nfilters) if np.isnan(spectrals[ilat, ifilt, 5]) == False]
        # Only write spxfile for latitudes with spectral information
        if lats:
            # Open textfile
            with open(f"../outputs/spxfiles/lat_{lats[0]}.txt", 'w') as f:
                create_merid_spx(f, lats, LCMs, mus, rads, rad_errs, waves)
            # Open spxfile
            with open(f"../outputs/spxfiles/lat_{lats[0]}.spx", 'w') as f:
                create_merid_spx(f, lats, LCMs, mus, rads, rad_errs, waves)
