# ground_based
Processing ground-based planetary observations for mapping and retrieval

This code currently deals entirely with observations from the VLT/VISIR instrument and assumes that images pre-processed using other software (namely NASA's DRM tool). This pre-processing performs cylindrical mapping and geometric registriation of the images using an artificial limb and orbital ephemerides.

This code simply takes these cylindrical maps of pixel values (or radiance if pre-calibrated) and emission angles to produce central meridian averages of flux at all latitudes for each observation and then all observations in given filter for the entire observing epoch. These steps are as follows:

Step 1: Read img, cmap and mufiles

Step 2: Geometric registration of pixel information

Step 3: Gather pixel information for all files

Step 4: Create central meridian average for each observation

Step 5: Create central meridian average for each wavelength

Step 6: Calibrate result of Step 5 to spacecraft data

Step 7: Calibrate individual cmaps to result of Step 6

Step 8: Store all cmap profiles and calibration parameters (individual and spectral)


These calibrated profiles (and associated calibration coefficients) can then be used to:

1. Plot meridional profiles

2. Create spectra from stored profiles (.spx files for input into NEMESIS)

3. Create cylindrical maps using the individual calibration coefficients derived from Step 7 (to help reduce global deviations from the mean in a given filter)

### UNDER CONSTRUCTION ###

1. Better calculation of errors
2. Saving of calibration coefficients
3. Reading arrays into PlotProfiles and WriteSpx
--a. Add the different "modes" to FindFiles
--b. Add the files to FindFiles under relevant
4. Plotting profiles in PlotProfiles
5. Writing spxfiles in WriteSpx
