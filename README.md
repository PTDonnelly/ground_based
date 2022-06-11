# ground_based
Processing ground-based planetary observations for mapping and retrieval

This code currently deals entirely with observations from the VLT/VISIR instrument and assumes that images pre-processed using other software (namely NASA's DRM tool). This pre-processing performs cylindrical mapping and geometric registriation of the images using an artificial limb and orbital ephemerides.

This code simply takes these cylindrical maps of pixel values (or radiance if pre-calibrated) and emission angles to produce central meridian averages of flux at all latitudes for each observation and then all observations in given filter for the entire observing epoch. These steps are as follows:

1. Read files: images (img), cylindrical maps (cmap) and emission angle maps (mumap)

2. Geometric registration of pixel information

3. Gather pixel information for all files

4. Create central meridian average for each observation

5. Create central meridian average for each wavelength

6. Calibrate result of Step 5 to spacecraft data

7. Calibrate individual cmaps to result of Step 6

8. Store all cmap profiles and calibration parameters (individual and spectral)


These calibrated profiles (and associated calibration coefficients) can then be used to:

1. Plot meridional profiles, using PlotProfiles subroutine

2. Create spectra from stored profiles (.spx files for input into NEMESIS)

3. Create cylindrical maps using the individual calibration coefficients derived from Step 7 (to help reduce global deviations from the mean in a given filter)

### UNDER CONSTRUCTION ###

1. Better calculation of errors (Done)
2. Saving of calibration coefficients (Done)
3. Reading arrays into PlotProfiles and WriteSpx: (a) Add the different "modes" to FindFiles (Done), (b) Add the files to FindFiles under relevant (Done)
4. Plotting profiles in PlotProfiles (Done)
5. Writing spxfiles in WriteSpx
6. Plotting global maps in PlotMaps
7. Plotting polar projection maps in PlotPoles (to be create)
8. For maps, define a module with both mapping options inside so that we can simply import PlotMaps and call PlotMaps.Cylindrical or PlotMaps.Polar for whatever case we need. This is also expandable in future.
9. Repeat with MeridProfiles? I.e. MeridProfiles.Create and MeridProfiles.Calibrate? To cut down on the number of separate functions in the directory.
10. Repeat with VisirFilters? I.e. VisirFilters.Wavelengths and VisirFilters.Wavenumbers? Although this is a non-ideal and dirty fix anyway, so maybe that is a chance to generalise those functions. Possibly combine with SetWave?