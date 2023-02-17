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

7. Calibrate result of Step 4 to result of Step 6

8. Perform calibration with these calculated parameters


These calibrated profiles (and associated calibration coefficients) can then be used to:

1. Plot meridional profiles, using PlotProfiles subroutine

2. Create spectra from stored profiles (.spx files for input into NEMESIS)

3. Create cylindrical maps using the individual calibration coefficients derived from Step 7 (to help reduce global deviations from the mean in a given filter)

### UNDER CONSTRUCTION ###

<<<<<<< Updated upstream
# Functions #
1. Add in a CTL binning scheme  as a test for whether the removal of the bulk CTL profile from each observation can create pretty global maps. To do this you also need 7-um vdop maps from DRM and the vdop correction. This is a rather time-expensive experiment so perhaps it should be lower prioriy.
2. Polar plotting of cylmaps
3. Optimise plotting codes?
=======
1. Writing spxfiles in WriteSpx
2. Plotting global maps in PlotMaps
3. Special calibraiton for 7-um filter: calibrate only to the equator of CIRS measurements
>>>>>>> Stashed changes

# Filestructure #
Implemented the above approach with subdirectories first rather than submodules first. Submodules can come later as and when we need more diverse applications to the code.
Thoughts for next version: consolidate codes into groups of functions and subfunctions for expansion in future
1. For Mapping: define a module with both mapping options inside so that we can simply import Mapping and call Mapping.Cylindrical or Mapping.Polar.
2. For Profiles: define the module MeridProfiles, such that MeridProfiles.Create and MeridProfiles.Calibrate. Eventually this could even become Binning.MeridProfile.Create, Binning.CTLProfile.Create, etc. (depending on what is needed).
4. For Winds: define the module PlotWinds, such that PlotWinds.Pseudo, PlotWinds.Retrieved, PlotWinds.DYNAMICO etc.
5. For reading input files: define the module ReadInput, such that ReadInput.Gravity, ReadInput.Calib, ReadInput.ZonalWind, ReadInput.Fits.
6. For writing output files: define the module WriteOutput, such that WriteOutput.MeridProfiles, WriteOutput.Calibration, WriteOutput.Spx.
7. For now, RegisterMaps is fine since we are dealing with VISIR observations in .fits format. When the instrument or data format changes, we will need to expand and maybe generalise.

We can also ask ourselves, do we want plotting routines to be nested inside the relevant area? I.e. Mapping.Cylindrical.Plot, MeridProfiles.Plot, Winds.Pseudo.Plot? Or keep all plotting routines in one module with different areas nested inside that? I.e. Plot.MeridProfiles, Plot.Winds.Pesudo, Plot.Mapping.Cylindrical? Maybe the secondway is better because then you can have the function Mapping.Cylindrical() and at the very end you can call Plot.Mapping.Cylindrical()? I am not sure at this point.

# UPDATE #

Implemented the above approach with subdirectories first rather than submodules first. Submodules can come later as and when we have more diverse applications to the code.