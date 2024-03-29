"""Manual define global parameters that configure many parts of the pipeline,
   such as the binning, calibration, plotting, and writing."""

import numpy as np
nx, ny = 720, 360                               # Dimensions of an individual cylindrical map
## Central Meridian binning
latrange    = -90, 90          			          # Latitude range for binning pixels (planetographic)
latstep     = 1                                 # Latitude increment for binning pixels (planetographic)
latgrid     = np.arange(-89.5, 90, latstep)      # Latitude range from pole-to-pole
nlatbins    = len(latgrid)                       # Number of latitude bins
merid_width = 80 #20 #80 #30 #60                                 # Longitude range about the central meridian for averaging
                                                 # 30 for bin_cmerid, 80 for bin_cpara, used also for bin_region and bin_av_region
## "Central Parallel" binning
lonrange    = 360, 0            		             # Longitude range for binning pixels (Sys III)
lonstep     = 1                                 # Longitude increment for binning pixel 
longrid     = np.arange(360, 0, -lonstep)        # Longitude range
nlonbins    = len(longrid)                       # Number of longitude bins
para_width  = 15 #1 #15 #10 #2                                  # Latitude range about the central parallel for averaging
                                                 # unused for bin_cmerid, 1 for bin_cpara, used also for bin_region and bin_av_region
LCP = -80                                        # Latitude Central Parallel (equivalent to LCMIII in cmaps)
## Regional and Regional Average binning
lat_target  = -75 #-80 #-20                                # Latitude target for the regional and regional average binning schemes (GRS=-20, S-AURORA=-80)
lon_target = 360 #180 #360 #157                                 # Longitude target for the regional and regional average binning schemes (GRS=157, S-AURORA=360)

nfilters    = 13                                 # Set manually if using irregularly-sampled data
# mu_max      = 80.0                 		       # Maximum emission angle

