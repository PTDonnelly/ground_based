import numpy as np

class BinningInputs():
    # mu_max      = 80.0                 		     # Maximum emission angle
    latrange    = -90, 90          			         # Latitude range for binning pixels (planetographic)
    latstep     = 1                                  # Latitude increment for binning pixels (planetographic)
    latgrid     = np.arange(-89.5, 90, latstep)      # Latitude range from pole-to-pole
    nlatbins    = len(latgrid)
    lonrange    = 360, 0            		         # Longitude range for binning pixels (Sys III)
    merid_width = 30                                 # Longitude range about the central meridian for averaging
    nfilters    = 13                                 # Set manually if using irregularly-sampled data