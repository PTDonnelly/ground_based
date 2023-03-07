from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    """Define flags to configure the execution of the pipeline"""
    
    ########## Module-level parameters
    ##### Running giantpipe
    # For calibrating mapped products (from DRM or Oliver's tool), you should only have to do this once.
    calibrate: bool = True
    # For generating plots of the dataset
    plot: bool = False
    # For generating spectra of the dataset (.spx files for NEMESIS input)
    spx: bool = False
    
    ########## Testing parameters
    clock: bool = True            # Used for timing the execution of the code
    profiler: bool = True        # Used for monitoring execution frequency and duration of code

    ########## Input flags
    ##### Filepaths
    # Point to data directory
    data_directory: str = "/Users/ptdonnelly/Documents/Research/data/visir/"
    # data_directory: str = "C:/Users/padra/Documents/Research/data/visir/"
    # Point to pipeline inputs directory
    input_directory: str = "/Users/ptdonnelly/Documents/Research/github/ground_based_2/inputs/"
    # Point to specific epoch directory
    epoch: str = "2016feb"
    ##### Source of data: "fits" or "json" or "nc" (the latter two are not currently used)
    data_source: str = "fits"
    
    ########## Spatial parameters
    ##### Spatial grids
    # Resolution of geographic grids (0.5 = half-degree resolution)
    grid_resolution = 0.5
    # Maximum extent of spatial grids for construction
    latitude_range = -90, 90
    longitude_range = 0, 360
    emission_angle_range = 0, 90
    ##### Binning-specific parameters
    # For binning the maps by different schemes (type: None) or (type: str = "central meridian", "centre to limb", "regional" etc.)
    binning_scheme: str = 'central meridian'
    # Central meridian binning width (in plus-minus degrees longitude around the LCMIII)
    merid_width = 20

    #### Output flags

    #################### OLD
    #         
    # # Input flags
    # calibrate = False # Read raw data and calibrate
    # datasource = 'fits' # Source of data: local cmaps ('fits') or local numpy arrays ('npy')




    # nx, ny = 720, 360                               # Dimensions of an individual cylindrical map
    # ## Central Meridian binning
    # latrange    = -90, 90          			          # Latitude range for binning pixels (planetographic)
    # latstep     = 30                                # Latitude increment for binning pixels (planetographic)
    # latgrid     = np.arange(-89.5, 90, latstep)      # Latitude range from pole-to-pole
    # nlatbins    = len(latgrid)                       # Number of latitude bins
    # merid_width = 60 #20 #80 #30 #60                                 # Longitude range about the central meridian for averaging
    #                                                 # 30 for bin_cmerid, 80 for bin_cpara, used also for bin_region and bin_av_region
    # ## "Central Parallel" binning
    # lonrange    = 360, 0            		             # Longitude range for binning pixels (Sys III)
    # lonstep     = 1                                 # Longitude increment for binning pixel 
    # longrid     = np.arange(360, 0, -lonstep)        # Longitude range
    # nlonbins    = len(longrid)                       # Number of longitude bins
    # para_width  = 2 #10 #2                                  # Latitude range about the central parallel for averaging
    #                                                 # unused for bin_cmerid, 1 for bin_cpara, used also for bin_region and bin_av_region
    # LCP = -80                                        # Latitude Central Parallel (equivalent to LCMIII in cmaps)
    # ## Regional and Regional Average binning
    # lat_target  = -80 #-20                                # Latitude target for the regional and regional average binning schemes (GRS=-20, S-AURORA=-80)
    # lon_target = 360 #157                                 # Longitude target for the regional and regional average binning schemes (GRS=157, S-AURORA=360)

    # ## Centre-to-limb binning
    # murange    = 0, 90          			           # Emission angle range for binning pixels (planetographic)
    # mustep     = 30                                 # Emission angle increment for binning pixels (planetographic)
    # mugrid     = np.arange(0.5, 90, mustep)       # Emission angle range from pole-to-pole
    # nmubins    = len(mugrid)                       # Number of latitude bins

    # nfilters    = 8                                 # Set manually if using irregularly-sampled data
    # # mu_max      = 80.0                 		       # Maximum emission angle

