import numpy as np
import numpy.typing as npt
from typing import List

import xarray as xr

from giantpipe import Config, Preprocess, Process

class Dataset:
    """Method-focused class responsible for constructing and containing the data products of a given epoch."""

    def __init__(self) -> None:
        return
    
    @classmethod
    def create(cls, filepaths: List[object]) -> List[dict]:
        """Reads VISIR images and cylindrical maps (.fits format), does geometric registration, and saves in NetCDF format."""
      
        # Read in VISIR FITS files and construct geographic data products
        for ifile, filepath in enumerate(filepaths):
            print(f"Register map: {ifile+1} / {len(filepaths)}")

            # Retrieve Header Data Units (HDUs) from FITS files and group into dictionary
            hdu_group = Preprocess.preprocess(filepath)

            # Construct data products from Header Data Units (HDUs)
            data_products = Process.process(hdu_group)

            # Save data products to file
            Process.save_netcdf(filepath, data_products)
        return

    @classmethod
    def ensure_no_missing_netcdf_files(cls) -> None:
        """Checks for netCDF dataset, if there are any missing (i.e. there are .fits versions but not .nc), creates the missing files."""
        
        missing_files = Preprocess.check_netcdf()
        if missing_files == None:
            print("\nAll NetCDF files present and accounted for.")
            pass
        else:
            print("\nSome NetCDF files are missing - going off to make some...")
            cls.create(missing_files)
            print("All NetCDF files present and accounted for.")

    @staticmethod
    def get_number_of_latitude_bins() -> int:
        grid_resolution = Config.grid_resolution
        latitude_range = Config.latitude_range
        return int((np.diff(latitude_range) / grid_resolution))

    @staticmethod
    def get_binning_extent(data: object):
        """Converts spatial extent in degrees to index positions"""

        # Get longitudinal extent in degrees
        min_longitude = int(data.attrs['LCMIII']) - Config.merid_width
        max_longitude = int(data.attrs['LCMIII']) + Config.merid_width
        
        # Get latitudinal extent in degrees
        min_latitude, max_latitude = Config.latitude_range
        if data.attrs['viewing'] == 1:
            min_latitude = -5
        elif data.attrs['viewing'] == -1:
            max_latitude = 5

        # Convert extents from degrees to indices
        min_latitude = (min_latitude + 90) * 2
        max_latitude = (max_latitude + 90) * 2
        min_longitude = (min_longitude * 2)
        max_longitude = (max_longitude * 2)

        # Ensure new index bounds conform to NetCDF grid
        if (min_latitude < 0) or (max_latitude > data.dims['latitude']) or (min_longitude < 0) or (max_longitude > data.dims['longitude']):
            raise ValueError("Array indices outside range of spatial grids in NetCDF file.")
        return min_longitude, max_longitude, min_latitude, max_latitude
    
    @classmethod
    def bin_central_meridian(cls) -> List[dict]:
        """Returns the meridional profiles from the data products contained in the NetCDF files."""

        # Point to NetCDF files
        filepaths = Preprocess.get_data_files(mode='nc')

        # Calculate number of bins in spatial grids
        number_of_latitude_bins = cls.get_number_of_latitude_bins()
        
        # Allocate memory for central meridian profiles (number of latitudes x number of files)
        profiles = np.empty((number_of_latitude_bins, len(filepaths), 3))
        profiles.fill(np.nan)

        # Open each file and extract data
        for ifile, filepath in enumerate(filepaths):
            with xr.open_dataset(filepath) as data:
                if data.dims['latitude'] != number_of_latitude_bins:
                    raise ValueError('NetCDF file has different grid to Config() definitions')

                # Get spatial extents of binning region
                min_longitude, max_longitude, min_latitude, max_latitude = cls.get_binning_extent(data)
                
                # Extract each data array
                radiance = data['radiance'].sel(longitude=slice(min_longitude, max_longitude), latitude=slice(min_latitude, max_latitude)).values
                radiance_error = data['radiance error'].sel(longitude=slice(min_longitude, max_longitude), latitude=slice(min_latitude, max_latitude)).values
                emission_angle = data['emission angle'].sel(longitude=slice(min_longitude, max_longitude), latitude=slice(min_latitude, max_latitude)).values

                # Store each 
                profiles[min_latitude:max_latitude, ifile, 0] = np.nanmean(radiance, axis=1)
                profiles[min_latitude:max_latitude, ifile, 1] = np.nanmean(radiance_error, axis=1)
                profiles[min_latitude:max_latitude, ifile, 2] = np.nanmean(emission_angle, axis=1)
        return profiles

    @classmethod
    def bin_centre_to_limb(cls) -> List[dict]:
        pass

    @classmethod
    def bin_regional(cls) -> List[dict]:
        pass

    @classmethod
    def bin(cls) -> List[dict]:
        """Runs the desired binning scheme on the data products returned by Dataset.create()."""

        # Begin binning as desired
        if Config.binning_scheme == 'central meridian':
            return cls.bin_central_meridian()
        elif Config.binning_scheme == 'centre to limb':
            return cls.bin_centre_to_limb()
        elif Config.binning_scheme == 'regional':
            return cls.bin_regional()

    @classmethod
    def radiometric_scaling(cls, profiles: npt.NDArray[np.float64]) -> None:
        """Calibrate ground-based data to published spacraft radiance profiles:
        Cassini/CIRS in the N-band and Voyager/ISS in the Q-band."""

        # Read calibration data


        # Scaling meridional profiles

        return
    
    @classmethod
    def run_calibration(cls) -> None:
        """Checks that all NetCDF files are present and calibrates the data products returned by Dataset.create()."""
        
        if Config.calibrate == False:
            return
        else:
            # Make sure all .fits files are present in .nc format
            cls.ensure_no_missing_netcdf_files()
            
            # Calibrate all observations
            print("\nCalibrating dataset:")

            # Get meridional profiles (calibration always uses central meridian profiles)
            profiles = cls.bin_central_meridian()
            
            # Perform radiometric scaling
            cls.radiometric_scaling(profiles)
            return
    
    @classmethod
    def make_plots(cls, profiles: npt.ArrayLike) -> None:
        """Plots the data products returned by Dataset.create()."""
        
        print("Plotting dataset:")

        # Get profiles
        profiles = cls.bin()

        # Begin plotting as desired
        if Config.binning_scheme == 'central meridian':
            cls.plot_central_meridian(profiles)
        elif Config.binning_scheme == 'centre to limb':
            cls.plot_centre_to_limb(profiles)
        elif Config.binning_scheme == 'regional':
            cls.plot_regional(profiles)
        return profiles
        
    @classmethod
    def run_plotting(cls) -> None:
        """Checks that all NetCDF files are present and runs plotting functions."""
        
        if Config.plot == False:
            return
        else:
            # Make sure all .fits files are present in .nc format
            cls.ensure_no_missing_netcdf_files()
            
            # Calibrate all observations
            cls.make_plots()
            return
    
    @classmethod
    def make_spx(cls, profiles: npt.ArrayLike) -> None:
        """Plots the data products returned by Dataset.create()."""

        print("Writing dataset to spectrum:")

        # Get profiles
        profiles = cls.bin()

        # Begin binning as desired
        if Config.binning_scheme == 'central meridian':
            cls.spx_central_meridian(profiles)
        elif Config.binning_scheme == 'centre to limb':
            cls.spx_centre_to_limb(profiles)
        elif Config.binning_scheme == 'regional':
            cls.spx_regional(profiles)
        return profiles
    
    @classmethod
    def run_spxing(cls) -> None:
        """Checks that all NetCDF files are present and runs spectral writing function."""
        
        if Config.spx == False:
            return
        else:
            # Make sure all .fits files are present in .nc format
            cls.ensure_no_missing_netcdf_files()
            
            # Calibrate all observations
            cls.make_spx()
            return