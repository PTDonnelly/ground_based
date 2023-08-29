import numpy as np
import numpy.typing as npt
from scipy import interpolate
from typing import Tuple, List, Dict

import netCDF4 as nc4
# from numba import jit
# from numba.experimental import jitclass
from sorcery import dict_of
from snoop import snoop
import xarray as xr

from giantpipe import Config, Preprocess

class Process:
    """Method-focused class responsible for accessing contents of FITS files
    from the FITS Header Data Unit (HDU).
    
    header_data_unit: dict = {"header": header, "data": data}"""

    def __init__(self, header_data_unit) -> None:
        self.header_data_unit: dict = header_data_unit
        return

    @classmethod
    def process(cls, hdu_group: Dict[str, dict]) -> Dict[str, npt.ArrayLike]:
        """Use the Process class to produce the FITS Header Data Units to produce
        geographic metadata and data products, then pack these into a dictionary
        for file storage."""

        image_hdu, radiance_hdu, emission_angle_hdu, doppler_velocity_hdu = cls.unpack_hdu_group(hdu_group)

        # Calculate instrumental measurement errors of radiance
        error = cls.get_errors(image_hdu, type='flat')

        # Construct two-dimensional grids of latitude and longitude
        longitude_grid_1D, latitude_grid_1D, longitude_grid_2D, latitude_grid_2D = cls.make_spatial_grids(radiance_hdu)

        # Construct data maps from cylindrical maps
        radiance = cls.make_radiance_map(radiance_hdu, doppler_velocity_hdu)
        radiance_error = cls.make_radiance_error_map(radiance, error)
        emission_angle = cls.make_emission_angle_map(emission_angle_hdu)
        doppler_velocity = cls.get_fits_data(doppler_velocity_hdu)

        # Construct metadata about observation
        metadata = cls.make_metadata(radiance_hdu)

        # Pack data into a dictionary
        return dict_of(longitude_grid_1D, latitude_grid_1D, longitude_grid_2D, latitude_grid_2D, radiance, radiance_error, emission_angle, doppler_velocity, metadata)
    
    staticmethod
    def unpack_hdu_group(hdu_group: Dict[str, dict]) -> List[dict]:
        return [value for _, value in hdu_group.items()]

    @staticmethod
    def get_header_contents(header: dict, header_parameter: str) -> str:
        return header[header_parameter]

    @staticmethod
    def get_fits_header(header_data_unit: Dict[str, dict]) -> object:
        return header_data_unit['header']
    
    @staticmethod
    def get_fits_data(header_data_unit: Dict[str, dict]) -> npt.NDArray[np.float64]:
        return header_data_unit['data']

    @classmethod
    def get_viewing(cls, header: object) -> int:
        """Determine the viewing of the image, needed for selecting data from the reliable
        hemisphere later. Save flag depending on Northern (1) or Southern (-1) viewing."""
        
        # Pull out telescope pointing information from FITS header
        chop_parameter = 'HIERARCH ESO TEL CHOP POSANG'
        ada_parameter = 'HIERARCH ESO ADA POSANG'
        chop_angle: float = cls.get_header_contents(header, chop_parameter)
        ada_angle: float  = cls.get_header_contents(header, ada_parameter) + 360
        return 1 if (chop_angle == ada_angle) else -1

    @staticmethod
    def error_type_flat() -> float:
        """Use a flat 5% uncertainty on each radiance value. This is the maximum expected
        value for the VISIR images based on previous studies."""
        return 0.05
    
    @classmethod
    def error_type_statistical(cls, header_data_unit: Dict[str, dict]) -> float:
        """Calculate an uncertainty for each individual image derived from the
        standard deviation of the background sky flux."""
        
        # Access FITS header and data
        header = cls.get_fits_header(header_data_unit)
        data = cls.get_fits_data(header_data_unit)

        # Determine Northern (1) or Southern (-1) viewing
        viewing = cls.get_viewing(header)
        
        # Only use unaffected hemisphere
        ny, nx = np.shape(data)
        img = data[int(ny/2):-1, :] if viewing == 1 else data
        img = data[0:int(ny/2)-1, :] if viewing == -1 else data
        
        # Set a radiance threshold to separate "planet" from "sky" flux
        thresh = 0.1
        
        # Calculate mean radiance value of the image
        mean_flux = np.mean(img)
        
        # Isolate sky
        keep  = (img < thresh * mean_flux)
        sky   = img[keep]
        error = np.std(sky)
        return error

    @classmethod
    def get_errors(cls, header_data_unit: Dict[str, object], type: str) -> float:
        """Calculate radiance error from the variability background sky flux"""

        # Run relevant error calculation mode
        if type == "statistical":
            # Calculate standard deviation of background sky (unique for each image)
            error = cls.error_type_statistical(header_data_unit)
        elif type == "flat":
            # Set to a constant typical value
            error = cls.error_type_flat()
        return error

    @staticmethod
    def make_grid_1D(grid_range: Tuple[float, float], grid_resolution: float) -> npt.NDArray[np.float64]:
        """Generate a one-dimensional spatial grid for geometric registration."""
        return np.arange(grid_range[0], grid_range[1], grid_resolution)

    @classmethod
    def make_grid_2D(cls, grid_type: str, x_size: float, y_size: float, grid_resolution: float) -> npt.NDArray[np.float64]:
        """Generate a spatial grid for geometric registration."""
        
        # Determine type of spatial grid being created
        if grid_type == 'longitude':
            number_of_repititions = y_size
            grid_range = Config.longitude_range
        elif grid_type == 'latitude':
            number_of_repititions = x_size
            grid_range = Config.latitude_range
        
        # Create two-dimensional gird from one-dimensional grid
        grid_1D = cls.make_grid_1D(grid_range, grid_resolution)
        grid_2D = np.tile(grid_1D, (number_of_repititions, 1))
        if grid_type == 'latitude':
            grid_2D = np.transpose(grid_2D)
        return grid_2D

    @staticmethod
    def get_image_resolution(x_size: int, y_size: int) -> float:
        """Check that image dimensions both conform to the same resolution.
        Check each dimension with respect to the number of longitudes and latitudes
        on a 1-degree grid (360 and 180, respectively)."""
        if (360 / x_size) != (180 / y_size):
            raise ValueError("Image dimensions are not on a common grid or do not conform to the same resolution.")
        else:
            return (360 / x_size)

    @classmethod
    def make_spatial_grids(cls, header_data_unit: Dict[str, npt.NDArray[np.float64]]) -> Dict[str, npt.NDArray[np.float64]]:
        "Read in raw cylindrical maps and perform geometric registration"

        # Access FITS header information
        header = cls.get_fits_header(header_data_unit)

        # Get image dimensions from FITS header
        x_parameter = "NAXIS1"
        y_parameter = "NAXIS2"
        x_size = cls.get_header_contents(header, x_parameter)
        y_size = cls.get_header_contents(header, y_parameter)

        # Calculate pixel resolution of image map
        image_resolution = cls.get_image_resolution(x_size, y_size)

        # Construct two-dimensional spatial grids
        longitude_grid_1D = cls.make_grid_1D(Config.longitude_range, image_resolution)
        latitude_grid_1D = cls.make_grid_1D(Config.latitude_range, image_resolution)

        # Construct two-dimensional spatial grids
        longitude_grid_2D = cls.make_grid_2D('longitude', x_size, y_size, image_resolution)
        latitude_grid_2D = cls.make_grid_2D('latitude', x_size, y_size, image_resolution)
        return longitude_grid_1D, latitude_grid_1D, longitude_grid_2D, latitude_grid_2D

    @classmethod
    def is_filter_7_microns(cls, header_data_unit: Dict[str, npt.NDArray[np.float64]]) -> bool:
        """Returns a boolean value indicating whether or not the file is 7.9-micron data."""
        
        # Access FITS header information
        header = cls.get_fits_header(header_data_unit)
        # Read filter wavelength
        wavelength = cls.get_header_contents(header, 'lambda')
        return wavelength == float(7.9)
   
    @staticmethod
    def interpolate_doppler_profile():
        """Read in Doppler velocity look-up table to get correction factor and create interpolate object for the Radiance-Doppler profile."""
        f = np.loadtxt(f"{Config.input_directory}doppler_shift_correction_profile.txt", dtype=float)
        correction_factor, doppler_velocity = f[:, 0], f[:, 1]
        return interpolate.interp1d(doppler_velocity, correction_factor)
    
    @classmethod
    def doppler_correction_of_radiance(cls, radiance_hdu: Dict[str, npt.NDArray[np.float64]], doppler_velocity_hdu: Dict[str, npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
        
        # Read in radiance and Doppler velocity map
        radiance = cls.get_fits_data(radiance_hdu)
        doppler_velocity = cls.get_fits_data(doppler_velocity_hdu)

        # Get Doppler velocity interpolation function
        interpolation_f = cls.interpolate_doppler_profile()
        return radiance / interpolation_f(doppler_velocity)

    @classmethod
    def make_radiance_map(cls, radiance_hdu: Dict[str, npt.NDArray[np.float64]], doppler_velocity_hdu: Dict[str, npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
        """Read in radiance values from cylindrical map and do unit conversion
        from native DRM units () into standard VISIR units ()"""

        check = cls.is_filter_7_microns(radiance_hdu)
        if check == False:
            # Read in raw radiance map
            radiance = cls.get_fits_data(radiance_hdu)
        else:
            # Do Doppler correction of 7.9-micron radiances
            radiance = cls.doppler_correction_of_radiance(radiance_hdu, doppler_velocity_hdu)
        
        # Coefficient needed to convert DRM units
        unit_conversion = 1e-7
        return radiance * unit_conversion
    
    @staticmethod
    def make_radiance_error_map(radiance: npt.NDArray[np.float64], error: float) -> npt.NDArray[np.float64]:
        """Read in radiance values from cylindrical map and construct a new cylindrical map
        with values radiance * error for each pixel"""
        return radiance * error

    @classmethod
    def make_emission_angle_map(cls, header_data_unit: Dict[str, npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
        """Read in cosine(emission angle) values from cylindrical map and do trignometric conversion
        from cosine(mu) to mu (in degrees)."""
        
        # Get cosine(emission angle) map from FITS data
        cosine_emission_angle = cls.get_fits_data(header_data_unit)

        # Calculate arc cosine of the map data
        emission_angle_radians = np.arccos(cosine_emission_angle)
        emission_angle_degrees = np.degrees(emission_angle_radians)
        return emission_angle_degrees
    
    @classmethod
    def make_metadata(cls, header_data_unit: Dict[str, object]) -> Dict[str, str]:
        """Pull important information about the observation from the FITS header
        and pack into dictionary to pass to the final data products (FITS and NetCDF)."""

        # Access FITS header information
        header = cls.get_fits_header(header_data_unit)

        # Get obervation ephemeris from FITS header
        date_time_parameter = "DATE-OBS"
        LCMIII_parameter = "LCMIII"
        wavelength_parameter = "lambda"
        date_time = cls.get_header_contents(header, date_time_parameter) # Date and time of observation (YYYY-MM-DDTHH:MM:SS:ssss)
        LCMIII = cls.get_header_contents(header, LCMIII_parameter) # Longitude of the Central Meridian (System III West)
        wavelength = cls.get_header_contents(header, wavelength_parameter) # Filter wavelength (in units of microns)
        viewing = cls.get_viewing(header) # Observation viewing (1: North or -1: South)
        return dict_of(date_time, LCMIII, wavelength, viewing)
      
    @classmethod
    def save_netcdf(cls, filepath: object, data_products: Dict[str, npt.ArrayLike]) -> None:
        """Generate a NetCDF file containing geometrically-registered cylindrical maps and metadata."""
        
        # Construct the filename of the netCDF file
        ncfile = Preprocess.fitsname_to_ncname(filepath)
        
        # Instantiate a NetCDF4 Dataset object in write mode
        netcdf_object = nc4.Dataset(f"{ncfile}", 'w', format='NETCDF4')
        netcdf_object.set_fill_off()

        # Add dimension (these objects are used in the dimension arguments to createVariable below, the names do not clash)
        _ = netcdf_object.createDimension('longitude', len(data_products['longitude_grid_1D']))
        _ = netcdf_object.createDimension('latitude', len(data_products['latitude_grid_1D']))
        
        # Add variables (prefix "a" refers to "variable")
        latitudes = netcdf_object.createVariable('latitudes', 'f4', ('latitude',))
        longitudes = netcdf_object.createVariable('longitudes', 'f4', ('longitude',))
        radiance = netcdf_object.createVariable('radiance', 'f4', ('latitude', 'longitude',))
        radiance_error = netcdf_object.createVariable('radiance error', 'f4', ('latitude', 'longitude',))
        emission_angle = netcdf_object.createVariable('emission angle', 'f4', ('latitude', 'longitude',))
        doppler_velocity = netcdf_object.createVariable('doppler velocity', 'f4', ('latitude', 'longitude',))
        
        # Assign values to variables
        longitudes[:] = data_products['longitude_grid_1D']
        latitudes[:] = data_products['latitude_grid_1D']
        radiance[:] = data_products['radiance']
        radiance_error[:] = data_products['radiance_error']
        emission_angle[:] = data_products['emission_angle']
        doppler_velocity[:] = data_products['doppler_velocity']

        # Set global NetCDF attributes
        netcdf_object.setncatts(data_products['metadata'])

        # Close the object (and write the file)
        netcdf_object.close()
        return