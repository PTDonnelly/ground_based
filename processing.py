
from astropy.io import fits
from copy import deepcopy
import glob
import json
import numpy as np
import numpy.typing as npt
from os.path import exists
from pathlib import Path
from typing import Tuple, List, Dict

import netCDF4 as nc4
# from numba import jit
# from numba.experimental import jitclass
from sorcery import dict_of
from snoop import snoop

from config import Config
from testing import Clock, Profiler

class Preprocess:
    """Method-focused class responsible for reading FITS files
    and constructing a dictionary containing the FITS Header Data Unit (HDU)."""

    def __init__(self, filepath) -> None:
        self.filepath: object = filepath
        return  

    @staticmethod
    def pathify(filepaths: List[str]) -> List[str]:
        """Use pathlib.Path() to avoid problems with pathing between Windows (backslash)
        and Unix systems (forwardslash)."""
        return [Path(filepath) for filepath in filepaths]

    @classmethod
    def find_data(cls, data_directory: str, epoch: str) -> List[str]:        
        """Find observations based on pathing and epoch description in Config()
        and universalise pathnames with Path()."""
        # filepaths = glob.glob(f"{data_directory}{epoch}/recal_*.fits.gz")
        filepaths = [
            f"{data_directory}{epoch}/recal_wvisir_J7.9_2016-02-15T05:21:59.7428_Jupiter_clean_withchop.fits.gz",
            f"{data_directory}{epoch}/recal_wvisir_PAH1_2016-02-15T04:50:48.2300_Jupiter_clean_withchop.fits.gz"
            ]
        return cls.pathify(filepaths)
    
    @classmethod
    def get_data_files(cls) -> List[str]: 
        """Find and gather all files for a given epoch of VISIR observations.
            Make sure file hierarchy (defined in config.py) is pointing to the right place."""

        # Specify location of data and observing epoch
        data_directory = Config.data_directory
        epoch = Config.epoch
        return cls.find_data(data_directory, epoch)

    @classmethod
    def preprocess(cls, filepath: object) -> Dict[str, object]:
        """Use the Preprocess class to produce the FITS Header Data Units"""
        
        # Retrieve filename for each input file
        filename = cls.get_filenames(filepath)
        # Specify FITS extension to be read
        extension = 0

        # Read the Header Data Units (HDUs) of image and cylindrical maps (radiance, emission angle, doppler velocity)
        image = cls.read_fits(filename['image'], extension)
        radiance = cls.read_fits(filename['radiance'], extension)
        emission_angle = cls.read_fits(filename['emission_angle'], extension)
        doppler_velocity = cls.read_fits(filename['doppler_velocity'], extension)
        return dict_of(image, radiance, emission_angle, doppler_velocity)

    @staticmethod
    def get_filename_cmap(filename: str) -> str:
        return f"{filename}.cmap.gz"
    
    @staticmethod
    def get_filename_mu(filename: str) -> str:
        return f"{filename}.mu.gz"

    @staticmethod
    def get_filename_vdop(filename: str) -> str:
        if not exists(f"{filename}.vdop.gz"):
            return None
        else:
            return f"{filename}.vdop.gz"

    @classmethod
    def get_filenames(cls, filepath: object) -> dict:
        """Obtain .fits filepaths for reduced image, cylindrical map (cmap),
        emission angle map (mumap), and doppler velocity map (vdop)."""
        
        # Trim the gzip suffix (".gz") to access the full path (parents[0]) with filename (stem)
        filepath_without_suffix = f"{Config.data_directory}{Config.epoch}/{filepath.stem}"
        image = filepath
        radiance = cls.get_filename_cmap(filepath_without_suffix)
        emission_angle = cls.get_filename_mu(filepath_without_suffix)
        doppler_velocity = cls.get_filename_vdop(filepath_without_suffix)
        return dict_of(image, radiance, emission_angle, doppler_velocity) 

    @staticmethod
    def read_fits(filename: dict, extension: int) -> dict:
        """Read header and data from FITS file
        filename: name of the FITS file
        extension: extension of the FITS file"""
        
        # Check if filename exists (i.e. contains a string, i.e. file actually exists)
        if not filename:
            # If not, return None
            return {"header": None, "data": None}
        else:
            # If yes, return a dictionary of the FITS information
            with fits.open(filename) as hdu:
                header = hdu[extension].header
                data = hdu[extension].data
            return dict_of(header, data)

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

        image_hdu, radiance_hdu, emission_angle_hdu, dopppler_velocity_hdu = cls.unpack_hdu_group(hdu_group)

        # Calculate instrumental measurement errors of radiance
        error = cls.get_errors(image_hdu, type='flat')

        # Construct two-dimensional grids of latitude and longitude
        longitude_grid_1D, latitude_grid_1D, longitude_grid_2D, latitude_grid_2D = cls.make_spatial_grids(radiance_hdu)

        # Construct data maps from cylindrical maps
        radiance = cls.make_radiance_map(radiance_hdu)
        radiance_error = cls.make_radiance_error_map(radiance_hdu, error)
        emission_angle = cls.make_emission_angle_map(emission_angle_hdu)
        doppler_velocity = cls.make_doppler_velocity_map(dopppler_velocity_hdu)

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
    def make_spatial_grids(cls, header_data_unit: Dict[str, object]) -> Dict[str, npt.NDArray[np.float64]]:
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
    def make_radiance_map(cls, header_data_unit: Dict[str, object]) -> npt.NDArray[np.float64]:
        """Read in radiance values from cylindrical map and do unit conversion
        from native DRM units () into standard VISIR units ()"""

        # Coefficient needed to convert DRM units
        unit_conversion = 1e-7
        return cls.get_fits_data(header_data_unit) * unit_conversion
    
    @classmethod
    def make_radiance_error_map(cls, header_data_unit: Dict[str, object], error: float) -> npt.NDArray[np.float64]:
        """Read in radiance values from cylindrical map and construct a new cylindrical map
        with values radiance * error for each pixel"""
        return cls.make_radiance_map(header_data_unit) * error

    @classmethod
    def make_emission_angle_map(cls, header_data_unit: Dict[str, object]) -> npt.NDArray[np.float64]:
        """Read in cosine(emission angle) values from cylindrical map and do trignometric conversion
        from cosine(mu) to mu (in degrees)."""
        
        # Get cosine(emission angle) map from FITS data
        cosine_emission_angle = cls.get_fits_data(header_data_unit)

        # Calculate arc cosine of the map data
        emission_angle_radians = np.arccos(cosine_emission_angle)
        emission_angle_degrees = np.degrees(emission_angle_radians)
        return emission_angle_degrees

    @classmethod
    def make_doppler_velocity_map(cls, header_data_unit: Dict[str, object]) -> npt.NDArray[np.float64]:
        """Read in Doppler velocity values from cylindrical map."""
        return cls.get_fits_data(header_data_unit)

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
        
    @staticmethod
    def make_data_dict(filepaths: List[str]) -> List[dict]:
        """Create dictionary to store metadata and data products"""

        # Build dictionary template for each observation
        dict_template = {
            "filename": [],
            "data_products": []  
            }
        # Build a list of this dictionary to contain all files in filepaths
        data_dict = [deepcopy(dict_template) for _ in filepaths]
        return data_dict

    @classmethod
    def append_products_to_dataset(cls, ifile: int, filepath: object, data_products: Dict[str, npt.ArrayLike], dataset: List[dict]) -> List[dict]:
        """Add each file to an overall dictionary containing the entire dataset for a given epoch."""
        dataset[ifile]['filename'] = filepath.stem
        dataset[ifile]['data_products'] = data_products
        return
    
    @staticmethod
    def save_json(filepath: object, data_products: Dict[str, npt.ArrayLike]) -> None:
        """Generate a JSON file containing geometrically-registered cylindrical maps
        and metadata."""
        
        if Config.make_json == False:
            return
        else:
            file_stem = filepath.stem
            filename = file_stem.split('_clean_withchop.fits')[0]
            jsonfile = f"{Config.data_directory}{Config.epoch}/json/{filename}"
            print(jsonfile)
            with open(f"{jsonfile}.json", 'w') as f:
                json.dump(data_products, f, cls=NumpyEncoder)
            return

    @classmethod
    def save_netcdf(cls, filepath: object, data_products: Dict[str, npt.ArrayLike]) -> None:
        """Generate a NetCDF file containing geometrically-registered cylindrical maps
        and metadata."""
        if Config.make_netcdf == False:
            return
        else:
            # Construct filename from filepath
            file_stem = filepath.stem
            filename = file_stem.split('_clean_withchop.fits')[0]
            ncfile = f"{Config.data_directory}{Config.epoch}/netcdf/{filename}"

            # Instantiate a NetCDF4 Dataset object in write mode
            netcdf_object = nc4.Dataset(f"{ncfile}.nc", 'w', format='NETCDF4')
            netcdf_object.set_fill_off()

            # Add dimensions (prefix "d" refers to "dimension")
            dlongitude = netcdf_object.createDimension('dlongitude', len(data_products['spatial_grids']['longitude_grid_1D']))
            dlatitude = netcdf_object.createDimension('dlatitude', len(data_products['spatial_grids']['latitude_grid_1D']))
            
            # Add variables (prefix "a" refers to "variable")
            alatitude = netcdf_object.createVariable('alatitude', 'f4', ('dlatitude',))
            alongitude = netcdf_object.createVariable('alongitude', 'f4', ('dlongitude',))
            radiance = netcdf_object.createVariable('Radiance', 'f4', ('dlatitude', 'dlongitude',))
            radiance_error = netcdf_object.createVariable('Radiance error', 'f4', ('dlatitude', 'dlongitude',))
            emission_angle = netcdf_object.createVariable('Emission angle', 'f4', ('dlatitude', 'dlongitude',))
            doppler_velocity = netcdf_object.createVariable('Doppler velocity', 'f4', ('dlatitude', 'dlongitude',))
            
            # Assign values to variables
            alongitude[:] = data_products['spatial_grids']['longitude_grid_1D']
            alatitude[:] = data_products['spatial_grids']['latitude_grid_1D']
            radiance[:] = data_products['radiance']
            radiance_error[:] = data_products['radiance_error']
            emission_angle[:] = data_products['emission_angle']
            doppler_velocity[:] = data_products['doppler_velocity']
            
            # Close the object (and write the file)
            netcdf_object.close()









            return
    
    @classmethod
    def save(cls, filepath: object, data_products: Dict[str, npt.ArrayLike]) -> None:
        """Save the mapped data products to different file formats."""
        # Store data products in JSON format
        cls.save_json(filepath, data_products)
        # Store data products in NetCDF format
        cls.save_netcdf(filepath, data_products)
        return

class NumpyEncoder(json.JSONEncoder):
    """Needed for encoding numpy arrays within Python dictionaries
    for storage in json file. Pass this class name as a value to 
    the 'cls' parameter of json.dump()"""

    def default(self, object: npt.NDArray[np.float64]):
        if isinstance(object, np.ndarray):
            return object.tolist()
        return json.JSONEncoder.default(self, object)

class Dataset:
    """Method-focused class responsible for constructing and containing the data products of a given epoch."""

    def __init__(self) -> None:
        return
    
    @classmethod
    def create(cls) -> List[dict]:
        """Reads VISIR images and cylindrical maps (.fits format), does geometric registration, and optionally outputs alternate formats."""

        # Point to observations
        filepaths = Preprocess.get_data_files()

        # Create dictionary to store metadata and data products
        dataset = Process.make_data_dict(filepaths)

        # Read in VISIR FITS files and construct geographic data products
        for ifile, filepath in enumerate(filepaths):
            # print(f"Register map: {ifile+1} / {len(filepaths)}")

            # Retrieve Header Data Units (HDUs) from FITS files and group into dictionary
            hdu_group = Preprocess.preprocess(filepath)

            # Construct data products from Header Data Units (HDUs)
            data_products = Process.process(hdu_group)

            # Save data products to file
            Process.save(filepath, data_products)

            # Add data_products to the dictionary containing the entire dataset for this epoch."""
            Process.append_products_to_dataset(ifile, filepath, data_products, dataset)

        return dataset

    @staticmethod
    def read_data_products(data: Dict[str, npt.ArrayLike], reformed_data: Dict[str, npt.ArrayLike]) -> List:
        """Unpack individual data products form the large dataset. Specifically, convert 
        a dictionary of lists to a list of dictionaries."""   
        for key1, key2 in zip(reformed_data.keys(), data.keys()):
            if key1 != key2:
                raise ValueError("Data dictionaries do not contain the same entries.")
            else:
                reformed_data[key1].append(data[key2]) 
        return reformed_data
    
    @classmethod
    def read_data_products_from_dataset(cls) -> Dict[str, list]:
        """Create the dataset from the reduced data files then immediately deconstruct it 
        from a list of dictionaries of length "len(filepaths)" (i.e. number of files), and pull out the individual data_products. 
        Then reform it into a dictionary of lists, where each key is an individual data product (physical variable)
        and its corresponding value is a list of length "len(filepaths)"."""
        
        # Create dataset
        dataset = cls.create()
        
        # Create template dictionary for the reformed data (e.g. List[dict] -> Dict[list])
        reformed_data = {"longitude_grid_1D": [],
                         "latitude_grid_1D": [],
                         "longitude_grid_2D": [],
                         "latitude_grid_2D": [],
                         "radiance": [],
                         "radiance_error": [],
                         "emission_angle": [],
                         "doppler_velocity": [],
                         "metadata": []
        }

        # Upack each file from dataset
        for data in dataset:
            data_products = data["data_products"]
            cls.read_data_products(data_products, reformed_data)
        return reformed_data

    @classmethod
    def bin_central_meridian(cls) -> List[dict]:
        """Returns the meridional profiles from the data products returned by Dataset.create()."""

        # Reform the dataset dictionary
        reformed_data = cls.read_data_products_from_dataset()
        
        # Pull out variables
        metadata = reformed_data['metadata']
        print(np.shape(metadata))
        latitude = reformed_data['latitude_grid_2D']
        radiance = reformed_data['radiance']
        print(np.shape(radiance))

        # Isolate latitude grids along the central meridian (nlat x nfiles)
        
        profiles = reformed_data['latitude_grid_1D']
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
            profiles = cls.bin_central_meridian()
        elif Config.binning_scheme == 'centre to limb':
            profiles = cls.bin_centre_to_limb()
        elif Config.binning_scheme == 'regional':
            profiles = cls.bin_regional()
        return profiles

    @classmethod
    def make_calibration(cls, profiles: npt.ArrayLike) -> List[dict]:
        """Calibrates the data products returned by Dataset.create()."""
        pass 
    
    @classmethod
    def calibrate(cls) -> List[dict]:
        """Calibrates the data products returned by Dataset.create()."""
        
        if Config.calibrate == False:
            return
        else:
            # Get meridional profiles (calibration always uses central meridian profiles)
            profiles = cls.bin_central_meridian()
            cls.make_calibration(profiles)
            return
    
    @classmethod
    def make_plots(cls, profiles: npt.ArrayLike) -> List[dict]:
        """Plots the data products returned by Dataset.create()."""

        # Begin plotting as desired
        if Config.binning_scheme == 'central meridian':
            cls.plot_central_meridian(profiles)
        elif Config.binning_scheme == 'centre to limb':
            cls.plot_centre_to_limb(profiles)
        elif Config.binning_scheme == 'regional':
            cls.plot_regional(profiles)
        return profiles
        
    @classmethod
    def plot(cls) -> List[dict]:
        """Plots the data products returned by Dataset.create()."""
        
        if Config.plot == False:
            return
        else:
            profiles = cls.bin()
            cls.make_plots(profiles)
            return
    
    @classmethod
    def make_spx(cls, profiles: npt.ArrayLike) -> List[dict]:
        """Plots the data products returned by Dataset.create()."""
    
        # Begin binning as desired
        if Config.binning_scheme == 'central meridian':
            cls.bin_central_meridian(profiles)
        elif Config.binning_scheme == 'centre to limb':
            cls.bin_centre_to_limb(profiles)
        elif Config.binning_scheme == 'regional':
            cls.bin_regional(profiles)
        return profiles
    
    @classmethod
    def spx(cls) -> List[dict]:
        """Plots the data products returned by Dataset.create()."""
        
        if Config.plot == False:
            return
        else:
            profiles = cls.bin()
            cls.make_spx(profiles)
            return
        
def start_monitor() -> Tuple[object, object]:
    profiler = Profiler()
    profiler.start()
    # Start clock
    clock = Clock()
    clock.start()
    return profiler, clock

def stop_monitor(profiler: object, clock: object):
    # Stop clock
    clock.stop()
    clock.elapsed_time()
    # Stop monitoring code and 
    profiler.stop()
    profiler.save_profile_report()

def run_giantpipe():
    """Run the giantpipe VISIR Data Reduction Pipeline."""
    
    # Start monitoring code
    profiler, clock = start_monitor()

    # For calibration of the VISIR data
    Dataset.calibrate()

    # For plotting the VISIR data
    Dataset.plot()

    # For generating spectral files from the VISIR data
    Dataset.spx()

    completion = "\ngiantpipe is finished, have a great day."
    print(completion)

    stop_monitor(profiler, clock)
