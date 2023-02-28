
from astropy.io import fits
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import Tuple, List, Dict
import icecream as ic
# from numba import jit
# from numba.experimental import jitclass

from config import Config

# @jitclass(jitPreprocess)
class Preprocess:
    """Method-focused class responsible for reading FITS files
    and constructing a dictionary containing the FITS Header Data Unit (HDU)."""

    def __init__(self, filepath):
        self.filepath: str = filepath      

    @classmethod
    def preprocess(cls, filepath: str) -> Dict[str, object]:
        """Use the Preprocess class to produce the FITS Header Data Units"""
        
        # Retrieve filename for each input file
        filename = cls.get_filename(filepath=filepath)
        # Specify FITS extension to be read
        extension = 0

        # Read the Header Data Units (HDUs) input file and cylindrical maps of radiance and emission angle
        image = cls.read_fits(filename=filename['image'], extension=extension)
        radiance = cls.read_fits(filename=filename['radiance'], extension=extension)
        emission_angle = cls.read_fits(filename=filename['emission_angle'], extension=extension)
        
        # Pack Header Data Units together
        hdu_group = cls.pack_hdu_group(image=image, radiance=radiance, emission_angle=emission_angle)
        return hdu_group

    @staticmethod
    def get_filename(filepath: str) -> dict:
        """Obtain .fits filenames for image, cylindrical map (cmap)
        and emission angle map (mumap)."""

        fname = filepath.split('.gz')
        radiance = f"{fname[0]}.cmap.gz"
        emission_angle = f"{fname[0]}.mu.gz"
        file = {"image": filepath, "radiance": radiance, "emission_angle": emission_angle}
        return file

    @staticmethod
    def read_fits(filename: dict, extension: int) -> dict:
        """Read header and data from FITS file
        filename: name of the FITS file
        extension: extension of the FITS file"""

        with fits.open(filename) as hdu:
            header = hdu[extension].header
            data = hdu[extension].data
            header_data_dict = {"header": header, "data": data}
        return header_data_dict

    @staticmethod
    def pack_hdu_group(image: Dict[str, object], radiance: Dict[str, object], emission_angle: Dict[str, object]) -> Dict[str, npt.ArrayLike]:
        return {"image": image, "radiance": radiance, "emission_angle": emission_angle}

# @jitclass(jitProcess)
class Process:
    """Method-focused class responsible for accessing contents of FITS files
    from the FITS Header Data Unit (HDU).
    
    header_data_unit: dict = {"header": header, "data": data}"""

    def __init__(self, header_data_unit):
        self.header_data_unit: dict = header_data_unit      

    @classmethod
    def process(cls, hdu_group: Dict[str, object]) -> Dict[str, npt.ArrayLike]:
        """Use the Process class to produce the geographic metadata and data products."""

        image_hdu, radiance_hdu, emission_angle_hdu = cls.unpack_hdu_group(hdu_group=hdu_group)

        # Calculate instrumental measurement errors of radiance
        error = cls.get_errors(header_data_unit=image_hdu, type='statistical')

        # Construct data maps from cylindrical maps
        radiance = cls.make_radiance_map(header_data_unit=radiance_hdu)
        radiance_error = cls.make_radiance_error_map(header_data_unit=radiance_hdu, error=error)
        emission_angle = cls.make_emission_angle_map(header_data_unit=emission_angle_hdu)

        # Construct two-dimensional grids of latitude and longitude
        spatial_grids = cls.make_spatial_grids(header_data_unit=radiance_hdu)

        # Construct metadata about observation
        metadata = cls.make_metadata(header_data_unit=radiance_hdu)

        # Pack data into a dictionary
        packed_data = cls.pack_data(radiance=radiance, radiance_error=radiance_error, emission_angle=emission_angle, spatial_grids=spatial_grids, metadata=metadata)
        return packed_data
    
    @staticmethod
    def get_header_contents(header: dict, header_parameter: str) -> str:
        return header[header_parameter]

    @staticmethod
    def get_header(header_data_unit: Dict[str, object]) -> List[str]:
        return header_data_unit['header']
    
    @staticmethod
    def get_data(header_data_unit: Dict[str, object]) -> npt.NDArray[np.float64]:
        return header_data_unit['data']

    @classmethod
    def get_viewing(cls, header: object) -> int:
        """Determine the viewing of the image, needed for selecting data
        from the good hemisphere later"""
        
        # Pull out telescope pointing information from FITS header
        chop_parameter = 'HIERARCH ESO TEL CHOP POSANG'
        ada_parameter = 'HIERARCH ESO ADA POSANG'
        chop_angle: float = cls.get_header_contents(header=header, header_parameter=chop_parameter)
        ada_angle: float  = cls.get_header_contents(header=header, header_parameter=ada_parameter) + 360
        
        # Save flag depending on Northern (1) or Southern (-1) viewing
        view = 1 if (chop_angle == ada_angle) else -1
        return view

    @staticmethod
    def error_type_flat() -> float:
        """Use a flat 5% uncertainty on each radiance value. This is the maximum expected
        value for the VISIR images based on previous studies."""
        return 0.05
    
    @classmethod
    def error_type_statistical(cls, header_data_unit: Dict[str, object]) -> float:
        """Calculate an uncertainty for each individual image derived from the
        standard deviation of the background sky flux."""
        
        # Access FITS header and data
        header = cls.get_header(header_data_unit=header_data_unit)
        data = cls.get_data(header_data_unit=header_data_unit)

        # Determine Northern (1) or Southern (-1) viewing
        viewing = cls.get_viewing(header=header)
        
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
        grid_1D = cls.make_grid_1D(grid_range=grid_range, grid_resolution=grid_resolution)
        grid_2D = np.tile(grid_1D, (number_of_repititions, 1))
        if grid_type == 'latitude':
            grid_2D = np.transpose(grid_2D)
        return grid_2D

    @classmethod
    def make_radiance_map(cls, header_data_unit: Dict[str, object]) -> npt.NDArray[np.float64]:
        """Read in radiance values from cylindrical map and do unit conversion
        from native DRM units () into standard VISIR units ()"""

        # Access FITS data extension for radiance
        data = cls.get_data(header_data_unit=header_data_unit)
        
        # Coefficient needed to convert DRM units
        unit_conversion = 1e-7

        # Scale entire cylindrical map
        radiance = data * unit_conversion
        return radiance
    
    @classmethod
    def make_radiance_error_map(cls, header_data_unit: Dict[str, object], error: float) -> npt.NDArray[np.float64]:
        """Read in radiance values from cylindrical map and construct a new cylindrical map
        with values radiance * error for each pixel"""

        # Access FITS data extension for radiance
        data = cls.make_radiance_map(header_data_unit=header_data_unit)

        # Scale entire cylindrical map
        radiance_error = data * error
        return radiance_error

    @classmethod
    def make_emission_angle_map(cls, header_data_unit: Dict[str, object]) -> npt.NDArray[np.float64]:
        """Read in cosine(emission angle) values from cylindrical map and do trignometric conversion
        from cosine(mu) to mu (in degrees)."""
        
        # Access FITS data extension for emission angle
        data = cls.get_data(header_data_unit=header_data_unit)

        # Get cosine(emission angle) map from FITS data
        cosine_emission_angle = data

        # Calculate arc cosine of the map data
        emission_angle_radians = np.arccos(cosine_emission_angle)
        emission_angle_degrees = np.degrees(emission_angle_radians)
        return emission_angle_degrees

    @staticmethod
    def get_image_resolution(x_size: int, y_size: int) -> float:
        """Check that image dimensions are equal and both conform to the same resolution"""

        if (360 / x_size) != 180 / y_size:
            quit()
        else:
            image_resolution = (360 / x_size)
        return image_resolution

    @classmethod
    def make_spatial_grids(cls, header_data_unit: Dict[str, object]) -> Dict[str, npt.NDArray[np.float64]]:
        "Read in raw cylindrical maps and perform geometric registration"

        # Access FITS header information
        header = cls.get_header(header_data_unit=header_data_unit)

        # Calculate bounds of image map
        x_size = cls.get_header_contents(header=header, header_parameter='NAXIS1')
        y_size = cls.get_header_contents(header=header, header_parameter='NAXIS2')

        # Calculate pixel resolution of image map
        image_resolution = cls.get_image_resolution(x_size=x_size, y_size=y_size)

        # Construct two-dimensional spatial grids
        longitude_grid_2D = cls.make_grid_2D(grid_type='longitude', x_size=x_size, y_size=y_size, grid_resolution=image_resolution)
        latitude_grid_2D = cls.make_grid_2D(grid_type='latitude', x_size=x_size, y_size=y_size, grid_resolution=image_resolution)
        spatial_grids = {"longitudes": longitude_grid_2D, "latitudes": latitude_grid_2D}
        return spatial_grids

    @classmethod
    def make_metadata(cls, header_data_unit: Dict[str, object]) -> Dict[str, str]:
        """Pull important information about the observation from the FITS header
        to populate the final data products (FITS and NetCDF)."""

        # Access FITS header information
        header = cls.get_header(header_data_unit=header_data_unit)

        # Date and time of observation (format: YYYY-MM-DDtHH:MM:SS:ssss)
        date_time = cls.get_header_contents(header=header, header_parameter='DATE-OBS')
        # Central meridian longitude (System III West Longitude)
        LCMIII = cls.get_header_contents(header=header, header_parameter='LCMIII')
        # Filter wavenumber
        wavenumber = cls.get_header_contents(header=header, header_parameter='lambda')
        # Observation viewing
        viewing = cls.get_viewing(header=header)

        metadata = {'date_time': date_time, 'LCMIII': LCMIII, 'wavenumber': wavenumber, 'viewing': viewing}
        return metadata
    
    @staticmethod
    def unpack_hdu_group(hdu_group: Dict[str, object]) -> List[object]:
        return hdu_group['image'], hdu_group['radiance'], hdu_group['emission_angle']

    @staticmethod
    def pack_data(radiance: npt.NDArray[np.float64], radiance_error: npt.NDArray[np.float64],
                emission_angle:npt.NDArray[np.float64], spatial_grids: npt.NDArray[np.float64],
                metadata: dict) -> Dict[str, npt.ArrayLike]:
        return {"radiance": radiance, "radiance_error": radiance_error, "emission_angle": emission_angle, "spatial_grids": spatial_grids, "metadata": metadata}
    
    @staticmethod
    def make_data_dict(filepaths: List[str]) -> List[dict]:
        """Create dictionary to store metadata and data products"""

        # Build dictionary template for each observation
        dict_template = {
            "filename": [],
            "radiance": [],
            "radiance_error": [],
            "emission_angle": [],
            "spatial_grids": [],
            "metadata":[]         
            }
        # Build a list of this dictionary to contain all files in filepaths
        data_dict = [dict_template for _ in filepaths]
        return data_dict

    @classmethod
    def make_json(cls, data_products: Dict[str, npt.ArrayLike]):
        """Generate a NetCDF file containing geometricall-registered radiance and emission angle maps
        with radiance error and other metadata."""
        pass

    @classmethod
    def make_netcdf(cls, data_products: Dict[str, npt.ArrayLike]):
        """Generate a NetCDF file containing geometricall-registered radiance and emission angle maps
        with radiance error and other metadata."""
        pass

    @classmethod
    def save_data_products(cls, data_products: Dict[str, npt.ArrayLike]) -> None:

        # Store data products in JSON format
        cls.make_json(data_products=data_products)

        # Store data products in NetCDF format
        cls.make_netcdf(data_products=data_products)

        pass


# Module-level functions
def find_filepaths(mode: str) -> List[str]:
        
    if mode == "single":
        filepaths = [
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_J7.9_2018-05-24T02:49:57.5419_Jupiter_clean_withchop.fits.gz"
            ]
    elif mode == "three":
        filepaths = [
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_J7.9_2018-05-24T02:49:57.5419_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_PAH1_2018-05-24T03:27:08.2479_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_Q3_2018-05-27T03:06:37.8961_Jupiter_clean_withchop.fits.gz",
            ]
    elif mode == "all":
        filepaths = [
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_J7.9_2016-02-15T08:47:39.7606_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_J7.9_2016-02-15T05:21:59.7428_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_J7.9_2016-02-15T05:25:36.2404_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_J7.9_2016-02-15T06:09:03.7779_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_J7.9_2016-02-15T06:12:40.2674_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_J7.9_2016-02-15T08:51:29.2734_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_J7.9_2016-02-16T05:49:55.5712_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_J7.9_2016-02-16T05:53:34.0963_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_J7.9_2016-02-16T08:05:29.5535_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_J7.9_2016-02-16T08:09:05.0788_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_J7.9_2016-02-16T08:53:29.5943_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_J7.9_2016-02-16T08:57:06.0889_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_NEII_1_2016-02-15T05:17:17.5740_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_NEII_1_2016-02-15T05:19:16.2549_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_NEII_1_2016-02-15T06:04:19.6242_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_NEII_1_2016-02-15T06:06:18.2720_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_NEII_1_2016-02-15T08:42:53.5568_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_NEII_1_2016-02-15T08:44:51.2919_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_NEII_1_2016-02-16T05:45:09.4158_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_NEII_1_2016-02-16T05:47:07.1067_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_NEII_1_2016-02-16T08:00:51.4820_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_NEII_1_2016-02-16T08:02:48.1118_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_NEII_1_2016-02-16T08:48:57.4374_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_NEII_1_2016-02-16T08:50:51.1209_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_NEII_2_2016-02-15T04:58:05.2964_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_NEII_2_2016-02-15T05:00:03.2720_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_NEII_2_2016-02-15T05:45:13.3344_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_NEII_2_2016-02-15T05:47:10.2689_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_NEII_2_2016-02-15T08:23:39.3850_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_NEII_2_2016-02-15T08:25:35.2972_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_NEII_2_2016-02-16T05:25:13.1801_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_NEII_2_2016-02-16T05:27:11.0699_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_NEII_2_2016-02-16T07:41:17.1905_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_NEII_2_2016-02-16T07:43:12.0791_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_NEII_2_2016-02-16T08:30:13.2218_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_NEII_2_2016-02-16T08:32:08.0715_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_PAH1_2016-02-15T04:48:49.7459_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_PAH1_2016-02-15T04:50:48.2300_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_PAH1_2016-02-15T05:35:53.1955_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_PAH1_2016-02-15T05:37:53.2969_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_PAH1_2016-02-15T08:14:17.2385_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_PAH1_2016-02-15T08:16:17.2684_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_PAH1_2016-02-16T05:15:37.3243_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_PAH1_2016-02-16T05:17:44.0822_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_PAH1_2016-02-16T07:31:39.6114_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_PAH1_2016-02-16T07:33:38.1577_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_PAH1_2016-02-16T08:21:11.1104_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_PAH1_2016-02-16T08:23:09.0714_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q1_2016-02-15T05:02:43.2867_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q1_2016-02-15T05:04:43.2782_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q1_2016-02-15T05:49:49.3163_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q1_2016-02-15T05:51:50.2548_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q1_2016-02-15T08:28:13.3064_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q1_2016-02-15T08:30:15.2725_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q1_2016-02-16T05:29:59.1336_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q1_2016-02-16T05:32:08.0700_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q1_2016-02-16T07:45:57.1691_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q1_2016-02-16T07:48:01.1119_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q1_2016-02-16T08:34:41.1556_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q1_2016-02-16T08:36:44.0646_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q2_2016-02-15T05:07:27.0295_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q2_2016-02-15T05:09:28.2387_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q2_2016-02-15T05:54:33.0492_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q2_2016-02-15T05:56:34.2312_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q2_2016-02-15T08:32:57.0731_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q2_2016-02-15T08:35:01.2418_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q2_2016-02-16T05:34:58.9051_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q2_2016-02-16T05:37:05.0917_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q2_2016-02-16T07:50:38.9118_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q2_2016-02-16T07:53:14.1398_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q2_2016-02-16T08:39:20.8892_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q2_2016-02-16T08:41:20.0595_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q3_2016-02-15T05:12:15.8414_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q3_2016-02-15T05:14:25.2561_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q3_2016-02-15T05:59:21.8466_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q3_2016-02-15T06:01:31.2718_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q3_2016-02-15T08:37:49.8844_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q3_2016-02-15T08:40:01.2619_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q3_2016-02-16T05:40:01.6675_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q3_2016-02-16T05:42:12.0928_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q3_2016-02-16T07:55:51.6819_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q3_2016-02-16T07:58:00.1225_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q3_2016-02-16T08:44:01.6597_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_Q3_2016-02-16T08:46:09.0626_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_SIV_2_2016-02-15T04:53:27.2566_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_SIV_2_2016-02-15T04:55:22.2554_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_SIV_2_2016-02-15T05:40:37.3549_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_SIV_2_2016-02-15T05:42:34.2363_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_SIV_2_2016-02-15T08:19:05.3466_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_SIV_2_2016-02-15T08:21:00.2466_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_SIV_2_2016-02-16T05:20:29.2882_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_SIV_2_2016-02-16T05:22:31.1138_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_SIV_2_2016-02-16T07:36:17.2161_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_SIV_2_2016-02-16T07:38:10.1222_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_SIV_2_2016-02-16T08:25:45.1583_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/data/visir/2016/2016feb15/wvisir_SIV_2_2016-02-16T08:27:38.0963_Jupiter_clean_withchop.fits.gz"
        ]
    return filepaths


# @jit(nopython=True)
def register_maps():

    # Point to data directory    
    filepaths = find_filepaths(mode="all")

    # Create dictionary to store metadata and data products
    data_products = Process.make_data_dict(filepaths=filepaths)

    # Read in VISIR FITS files and construct geographic data products
    for ifile, filepath in enumerate(filepaths):
        
        # Retrieve Header Data Units (HDUs) from FITS files and group into dictionary
        hdu_group = Preprocess.preprocess(filepath=filepath)

        # Construct data products from Header Data Units (HDUs)
        data_products = Process.process(hdu_group=hdu_group)

        # Output data products to file
        Process.save_data_products(data_products=data_products)

        print(f"Register map: {ifile+1} / {len(filepaths)}")