"""
PTD 01/06/22: Code to calibrate VISIR cylindrical maps and
              prepare spectra for input into NEMESIS.
"""

from astropy.io import fits
from dataclasses import dataclass
import numpy as np
import numpy.typing as npt
from typing import Tuple

import icecream as ic

@dataclass
class Config:
    """Define flags to configure the execution of the pipeline"""
    # Input flags
    calibrate = False # Read raw data and calibrate
    datasource = 'fits' # Source of data: local cmaps ('fits') or local numpy arrays ('npy')
    
    # Data processing flags
    create_netcdf = False   # Create a NetCDF file from each cylindical map of radiance and emission angle, including observation metadata
    
    # Binning parameters
    grid_resolution = 0.5
    latitude_range = -90, 90
    longitude_range = 0, 360
    emission_angle_range = 0, 90



class ProcessData():
    """Method-focused class responsible for accessing header and
    data information from VISIR FITS files"""

    def __init__(self, filepath):
        self.filepath: str = filepath       

    def get_filename(self, filepath: str) -> dict:
        """Obtain .fits filenames for image, cylindrical map (cmap)
        and emission angle map (mumap)."""

        fname = filepath.split('.gz')
        cmap = f"{fname[0]}.cmap.gz"
        mumap = f"{fname[0]}.mu.gz"
        file = {"image": filepath, "cmap": cmap, "mumap": mumap}
        return file

    def read_fits(self, filename: dict, extension: int) -> dict:
        """Read header and data from FITS file
        filename: name of the FITS file
        extension: extension of the FITS file"""

        with fits.open(filename) as hdu:
            header = hdu[extension].header
            data = hdu[extension].data
            header_data_dict = {"header": header, "data": data}
        return header_data_dict

    def get_header_info(self, header: dict, header_parameter: str) -> str:
        """Extract header information from the FITS header"""
        
        header_info = header[header_parameter]
        return header_info

    def get_viewing(self, header: dict) -> int:
        """Determine the viewing of the image, needed for selecting data
        from the good hemisphere later"""
        
        # Pull out telescope pointing information from FITS header
        chop_parameter = 'HIERARCH ESO TEL CHOP POSANG'
        ada_parameter = 'HIERARCH ESO ADA POSANG'
        chop_angle: float = self.get_header_info(header=header, header_parameter=chop_parameter)
        ada_angle: float  = self.get_header_info(header=header, header_parameter=ada_parameter) + 360
        
        # Save flag depending on Northern (1) or Southern (-1) viewing
        view = 1 if (chop_angle == ada_angle) else -1
        return view

    def error_type_flat(self) -> float:
        """Use a flat 5% uncertainty on each radiance value. This is the maximum expected
        value for the VISIR images based on previous studies."""
        
        return 0.05

    def error_type_statistical(self, header_data_unit: dict) -> float:
        """Calculate an uncertainty for each individual image derived from the
        standard deviation of the background sky flux."""
        
        # Access FITS header and data
        header, data = header_data_unit['header'], header_data_unit['data']
        # Determine Northern (1) or Southern (-1) viewing
        viewing = self.get_viewing(header=header)
        
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

    def calculate_errors(self, header_data_unit: dict, type: str) -> float:
        """Calculate radiance error from the variability background sky flux"""

        # Run relevant error calculation mode
        if type == "statistical":
            # Calculate standard deviation of background sky (unique for each image)
            error = self.error_type_statistical(header_data_unit)
        elif type == "flat":
            # Set to a constant typical value
            error = self.error_type_flat()
        return error

    def make_grid_1D(self, grid_range: Tuple[float, float], grid_resolution: float) -> npt.NDArray[np.float64]:
        """Generate a one-dimensional spatial grid for geometric registration."""
        
        grid_1D = np.arange(grid_range[0], grid_range[1], grid_resolution)
        return grid_1D

    def make_grid_2D(self, grid_type: str, x_size: float, y_size: float, grid_resolution: float) -> npt.NDArray[np.float64]:
        """Generate a spatial grid for geometric registration."""
        
        if grid_type == 'longitude':
            number_of_repititions = y_size
            grid_range = Config().longitude_range
        elif grid_type == 'latitude':
            number_of_repititions = x_size
            grid_range = Config().latitude_range
        
        grid_1D = self.make_grid_1D(grid_range=grid_range, grid_resolution=grid_resolution)
        grid_2D = np.tile(grid_1D, (number_of_repititions, 1))
        if grid_type == 'latitude':
            grid_2D = np.transpose(grid_2D)
        return grid_2D

    def make_radiance_map(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Read in radiance values from cylindrical map and do unit conversion
        from native DRM units () into standard VISIR units ()"""

        # Coefficient needed to convert DRM units
        unit_conversion = 1e-7

        # Scale entire cylindrical map
        radiance = data * unit_conversion
        return radiance
    
    def make_radiance_error_map(self, data: npt.NDArray[np.float64], error: float) -> npt.NDArray[np.float64]:
        """Read in radiance values from cylindrical map and construct a new cylindrical map
        with values radiance * error for each pixel"""

        # Scale entire cylindrical map
        radiance_error = data * error
        return radiance_error

    def make_mu_map(self, data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Read in cosine(emission angle) values from cylindrical map and do trignometric conversion
        from cosine(mu) to mu (in degrees)."""
        
        # Get cosine(emission angle) map from FITS data
        cosine_mu = data

        # Calculate arc cosine of the map data
        mu_radians = np.arccos(cosine_mu)
        mu_degrees = np.degrees(mu_radians)
        return mu_degrees

    def get_image_resolution(self, x_size: int, y_size: int) -> float:
        """Check that image dimensions are equal and both conform to the same resolution"""

        if (360 / x_size) != 180 / y_size:
            quit()
        else:
            image_resolution = (360 / x_size)
        return image_resolution

    def make_spatial_grids(self, header: object) -> dict:
        "Read in raw cylindrical maps and perform geometric registration"

        # Calculate bounds of image map
        x_size = self.get_header_info(header=header, header_parameter='NAXIS1')
        y_size = self.get_header_info(header=header, header_parameter='NAXIS2')

        # Calculate pixel resolution of image map
        image_resolution = self.get_image_resolution(x_size=x_size, y_size=y_size)

        # Construct two-dimensional spatial grids
        longitude_grid_2D = self.make_grid_2D(grid_type='longitude', x_size=x_size, y_size=y_size, grid_resolution=image_resolution)
        latitude_grid_2D = self.make_grid_2D(grid_type='latitude', x_size=x_size, y_size=y_size, grid_resolution=image_resolution)
        spatial_grids = {"longitudes": longitude_grid_2D, "latitudes": latitude_grid_2D}
        return spatial_grids

    def make_metadata(self, header: dict) -> dict:
        """Pull important information about the observation from the FITS header
        to populate the final data products (FITS and NetCDF)."""

        # Date and time of observation (format: YYYY-MM-DDtHH:MM:SS:ssss)
        date_time = self.get_header_info(header=header, header_parameter='DATE-OBS')
        # Central meridian longitude (System III West Longitude)
        LCMIII = self.get_header_info(header=header, header_parameter='LCMIII')
        # Filter wavenumber
        wavenumber = self.get_header_info(header=header, header_parameter='lambda')
        # Observation viewing
        viewing = self.get_viewing(header=header)

        metadata = {'date_time': date_time, 'LCMIII': LCMIII, 'wavenumber': wavenumber, 'viewing': viewing}
        return metadata

    def make_netcdf(self, radiance: npt.NDArray[np.float64], mu: npt.NDArray[np.float64], error: int, metadata: dict):
        """Generate a NetCDF file containing geometricall-registered radiance and emission angle maps
        with radiance error and other metadata."""
        pass

def find_filepaths(mode: str) -> list:
        
    if mode == "single":
        filepaths = [
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_J7.9_2018-05-24T02:49:57.5419_Jupiter_clean_withchop.fits.gz"
            ]
    elif mode == "all":
        filepaths = [
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_ARIII_2018-05-25T03:45:53.1419_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_ARIII_2018-05-26T00:03:13.9569_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_ARIII_2018-05-26T05:44:15.9897_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_ARIII_2018-05-26T07:19:42.0774_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_ARIII_2018-05-26T23:29:55.9051_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_ARIII_2018-05-27T00:39:50.8993_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_ARIII_2018-05-27T02:39:55.9284_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_J7.9_2018-05-24T02:49:57.5419_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_J7.9_2018-05-24T05:51:00.6121_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_J7.9_2018-05-25T00:22:22.2779_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_J7.9_2018-05-25T04:08:22.2746_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_J7.9_2018-05-26T00:21:09.9792_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_J7.9_2018-05-26T06:02:47.9655_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_J7.9_2018-05-26T07:38:32.1062_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_J7.9_2018-05-26T23:49:49.9432_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_NEII_1_2018-05-24T06:10:42.6362_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_NEII_1_2018-05-25T00:11:20.2744_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_NEII_1_2018-05-25T03:55:29.2519_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_NEII_1_2018-05-26T00:12:26.0008_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_NEII_1_2018-05-26T05:55:47.1495_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_NEII_1_2018-05-26T07:29:24.0714_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_NEII_1_2018-05-26T23:37:12.7783_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_NEII_1_2018-05-27T00:47:16.7533_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_NEII_2_2018-05-24T02:45:35.7531_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_NEII_2_2018-05-24T03:39:18.6152_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_NEII_2_2018-05-24T05:42:53.6526_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_NEII_2_2018-05-24T07:15:35.6747_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_NEII_2_2018-05-25T00:13:56.7964_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_NEII_2_2018-05-25T04:02:03.2717_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_NEII_2_2018-05-26T00:16:46.6576_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_NEII_2_2018-05-26T06:00:13.1871_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_NEII_2_2018-05-26T23:43:36.9132_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_PAH1_2018-05-24T02:36:43.5199_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_PAH1_2018-05-24T03:27:08.2479_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_PAH1_2018-05-24T04:49:11.6224_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_PAH1_2018-05-24T05:34:02.1968_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_PAH1_2018-05-25T03:23:18.2624_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_PAH1_2018-05-25T23:44:28.6926_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_PAH1_2018-05-26T01:13:36.7987_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_PAH1_2018-05-26T23:09:18.7096_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_PAH2_2_2018-05-24T06:41:15.5878_Jupiter_cburst_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_PAH2_2018-05-24T06:33:21.6415_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_PAH2_2018-05-24T06:37:01.6178_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_Q1_2018-05-25T04:24:52.2863_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_Q1_2018-05-26T00:42:18.1412_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_Q1_2018-05-26T06:23:41.1581_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_Q1_2018-05-26T07:56:58.6145_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_Q1_2018-05-27T00:05:03.3551_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_Q1_2018-05-27T01:16:37.3572_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_Q1_2018-05-27T03:15:15.3243_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_Q2_2018-05-25T04:18:05.8435_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_Q2_2018-05-26T00:35:44.6206_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_Q2_2018-05-26T06:17:10.6564_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_Q2_2018-05-26T07:54:25.1460_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_Q2_2018-05-27T00:02:20.9139_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_Q2_2018-05-27T01:21:22.7250_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_Q2_2018-05-27T03:10:37.3636_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_Q3_2018-05-25T04:14:46.2779_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_Q3_2018-05-26T00:31:03.1137_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_Q3_2018-05-26T06:10:47.8855_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_Q3_2018-05-26T07:46:39.9647_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_Q3_2018-05-26T23:56:07.9176_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_Q3_2018-05-27T01:06:08.9103_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_Q3_2018-05-27T03:06:37.8961_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_SIV_1_2018-05-24T06:24:13.7864_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_SIV_1_2018-05-24T23:57:02.2345_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_SIV_1_2018-05-25T03:41:11.7568_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_SIV_1_2018-05-26T00:00:38.1179_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_SIV_1_2018-05-26T01:27:26.5086_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_SIV_1_2018-05-26T05:39:46.4224_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_SIV_1_2018-05-26T07:15:06.4006_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_SIV_1_2018-05-26T23:25:14.8980_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_SIV_1_2018-05-27T00:35:11.8896_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_SIV_1_2018-05-27T02:32:55.1965_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_SIV_2_2018-05-24T04:53:42.6259_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_SIV_2_2018-05-24T05:38:31.6363_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_SIV_2_2018-05-25T23:49:02.1191_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_SIV_2_2018-05-26T01:20:03.1395_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_SIV_2_2018-05-26T05:32:18.1273_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_SIV_2_2018-05-26T23:15:47.9167_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_SIV_2_2018-05-27T00:22:51.9127_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_SIV_2_2018-05-27T03:49:46.9099_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_SIV_2018-05-24T23:52:11.2499_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_SIV_2018-05-25T23:53:41.9626_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_SIV_2018-05-26T01:22:38.0760_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_SIV_2018-05-26T05:36:57.1244_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_SIV_2018-05-26T23:18:24.7716_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_SIV_2018-05-27T00:25:24.7290_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_SIV_2018-05-27T02:27:30.7403_Jupiter_clean_withchop.fits.gz",
            "/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/recal_wvisir_SIV_2018-05-27T03:54:48.9331_Jupiter_clean_withchop.fits.gz"
        ]
    return filepaths

def main():
    
    filepaths = find_filepaths(mode="single")

    for ifile, filepath in enumerate(filepaths):
        # Read in VISIR maps from .fits files
        data = ProcessData(filepath=filepath)

        # Retrieve filename for each input file
        filename = data.get_filename(filepath=filepath)
        
        # Read the contents of each input file
        image = data.read_fits(filename=filename['image'], extension=0)

        # Read in raw cylindrical maps of radiance and emission angle
        raw_radiance = data.read_fits(filename=filename['cmap'], extension=0)
        raw_mu = data.read_fits(filename=filename['mumap'], extension=0)
        
        # Calculate instrumental measurement errors of radiance
        error = data.calculate_errors(header_data_unit=image, type='statistical')

        # Construct data maps from cylindrical maps
        radiance = data.make_radiance_map(data=raw_radiance['data'])
        radiance_error = data.make_radiance_error_map(data=radiance, error=error)
        mu = data.make_mu_map(data=raw_mu['data'])

        # Generate two-dimensional grids of latitude and longitude
        spatial_grids = data.make_spatial_grids(header=raw_radiance['header'])

        ### Generate useful metadata about observation
        metadata = data.make_metadata(header=raw_radiance['header'])

        print(f"Register map: {ifile} / {len(filepaths)}")

        # # Create NetCDF file from registered radiance and emission angle maps
        # data.make_netcdf(radiance=radiance, mu=mu, error=error, metadata=metadata)

if __name__ == '__main__':
    
    main() 