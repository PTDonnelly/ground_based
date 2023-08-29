
from astropy.io import fits
import glob
from os.path import exists
from pathlib import Path
from typing import List, Dict

from sorcery import dict_of

from giantpipe import Config

class Preprocess:
    """Method-focused class responsible for reading FITS files
    and constructing a dictionary containing the FITS Header Data Unit (HDU)."""

    def __init__(self, filepath) -> None:
        self.filepath: object = filepath
        return  

    @staticmethod
    def pathify(filepaths: List[str]) -> List[str]:
        """Use the object-oriented pathlib.Path() to avoid problems with pathing between 
        Windows (backslash) and Unix systems (forwardslash)."""
        return [Path(filepath) for filepath in filepaths]

    @classmethod
    def find_data(cls, data_directory: str, epoch: str, mode: str) -> List[str]:        
        """Find observations based on pathing and epoch description in Config()
        and universalise pathnames with Path()."""
        
        if mode == 'fits':
            # filepaths = glob.glob(f"{data_directory}{epoch}/recal_*.fits.gz")
            filepaths = [
                f"{data_directory}{epoch}/recal_wvisir_J7.9_2016-02-15T05 21 59.7428_Jupiter_clean_withchop.fits.gz",
                f"{data_directory}{epoch}/recal_wvisir_NEII_1_2016-02-15T08 42 53.5568_Jupiter_clean_withchop.fits.gz",
                f"{data_directory}{epoch}/recal_wvisir_PAH1_2016-02-15T04 50 48.2300_Jupiter_clean_withchop.fits.gz"
                ]
            
        if mode == 'nc':
           filepaths = glob.glob(f"{data_directory}{epoch}/netcdf/recal_*.nc")
        return cls.pathify(filepaths)
    
    @classmethod
    def get_data_files(cls, mode: str) -> List[object]: 
        """Find and gather all files for a given epoch of VISIR observations.
            Make sure file hierarchy (defined in config.py) is pointing to the right place."""

        # Specify location of data and observing epoch
        data_directory = Config.data_directory
        epoch = Config.epoch
        return cls.find_data(data_directory, epoch, mode)

    @staticmethod
    def fitsname_to_ncname(filepath: object) -> str:
        # Construct filename from filepath
        file_stem = filepath.stem
        filename = file_stem.split('_clean_withchop.fits')[0]
        return Path(f"{Config.data_directory}{Config.epoch}/netcdf/{filename}.nc")
    
    @staticmethod
    def find_missing_netcdf_files(check: List[bool], filepaths: List[object]) -> List[object]:
        return  [file for file, file_check in zip(list(filepaths), check) if file_check == False]

    @classmethod
    def check_netcdf_exists(cls, filepaths: List[object]) -> bool:
        """Check if the dataset exists in NetCDF format. If not, pass False so that it can be created by Dataset.create()"""

        # Check if the same files exist in NetCDF format
        check = []
        for filepath in filepaths:
            ncfile = cls.fitsname_to_ncname(filepath)
            if ncfile.exists():
                check.append(True)
            else:
                check.append(False)
        return check
    
    @classmethod
    def check_netcdf(cls)-> List[object]:
        """Checks for netCDF dataset, if there are any missing (i.e. there are .fits versions but not .nc), creates the missing files.
        Only returns the list of objects if necessary."""

        # Point to .fits files
        filepaths = Preprocess.get_data_files(mode='fits')

        # Compare to .nc files
        check = cls.check_netcdf_exists(filepaths)
        if all(check):
            # If all .nc files are there, continue
            return None
        else:
            # If any files are missing, make them
            return cls.find_missing_netcdf_files(check, filepaths)

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
    def get_filenames(cls, filepath: object) -> Dict[str, object]:
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
    def read_fits(filename: object, extension: int) -> dict:
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
