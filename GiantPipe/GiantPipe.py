"""
PTD 01/06/22: Code to calibrate VISIR cylindrical maps and
              prepare spectra for input into NEMESIS.
"""
from astropy.io import fits
import numpy as np
import numpy.typing as npt
from dataclasses import dataclass

@dataclass
class Config:
    """Define flags to configure the execution of the pipeline"""
    # Input flags
    calibrate = False # Read raw data and calibrate
    datasource = 'fits' # Source of data: local cmaps ('fits') or local numpy arrays ('npy')
    
    # Binning
    bin_cmerid  = False # Use central meridian binning scheme
    bin_cpara   = False # Use central parallel binning scheme
    bin_ctl     = False # Use centre-to-limb binning scheme
    bin_region  = False # Use regional binning scheme (for a zoom 2D retrieval)
    ###bin_av_region = False # Use averaged regional binning scheme (for a single profile retrieval)
    # Output flags
    save        = False     # Store calculated profiles to local files
    plotting    = False      # Plot calculated profiles
    mapping     = False      # Plot maps of observations or retrieval
    spx         = False      # Write spxfiles as spectral input for NEMESIS
    retrieval   = False      # Plot NEMESIS outputs 

class ProcessData():
    """Method-focused class responsible for contianing VISIR data"""

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

    def register_map(self, header_data_unit: dict) -> npt.NDArray[np.float64]:
        "Read in raw cylindrical maps and perform geometric registration"

        header = header_data_unit['header']
        data = header_data_unit['data']
        
        x_size = self.get_header_info(header=header, header_parameter='NAXIS1')
        y_size = self.get_header_info(header=header, header_parameter='NAXIS2')
       
        # # Loop over each pixel to assign to the structure.
        # xstart  = float(naxis1) - Globals.lonrange[0]/(360/naxis1)
        # xstop   = float(naxis1) - Globals.lonrange[1]/(360/naxis1) 
        # ystart  = (float(naxis2)/2) + Globals.latrange[0]/(180/naxis2)
        # ystop   = (float(naxis2)/2) + Globals.latrange[1]/(180/naxis2) 
        # x_range = np.arange(xstart, xstop, 1, dtype=int)
        # y_range = np.arange(ystart, ystop, 1, dtype=int)
        # for x in x_range:
        #     for y in y_range:
        #         # Only assign latitude and longitude if non-zero pixel value
        #         if (cyldata[y, x] > 0):
        #             # Calculate finite spatial element (lat-lon co-ordinates)
        #             lat = Globals.latrange[0] + ((180 / naxis2) * y)
        #             lon = Globals.lonrange[0] - ((360 / naxis1) * x)
        #             # Adjust co-ordinates from edge to centre of bins
        #             lat = lat + Globals.latstep/res
        #             lon = lon - Globals.latstep/res
        #             # Convert from planetographic to planetocentric latitudes
        #             mu_ang = mudata[y, x]
        #             mu  = 180/pi * acos(mu_ang)
        #             # Calculate pxel radiance and error
        #             rad = cyldata[y, x] * 1e-7

        registered_map = 1
        return registered_map

    def calculate_errors(self, data: npt.ArrayLike, viewing: int, type: str) -> float:
        """Calculate radiance error from the variability background sky flux"""

        def error_type_flat(self, data: npt.ArrayLike) -> float:
            """Use a flat 5% uncertainty on each radiance value. This is the maximum expected
            value for the VISIR images based on previous studies."""
            return 0.05
        
        def error_type_statistical(self, data: npt.ArrayLike, viewing: int) -> float:
            """Calculate an uncertainty for each individual image derived from the
            standard deviation of the background sky flux."""
            
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

        if type == "statistical":
            # Calculate standard deviation of background sky (unique for each image)
            error = error_type_statistical(self, data=data)
            return error
        
        if type == "flat":
            # Set to a constant typical value
            error = error_type_flat(self, data=data)
            return error

    def create_metadata(self, header: dict) -> dict:
        """Pull important information about the observation from the FITS header
        to populate the final data products (FITS and NetCDF)."""

        # Date and time of observation (format: YYYY-MM-DDtHH:MM:SS:ssss)
        date_time = self.get_header_info(header=header, header_parameter='DATE-OBS')
        # Central meridian longitude (System III West Longitude)
        LCMIII = self.get_header_info(header=header, header_parameter='LCMIII')
        # Filter wavenumber
        wavenumber = self.get_header_info(header=header, header_parameter='lambda')

        metadata = {'date_time': date_time, 'LCMIII': LCMIII, 'wavenumber': wavenumber}
        return metadata

def main():

    # Read in VISIR maps from .fits files
    data = ProcessData()
    
    # Retrieve filename for each input file
    filepath = "C:/Users/padra/Documents/Research/projects/shared/dbardet/data/recal_wvisir_J7.9_2018-05-24T02_49_57.5419_Jupiter_clean_withchop.fits.gz"
    filename = data.get_filename(filepath=filepath)
    
    # Read the contents of each input file
    image = data.read_fits(filename=filename['image'], extension=0)
    
    ### Determine Northern (1) or Southern (-1) viewing
    viewing = data.get_viewing(header=image['header'])

    # Read in raw cylindrical maps of radiance and emission angle
    raw_radiance = data.read_fits(filename=filename['cmap'], extension=0)
    raw_mu = data.read_fits(filename=filename['mumap'], extension=0)
    
    # Perform geometric registration on raw maps
    radiance_map = data.register_map(header_data_unit=raw_radiance)
    mu_map = data.register_map(header_data_unit=raw_mu)
   
    # Calculate instrumental measurement errors of radiance
    rad_error = data.calculate_errors(image['data'], viewing=viewing, type='flat')

    ### Generate useful metadata about observation
    metadata = data.create_metadata(header=raw_radiance['header'])

    # Create NetCDF file from registered radiance and emission angle maps
    a, b, c = 1, 2, 3

if __name__ == '__main__':
    main() 