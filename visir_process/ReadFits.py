from astropy.io import fits

def ReadFits(filename):
    """Open file, store to float array then close file"""

    # Extract filename
    name = filename.split('.gz')

    # Header information
    with fits.open(filename) as hdu:
        imghead = hdu[0].header
        imgdata = hdu[0].data
    # Cylindrical map data
    cname = f"{name[0]}.cmap.gz"
    with fits.open(cname) as hdu:
        cylhead = hdu[0].header
        cyldata = hdu[0].data
    # Emission angle map data
    muname = f"{name[0]}.mu.gz"
    with fits.open(muname) as hdu:
        muhead = hdu[0].header
        mudata = hdu[0].data
    
    return imghead, imgdata, cylhead, cyldata, muhead, mudata