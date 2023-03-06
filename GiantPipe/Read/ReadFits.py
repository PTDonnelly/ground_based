from astropy.io import fits
from pathlib import Path

def ReadFits(filepath):
    """Open file, store to float array then close file"""

    # Extract filepath
    name = filepath.split('.gz')

    # Header information
    with fits.open(filepath) as hdu:
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
    # Doppler velocity angle map data
    vdopname = f"{name[0]}.vdop.gz"
    if not Path(vdopname).exists():
        vdophead, vdopdata = None, None
    else:
        with fits.open(vdopname) as hdu:
            vdophead = hdu[0].header
            vdopdata = hdu[0].data
    return imghead, imgdata, cylhead, cyldata, muhead, mudata, vdophead, vdopdata