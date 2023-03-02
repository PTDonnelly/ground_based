from astropy.io import fits

def WriteFits(filepath, imghead, imgdata, cylhead, cyldata, muhead, mudata):
    """Create fits file, store new array and their existing headers, then close file"""

    # Extract filepath
    fpath = filepath.split('.gz')
    # Cut off file extention
    fname = fpath[0].split('/')
    # Reconstruct filepath without filename
    newpath = [f"{f}/" for f in fname[:-1]]

    # Image file and Header
    imgname = f"recal_{fname[-1]}.gz"
    imgpath = ''.join(newpath) + imgname
    hdu = fits.PrimaryHDU(data=imgdata, header=imghead)
    hdul = fits.HDUList([hdu])
    hdul.writeto(imgpath, overwrite=True, output_verify='silentfix')

    # Cylindrical map and header
    cylname = f"recal_{fname[-1]}.cmap.gz"
    cylpath = ''.join(newpath) + cylname
    hdu = fits.PrimaryHDU(data=cyldata, header=cylhead)
    hdul = fits.HDUList([hdu])
    hdul.writeto(cylpath, overwrite=True, output_verify='silentfix')

    # Emission angle map and header (not explicitly necessary but convenient
    # to keep all data files together)
    muname = f"recal_{fname[-1]}.mu.gz"
    mupath = ''.join(newpath) + muname
    hdu = fits.PrimaryHDU(data=mudata, header=muhead)
    hdul = fits.HDUList([hdu])
    hdul.writeto(mupath, overwrite=True, output_verify='silentfix')