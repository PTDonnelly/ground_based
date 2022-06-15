def Wavenumbers(ifilt):
    """ Returns wavenumber when supplied filter index, 
        Only use when binning entire filter set, is not
        equipped to handle subsets and combinations"""

    if ifilt == 0:
        wavenumber= 1253.0625
    if ifilt == 1:
        wavenumber= 1163.0755
    if ifilt == 2:
        wavenumber= 1112.15
    if ifilt == 3:
        wavenumber= 1012.52
    if ifilt == 4:
        wavenumber= 956.922
    if ifilt == 5:
        wavenumber= 929.27953
    if ifilt == 6:
        wavenumber= 887.524
    if ifilt == 7:
        wavenumber= 852.115
    if ifilt == 8:
        wavenumber= 815.70010
    if ifilt == 9:
        wavenumber= 766.75227
    if ifilt == 10:
        wavenumber = 563.97095
    if ifilt == 11:
        wavenumber = 532.73522
    if ifilt == 12:
        wavenumber = 511.97345

    return wavenumber


def Wavelengths(ifilt):
    """ Returns wavelength when supplied filter index
        Only use when binning entire filter set, is not
        equipped to handle subsets and combinations"""

    if ifilt == 0:
        wavelength = 7.90
    if ifilt == 1:
        wavelength = 8.59
    if ifilt == 2:
        wavelength = 8.99
    if ifilt == 3:
        wavelength = 9.82
    if ifilt == 4:
        wavelength = 10.49
    if ifilt == 5:
        wavelength = 10.77
    if ifilt == 6:
        wavelength = 11.25
    if ifilt == 7:
        wavelength = 11.88
    if ifilt == 8:
        wavelength = 12.27
    if ifilt == 9:
        wavelength = 13.04
    if ifilt == 10:
        wavelength  = 17.65
    if ifilt == 11:
        wavelength  = 18.72
    if ifilt == 12:
        wavelength  = 19.50

    return wavelength