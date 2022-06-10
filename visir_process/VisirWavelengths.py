def VisirWavelengths(ifilt):
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