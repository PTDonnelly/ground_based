def SetWave(wavelength, wavenumber):
    """Convert between wavelength and wavenumber for visir filters"""
    
    if (wavelength == 7.90) or (wavenumber == 1253.0625):
        wavelength = 7.90
        wavenumber = 1253.0625
        ifilt_sc, ifilt_v = 0, 0

    # if (wavelength == 8.59) :
    #     wavenumber = 1164.14
    #    ifilt = 1
    if (wavelength == 8.59) or (wavenumber == 1163.0755):
        wavelength = 8.59
        wavenumber = 1163.0755
        ifilt_sc, ifilt_v = 1, 1

    if (wavelength == 8.99) or (wavenumber == 1112.15):
        wavelength = 8.99
        wavenumber = 1112.15
        ifilt_sc, ifilt_v = 2, 2

    if (wavelength == 9.82) or (wavenumber == 1012.52):
        wavelength = 9.82
        wavenumber = 1012.52
        ifilt_sc, ifilt_v = 3, 3
    # if (wavelength == 9.821):
    #     wavelength = 9.821
    #     wavenumber = 1012.52
    #     ifilt = 3

    if (wavelength == 10.49) or (wavenumber == 956.922):
        wavelength = 10.49
        wavenumber = 956.922
        ifilt_sc, ifilt_v = 4, 4

    if (wavelength == 10.77) or (wavenumber == 929.27953):
        wavelength = 10.77
        wavenumber = 929.27953
        ifilt_sc, ifilt_v = 5, 5
    # if (wavelength == 10.77):
    #     wavenumber = 928.505
    #    ifilt = 5

    if (wavelength == 11.25) or (wavenumber == 887.524):
        wavelength = 11.25
        wavenumber = 887.524
        ifilt_sc, ifilt_v = 6, 6

    # if (wavelength == 11.85):
    #     wavenumber = False
    #     ifilt = 7

    if (wavelength == 11.88) or (wavenumber == 852.115):
        wavelength = 11.88
        wavenumber = 852.115
        ifilt_sc, ifilt_v = 8, 7
        
    if (wavelength == 12.27) or (wavenumber == 815.70010):
        wavelength = 12.27
        wavenumber = 815.70010
        ifilt_sc, ifilt_v = 9, 8

    # if (wavelength == 12.81):
    #     wavenumber = False
    #     ifilt = 10
        
    # if (wavelength == 13.04):
    #     wavenumber = 766.871
    #    ifilt = 11
    if (wavelength == 13.04) or (wavenumber == 766.75227):
        wavelength = 13.04
        wavenumber = 766.75227
        ifilt_sc, ifilt_v = 11, 9

    # if (wavelength == 17.65):
    #     wavenumber = 566.572
    #    ifilt = 12
    if (wavelength == 17.65) or (wavenumber == 563.97095):
        wavelength = 17.65
        wavenumber = 563.97095
        ifilt_sc, ifilt_v = 12, 10

    # if (wavelength == 18.72):
    #     wavenumber = 534.188
    #    ifilt = 13
    if (wavelength == 18.72) or (wavenumber == 532.73522):
        wavelength = 18.72
        wavenumber = 532.73522
        ifilt_sc, ifilt_v = 13, 11

    # if (wavelength == 19.50):
    #     wavenumber = 512.821
    #    ifilt = 14
    if (wavelength == 19.50) or (wavenumber == 511.97345):
        wavelength = 19.50
        wavenumber = 511.97345
        ifilt_sc, ifilt_v = 14, 12
        
    return wavelength, wavenumber, ifilt_sc, ifilt_v