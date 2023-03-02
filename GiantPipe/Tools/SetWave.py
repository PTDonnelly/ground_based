def SetWave(filename, wavelength, wavenumber, ifilt):


    """Convert between filter name, wavelength, wavenumber and filter index
        for visir filters. Takes whatever value you have and outputs all 
        infomation about the filter as well as indices (with respect to the
        full filterset) for calibration."""
    
    if filename == None:
        filter_name = None 
    else:
        # Consider generalising this too
        filter_name = filename.split('visir_')
        filter_name = filter_name[-1].split('_20')
        filter_name = filter_name[0]

    if (filter_name == 'J7.9') or (wavelength == 7.90) or (wavenumber == 1253.0625) or (ifilt == 0):
        filter_name = 'J7.9'
        wavelength = 7.90
        wavenumber = 1253.0625
        ifilt_sc, ifilt_v = 0, 0

    # if (wavelength == 8.59) :
    #     wavenumber = 1164.14
    #    ifilt = 1
    if (filter_name == 'PAH1') or (wavelength == 8.59) or (wavenumber == 1163.0755) or (ifilt == 1):
        filter_name = 'PAH1'
        wavelength = 8.59
        wavenumber = 1163.0755
        ifilt_sc, ifilt_v = 1, 1

    if (filter_name == 'ARIII') or (wavelength == 8.99) or (wavenumber == 1112.15) or (ifilt == 2):
        filter_name = 'ARIII'
        wavelength = 8.99
        wavenumber = 1112.15
        ifilt_sc, ifilt_v = 2, 2

    if (filter_name == 'SIV_1') or (wavelength == 9.82) or (wavenumber == 1012.52) or (ifilt == 3):
        filter_name = 'SIV_1'
        wavelength = 9.82
        wavenumber = 1012.52
        ifilt_sc, ifilt_v = 3, 3
    # if (wavelength == 9.821):
    #     wavelength = 9.821
    #     wavenumber = 1012.52
    #     ifilt = 3

    if (filter_name == 'SIV') or (wavelength == 10.49) or (wavenumber == 956.922) or (ifilt == 4):
        filter_name = 'SIV'
        wavelength = 10.49
        wavenumber = 956.922
        ifilt_sc, ifilt_v = 4, 4

    if (filter_name == 'SIV_2') or (wavelength == 10.77) or (wavenumber == 929.27953) or (ifilt == 5):
        filter_name = 'SIV_2'
        wavelength = 10.77
        wavenumber = 929.27953
        ifilt_sc, ifilt_v = 5, 5
    # if (wavelength == 10.77):
    #     wavenumber = 928.505
    #    ifilt = 5

    if (filter_name == 'PAH2') or (wavelength == 11.25) or (wavenumber == 887.524) or (ifilt == 6):
        filter_name = 'PAH2'
        wavelength = 11.25
        wavenumber = 887.524
        ifilt_sc, ifilt_v = 6, 6

    # if (wavelength == 11.85):
    #     wavenumber = False
    #     ifilt = 7

    if (filter_name == 'PAH2_2') or (wavelength == 11.88) or (wavenumber == 852.115) or (ifilt == 7):
        filter_name = 'PAH2_2'
        wavelength = 11.88
        wavenumber = 852.115
        ifilt_sc, ifilt_v = 8, 7
        
    if (filter_name == 'NEII_1') or (wavelength == 12.27) or (wavenumber == 815.70010) or (ifilt == 8):
        filter_name = 'NEII_1'
        wavelength = 12.27
        wavenumber = 815.70010
        ifilt_sc, ifilt_v = 9, 8

    # if (wavelength == 12.81):
    #     wavenumber = False
    #     ifilt = 10
        
    # if (wavelength == 13.04):
    #     wavenumber = 766.871
    #    ifilt = 11
    if (filter_name == 'NEII_2') or (wavelength == 13.04) or (wavenumber == 766.75227) or (ifilt == 9):
        filter_name = 'NEII_2'
        wavelength = 13.04
        wavenumber = 766.75227
        ifilt_sc, ifilt_v = 11, 9

    # if (wavelength == 17.65):
    #     wavenumber = 566.572
    #    ifilt = 12
    if (filter_name == 'Q1') or (wavelength == 17.65) or (wavenumber == 563.97095) or (ifilt == 10):
        filter_name = 'Q1'
        wavelength = 17.65
        wavenumber = 563.97095
        ifilt_sc, ifilt_v = 12, 10

    # if (wavelength == 18.72):
    #     wavenumber = 534.188
    #    ifilt = 13
    if (filter_name == 'Q2') or (wavelength == 18.72) or (wavenumber == 532.73522) or (ifilt == 11):
        filter_name = 'Q2'
        wavelength = 18.72
        wavenumber = 532.73522
        ifilt_sc, ifilt_v = 13, 11

    # if (wavelength == 19.50):
    #     wavenumber = 512.821
    #    ifilt = 14
    if (filter_name == 'Q3') or (wavelength == 19.50) or (wavenumber == 511.97345) or (ifilt == 12):
        filter_name = 'Q3'
        wavelength = 19.50
        wavenumber = 511.97345
        ifilt_sc, ifilt_v = 14, 12

    return filter_name, wavelength, wavenumber, ifilt_sc, ifilt_v

def SetWaveReduced(filename, wavelength, wavenumber, ifilt):
    """Convert between filter name, wavelength, wavenumber and filter index
        for visir filters. Takes whatever value you have and outputs all 
        infomation about the filter as well as indices (with respect to the
        full filterset) for calibration."""
    
    if filename == None:
        filter_name = None 
    else:
        # Consider generalising this too
        filter_name = filename.split('visir_')
        filter_name = filter_name[-1].split('_20')
        filter_name = filter_name[0]

    if (filter_name == 'J7.9') or (wavelength == 7.90) or (wavenumber == 1253.0625) or (ifilt == 0):
        filter_name = 'J7.9'
        wavelength = 7.90
        wavenumber = 1253.0625
        ifilt_sc, ifilt_v = 0, 0

    if (filter_name == 'PAH1') or (wavelength == 8.59) or (wavenumber == 1163.0755) or (ifilt == 1):
        filter_name = 'PAH1'
        wavelength = 8.59
        wavenumber = 1163.0755
        ifilt_sc, ifilt_v = 1, 1

    if (filter_name == 'SIV_2') or (wavelength == 10.77) or (wavenumber == 929.27953) or (ifilt == 2):
        filter_name = 'SIV_2'
        wavelength = 10.77
        wavenumber = 929.27953
        ifilt_sc, ifilt_v = 5, 2
        
    if (filter_name == 'NEII_1') or (wavelength == 12.27) or (wavenumber == 815.70010) or (ifilt == 3):
        filter_name = 'NEII_1'
        wavelength = 12.27
        wavenumber = 815.70010
        ifilt_sc, ifilt_v = 9, 3

    if (filter_name == 'NEII_2') or (wavelength == 13.04) or (wavenumber == 766.75227) or (ifilt == 4):
        filter_name = 'NEII_2'
        wavelength = 13.04
        wavenumber = 766.75227
        ifilt_sc, ifilt_v = 11, 4

    if (filter_name == 'Q1') or (wavelength == 17.65) or (wavenumber == 563.97095) or (ifilt == 5):
        filter_name = 'Q1'
        wavelength = 17.65
        wavenumber = 563.97095
        ifilt_sc, ifilt_v = 12, 5

    if (filter_name == 'Q2') or (wavelength == 18.72) or (wavenumber == 532.73522) or (ifilt == 6):
        filter_name = 'Q2'
        wavelength = 18.72
        wavenumber = 532.73522
        ifilt_sc, ifilt_v = 13, 6

    if (filter_name == 'Q3') or (wavelength == 19.50) or (wavenumber == 511.97345) or (ifilt == 7):
        filter_name = 'Q3'
        wavelength = 19.50
        wavenumber = 511.97345
        ifilt_sc, ifilt_v = 14, 7

    return filter_name, wavelength, wavenumber, ifilt_sc, ifilt_v