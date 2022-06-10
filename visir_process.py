"""
PTD 01/06/22: Code to calibrate VISIR cylindrical maps and
              prepare spectra for input into NEMESIS.
"""

def main():
    import os
    import time
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from math import acos, cos, radians, pi
    from scipy.interpolate import interp1d
    
    start = time.time()
    
    ##### Define subroutines #####
    def visir_wavelengths(ifilt):
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
    
    def visir_wavenumbers(ifilt):
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
    
    def set_wave(wavelength, wavenumber):
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
    
    def find_files(mode):
        """Put all input data in a single list"""

        if mode == 'images':
            # files = ['cal_wvisir_NEII_2_2018-05-24T02_45_35.7531_Jupiter_clean_withchop.fits.gz']
            data_dir = 'data/'
            files = [f"{data_dir}cal_wvisir_ARIII_2018-05-25T03:45:53.1419_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_ARIII_2018-05-26T00:03:13.9569_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_ARIII_2018-05-26T05:44:15.9897_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_ARIII_2018-05-26T07:19:42.0774_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_ARIII_2018-05-26T23:29:55.9051_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_ARIII_2018-05-27T00:39:50.8993_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_ARIII_2018-05-27T02:39:55.9284_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_J7.9_2018-05-24T02:49:57.5419_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_J7.9_2018-05-24T05:51:00.6121_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_J7.9_2018-05-25T00:22:22.2779_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_J7.9_2018-05-25T04:08:22.2746_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_J7.9_2018-05-26T00:21:09.9792_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_J7.9_2018-05-26T06:02:47.9655_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_J7.9_2018-05-26T07:38:32.1062_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_J7.9_2018-05-26T23:49:49.9432_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_NEII_1_2018-05-24T06:10:42.6362_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_NEII_1_2018-05-25T00:11:20.2744_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_NEII_1_2018-05-25T03:55:29.2519_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_NEII_1_2018-05-26T00:12:26.0008_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_NEII_1_2018-05-26T05:55:47.1495_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_NEII_1_2018-05-26T07:29:24.0714_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_NEII_1_2018-05-26T23:37:12.7783_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_NEII_1_2018-05-27T00:47:16.7533_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_NEII_2_2018-05-24T02:45:35.7531_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_NEII_2_2018-05-24T03:39:18.6152_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_NEII_2_2018-05-24T05:42:53.6526_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_NEII_2_2018-05-24T07:15:35.6747_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_NEII_2_2018-05-25T00:13:56.7964_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_NEII_2_2018-05-25T04:02:03.2717_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_NEII_2_2018-05-26T00:16:46.6576_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_NEII_2_2018-05-26T06:00:13.1871_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_NEII_2_2018-05-26T23:43:36.9132_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_PAH1_2018-05-24T02:36:43.5199_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_PAH1_2018-05-24T03:27:08.2479_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_PAH1_2018-05-24T04:49:11.6224_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_PAH1_2018-05-24T05:34:02.1968_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_PAH1_2018-05-25T03:23:18.2624_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_PAH1_2018-05-25T23:44:28.6926_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_PAH1_2018-05-26T01:13:36.7987_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_PAH1_2018-05-26T23:09:18.7096_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_PAH2_2_2018-05-24T06:41:15.5878_Jupiter_cburst_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_PAH2_2018-05-24T06:33:21.6415_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_PAH2_2018-05-24T06:37:01.6178_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_Q1_2018-05-25T04:24:52.2863_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_Q1_2018-05-26T00:42:18.1412_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_Q1_2018-05-26T06:23:41.1581_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_Q1_2018-05-26T07:56:58.6145_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_Q1_2018-05-27T00:05:03.3551_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_Q1_2018-05-27T01:16:37.3572_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_Q1_2018-05-27T03:15:15.3243_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_Q2_2018-05-25T04:18:05.8435_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_Q2_2018-05-26T00:35:44.6206_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_Q2_2018-05-26T06:17:10.6564_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_Q2_2018-05-26T07:54:25.1460_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_Q2_2018-05-27T00:02:20.9139_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_Q2_2018-05-27T01:21:22.7250_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_Q2_2018-05-27T03:10:37.3636_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_Q3_2018-05-25T04:14:46.2779_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_Q3_2018-05-26T00:31:03.1137_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_Q3_2018-05-26T06:10:47.8855_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_Q3_2018-05-26T07:46:39.9647_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_Q3_2018-05-26T23:56:07.9176_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_Q3_2018-05-27T01:06:08.9103_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_Q3_2018-05-27T03:06:37.8961_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_SIV_1_2018-05-24T06:24:13.7864_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_SIV_1_2018-05-24T23:57:02.2345_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_SIV_1_2018-05-25T03:41:11.7568_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_SIV_1_2018-05-26T00:00:38.1179_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_SIV_1_2018-05-26T01:27:26.5086_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_SIV_1_2018-05-26T05:39:46.4224_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_SIV_1_2018-05-26T07:15:06.4006_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_SIV_1_2018-05-26T23:25:14.8980_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_SIV_1_2018-05-27T00:35:11.8896_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_SIV_1_2018-05-27T02:32:55.1965_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_SIV_2_2018-05-24T04:53:42.6259_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_SIV_2_2018-05-24T05:38:31.6363_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_SIV_2_2018-05-25T23:49:02.1191_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_SIV_2_2018-05-26T01:20:03.1395_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_SIV_2_2018-05-26T05:32:18.1273_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_SIV_2_2018-05-26T23:15:47.9167_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_SIV_2_2018-05-27T00:22:51.9127_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_SIV_2_2018-05-27T03:49:46.9099_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_SIV_2018-05-24T23:52:11.2499_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_SIV_2018-05-25T23:53:41.9626_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_SIV_2018-05-26T01:22:38.0760_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_SIV_2018-05-26T05:36:57.1244_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_SIV_2018-05-26T23:18:24.7716_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_SIV_2018-05-27T00:25:24.7290_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_SIV_2018-05-27T02:27:30.7403_Jupiter_clean_withchop.fits.gz",
                    f"{data_dir}cal_wvisir_SIV_2018-05-27T03:54:48.9331_Jupiter_clean_withchop.fits.gz"]

            # files = [f"{data_dir}cal_wvisir_ARIII_2018-05-25T03:45:53.1419_Jupiter_clean_withchop.fits.gz",
            #         f"{data_dir}cal_wvisir_ARIII_2018-05-26T00:03:13.9569_Jupiter_clean_withchop.fits.gz",
            #         f"{data_dir}cal_wvisir_J7.9_2018-05-24T02:49:57.5419_Jupiter_clean_withchop.fits.gz",
            #         f"{data_dir}cal_wvisir_J7.9_2018-05-24T05:51:00.6121_Jupiter_clean_withchop.fits.gz",
            #         f"{data_dir}cal_wvisir_NEII_1_2018-05-24T06:10:42.6362_Jupiter_clean_withchop.fits.gz",
            #         f"{data_dir}cal_wvisir_NEII_1_2018-05-25T00:11:20.2744_Jupiter_clean_withchop.fits.gz",
            #         f"{data_dir}cal_wvisir_NEII_2_2018-05-24T02:45:35.7531_Jupiter_clean_withchop.fits.gz",
            #         f"{data_dir}cal_wvisir_NEII_2_2018-05-24T03:39:18.6152_Jupiter_clean_withchop.fits.gz",
            #         f"{data_dir}cal_wvisir_PAH1_2018-05-24T02:36:43.5199_Jupiter_clean_withchop.fits.gz",
            #         f"{data_dir}cal_wvisir_PAH1_2018-05-24T03:27:08.2479_Jupiter_clean_withchop.fits.gz",
            #         f"{data_dir}cal_wvisir_PAH2_2_2018-05-24T06:41:15.5878_Jupiter_cburst_withchop.fits.gz",
            #         f"{data_dir}cal_wvisir_PAH2_2018-05-24T06:33:21.6415_Jupiter_clean_withchop.fits.gz",
            #         f"{data_dir}cal_wvisir_PAH2_2018-05-24T06:37:01.6178_Jupiter_clean_withchop.fits.gz",
            #         f"{data_dir}cal_wvisir_Q1_2018-05-25T04:24:52.2863_Jupiter_clean_withchop.fits.gz",
            #         f"{data_dir}cal_wvisir_Q1_2018-05-26T00:42:18.1412_Jupiter_clean_withchop.fits.gz",
            #         f"{data_dir}cal_wvisir_Q2_2018-05-25T04:18:05.8435_Jupiter_clean_withchop.fits.gz",
            #         f"{data_dir}cal_wvisir_Q2_2018-05-26T00:35:44.6206_Jupiter_clean_withchop.fits.gz",
            #         f"{data_dir}cal_wvisir_Q3_2018-05-25T04:14:46.2779_Jupiter_clean_withchop.fits.gz",
            #         f"{data_dir}cal_wvisir_Q3_2018-05-26T00:31:03.1137_Jupiter_clean_withchop.fits.gz",
            #         f"{data_dir}cal_wvisir_SIV_1_2018-05-24T06:24:13.7864_Jupiter_clean_withchop.fits.gz",
            #         f"{data_dir}cal_wvisir_SIV_1_2018-05-24T23:57:02.2345_Jupiter_clean_withchop.fits.gz",
            #         f"{data_dir}cal_wvisir_SIV_2_2018-05-24T04:53:42.6259_Jupiter_clean_withchop.fits.gz",
            #         f"{data_dir}cal_wvisir_SIV_2_2018-05-24T05:38:31.6363_Jupiter_clean_withchop.fits.gz",
            #         f"{data_dir}cal_wvisir_SIV_2018-05-24T23:52:11.2499_Jupiter_clean_withchop.fits.gz",
            #         f"{data_dir}cal_wvisir_SIV_2018-05-25T23:53:41.9626_Jupiter_clean_withchop.fits.gz"]

        if mode == 'singles':
            profile_dir = 'individual_merid_profiles/'
            files = [f"{profile_dir}'cal_wvisir_NEII_2_2018-05-24T02_45_35.7531_Jupiter_clean_withchop_merid_profile.npy"]
        
        if mode == 'spectrals':
            profile_dir = 'spectral_merid_profiles/'
            files = [f"{profile_dir}766.75227_merid_profile.npy"]
        
        return files

    def read_fits(filename):
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

    def read_cal(filename):
        """Read calibration file and return array of profiles"""
    
        # Open file
        with open(filename) as f:
            # Read contents
            lines = f.readlines()
            # Save number of filters, Voyager samples, and Cassini samples
            nfilt = int(lines[0])-1     # ignore 0 element (latitude), keep 1-15 (filters)
            niris = int(lines[1])
            ncirs = int(lines[2])-3     # cut off the 60N-60S Q band lines at the end
            # Define temporary variables (needed because of irregular formatting of calfile)
            ilat, lat_count = 0, 0
            newline, caldata = [], []
            for iline, line in enumerate(lines[3:]):
                if lat_count < 3:
                    # Gather spectra at each latitude
                    l = line.split()
                    [newline.append(il) for il in l]
                    lat_count += 1
                if lat_count == 3:
                    # Store spectral information
                    caldata.append(newline)
                    # Reset temporary variables
                    lat_count, newline = 0, []
                    ilat += 1
        # Separate spacecraft profiles (converted to units of W ..)
        irisdata = np.asarray(caldata[0:niris-1], dtype='float')
        cirsdata = np.asarray(caldata[niris:niris+ncirs-1], dtype='float')
        # Interpolate spacecraft profiles onto "lat_grid" (common with data)
        iris = np.zeros((n_lat_bins, nfilt, 2))
        cirs = np.zeros((n_lat_bins, nfilt, 2))
        for ifilt in range(nfilt):
            # Interp IRIS to VISIR
            f = interp1d(irisdata[:, 0], irisdata[:, ifilt+1])
            lmin, lmax = np.min(irisdata[:, 0]), np.max(irisdata[:, 0])
            keep = (lat_grid > lmin) & (lat_grid < lmax)
            # Store interpolated profile
            iris[keep, ifilt, 0] = lat_grid[keep]
            iris[keep, ifilt, 1] = f(lat_grid[keep]) * 10E-8
            # Interp CIRS to VISIR
            f = interp1d(cirsdata[:, 0], cirsdata[:, ifilt+1])
            lmin, lmax = np.min(cirsdata[:, 0]), np.max(cirsdata[:, 0])
            keep = (lat_grid > lmin) & (lat_grid < lmax)
            # Store interpolated profile
            cirs[keep, ifilt, 0] = lat_grid[keep]
            cirs[keep, ifilt, 1] = f(lat_grid[keep]) * 10E-8

        # Throw away zeros
        iris[iris == 0] = np.nan
        cirs[cirs == 0] = np.nan

        return iris, cirs

    def calculate_errors(image):
        """Carry out some experiments on the measurement error on the radiance values"""

        # Currently set to a typical reasonable value
        nominal_error = 0.05
        
        return nominal_error
    
    def register_maps():
        """ Step 1: Read img, cmap and mufiles
            Step 2: Geometric registration of pixel information
            Step 3: Gather pixel information for all files"""

        # Define local inputs
        nx, ny = 720, 360                  # Dimensions of an individual cylindrical map (needed for dictionary definition)
        res    = ny / 180                  # Resolution of maps: res = 1 (1 degree), res = 2 (0.5 degree) etc.

        # Create np.array for all pixels in all cmaps and mumaps
        spectrum = np.empty((ny, nx, nfiles, 7))

        # Define arrays
        viewing_mode   = np.empty(nfiles)
        wavelength     = np.empty(nfiles)
        wavenumber     = np.empty(nfiles)
        LCMIII         = np.empty(nfiles)

        # Define flags
        pg2pc = 0                   # Optional conversion of latitudes from planetographic to planetocentric

        # Loop over files
        for ifile, fname in enumerate(files):
            print(ifile, fname)
            ## Step 1: Read img, cmap and mufiles
            imghead, imgdata, cylhead, cyldata, muhead, mudata = read_fits(filename=f"{fname}")

            ## Step 2: Geometric registration of pixel information
            # Save flag depending on Northern (1) or Southern (-1) viewing
            chopang = imghead['HIERARCH ESO TEL CHOP POSANG']
            posang  = imghead['HIERARCH ESO ADA POSANG'] + 360
            view = 1 if chopang == posang else -1
            viewing_mode[ifile] = view
            
            # Store central meridian longitude
            LCMIII[ifile] = cylhead['LCMIII']

            # Assign spatial information to pixels
            naxis1    = cylhead['NAXIS1']
            naxis2    = cylhead['NAXIS2']
            naxis1_mu = muhead['NAXIS1']
            naxis2_mu = muhead['NAXIS2']

            # Set the central wavelengths for each filter. Must be
            # identical to the central wavelength specified for the
            # production of the k-tables
            wavelen, wavenum, _, _  = set_wave(wavelength=cylhead['lambda'], wavenumber=False)
            wavelength[ifile] = wavelen
            wavenumber[ifile] = wavenum
            
            # Loop over each pixel to assign to the structure.
            xstart  = float(naxis1) - lon_range[0]/(360/naxis1)
            xstop   = float(naxis1) - lon_range[1]/(360/naxis1) 
            ystart  = (float(naxis2)/2) + lat_range[0]/(180/naxis2)
            ystop   = (float(naxis2)/2) + lat_range[1]/(180/naxis2) 
            x_range = np.arange(xstart, xstop, 1)
            y_range = np.arange(ystart, ystop, 1)
            for ix, x in enumerate(x_range):
                for iy, y in enumerate(y_range): 
                    # Only assign latitude and longitude if non-zero pixel value
                    if (cyldata[iy, ix] > 0):
                        # Calculate finite spatial element (lat-lon co-ordinates)
                        lat = lat_range[0] + ((180 / naxis2) * y)
                        lon = lon_range[0] - ((360 / naxis1) * x)
                        # Adjust co-ordinates from edge to centre of bins
                        lat = lat + lat_step/res
                        lon = lon - lat_step/res
                        # Convert from planetographic to planetocentric latitudes
                        mu_ang = mudata[iy, ix]
                        mu  = 180/pi * acos(mu_ang)
                        # Calculate pixel radiance and error
                        rad = cyldata[iy, ix] * 1e-7
                        rad_error = calculate_errors(imgdata)
                        
                        ## Step 3: Gather pixel information for all files
                        # Store spectral information in spectrum array
                        spectrum[iy, ix, ifile, 0] = lat
                        spectrum[iy, ix, ifile, 1] = LCMIII[ifile]
                        spectrum[iy, ix, ifile, 2] = mu
                        spectrum[iy, ix, ifile, 3] = rad
                        spectrum[iy, ix, ifile, 4] = rad_error * rad
                        spectrum[iy, ix, ifile, 5] = wavenum
                        spectrum[iy, ix, ifile, 6] = view
        # Throw away zeros
        spectrum[spectrum == 0] = np.nan

        return spectrum, wavelength, wavenumber, LCMIII

    def create_merid_profiles():
        """ Step 4: Create central meridian average for each cmap
            Step 5: Create central meridian average for each wavelength"""

        # Create np.array for all individual mean profiles (one per file)
        single_merids = np.zeros((n_lat_bins, nfiles, 7))
        # Create np.array for all spectral mean profiles (one per filter)
        spectral_merids = np.zeros((n_lat_bins, nfilters, 6))

        # Loop over latitudes and create individual mean profiles
        print('Binning singles:')
        for ilat, _ in enumerate(lat_grid):
            # Define centre and edges of latitude bin
            clat = lat_range[0] + (lat_step)*ilat + (lat_step/2)
            lat1 = lat_range[0] + (lat_step)*ilat
            lat2 = lat_range[0] + (lat_step)*(ilat+1)
            # Loop over the spectrum array of each input file
            for ifile in range(nfiles):
                clon = LCMIII[ifile]
                lon1 = LCMIII[ifile] + merid_width
                lon2 = LCMIII[ifile] - merid_width
                # Select lat-lon region around central meridian to calculate average
                lats = spectrum[:, :, ifile, 0]
                lons = spectrum[:, :, ifile, 1]
                keep = (lats >= lat1) & (lats < lat2) & (lons < lon1) & (lons > lon2)
                spx = spectrum[keep, ifile, :]
                # Throw away hemisphere with negative beam
                view = np.mean(spx[:, 6])
                if (view == 1) and (lat1 >=-15) or (view == -1) and (lat1 <= 15):
                    if np.any(spx):
                        # Pull out variables
                        LCM      = np.nanmean(spx[:, 1])
                        mu       = np.nanmin(spx[:, 2])
                        rad      = np.nanmean(spx[:, 3])
                        rad_err  = np.nanmean(spx[:, 4])
                        wavenum  = spx[:, 5][0]
                        view     = spx[:, 6][0]
                        # Store individual meridional profiles
                        single_merids[ilat, ifile, 0] = clat
                        single_merids[ilat, ifile, 1] = LCM
                        single_merids[ilat, ifile, 2] = mu
                        single_merids[ilat, ifile, 3] = rad
                        single_merids[ilat, ifile, 4] = rad_err
                        single_merids[ilat, ifile, 5] = wavenum
                        single_merids[ilat, ifile, 6] = view

        # Loop over filters and create mean spectral profiles
        print('Binning spectrals:')
        for ifilt in range(nfilters):
            # Loop over latitudes and create individual mean profiles
            for ilat, _ in enumerate(lat_grid):
                # Define centre and edges of latitude bin
                clat = lat_range[0] + (lat_step)*ilat + (lat_step/2)
                lat1 = lat_range[0] + (lat_step)*ilat
                lat2 = lat_range[0] + (lat_step)*(ilat+1)
                # Select a filter to calculate average
                wave = visir_wavenumbers(ifilt)
                filters = single_merids[ilat, :, 5]
                keep = (filters == wave)
                spx = single_merids[ilat, keep, :]
                if np.any(spx):
                    # Pull out variables
                    LCM      = np.nanmean(spx[:, 1])
                    mu       = np.nanmin(spx[:, 2])
                    rad      = np.nanmean(spx[:, 3])
                    rad_err  = np.nanmean(spx[:, 4])
                    wavenum  = spx[:, 5][0]
                    # Store spectral meridional profiles
                    spectral_merids[ilat, ifilt, 0] = clat
                    spectral_merids[ilat, ifilt, 1] = LCM
                    spectral_merids[ilat, ifilt, 2] = mu
                    spectral_merids[ilat, ifilt, 3] = rad
                    spectral_merids[ilat, ifilt, 4] = rad_err
                    spectral_merids[ilat, ifilt, 5] = wavenum
        # Throw away zeros
        single_merids[single_merids == 0] = np.nan
        spectral_merids[spectral_merids == 0] = np.nan

        # # Clear spectrum array from local variables
        # del locals()['spectrum']

        return single_merids, spectral_merids

    def calibrate_merid_profiles():
        """ Step 6: Calibrate spectral_merids to spacecraft data
                    (i.e. create calib_spectral_merids)
            Step 7: Calibrate single_merids to calib_spectral_merids
                    (i.e. create calib_single_merids)"""

        # Create arrays to store calibration coefficients
        calib_coeff_single   = np.ones((nfiles, 2))
        calib_coeff_spectral = np.ones((nfilters, 2))
        
        # Read in Voyager and Cassini data into arrays
        calfile = "visir.jup.filtered-iris-cirs.10-12-15.data.v3"
        iris, cirs = read_cal(calfile)

        # Calculate calibration coefficients for the spectral merid profiles
        print('Calibrating spectrals:')
        for iwave in range(nfilters):
            # Get filter index for calibration file
            waves = spectral_merids[:, iwave, 5]
            wave  = waves[(waves > 0)][0]
            _, _, ifilt_sc, ifilt_v = set_wave(wavelength=False, wavenumber=wave)
            # Calculate averages for calibration
            if ifilt_sc < 12:
                # Establish shared latitudes for accurate averaging
                lmin_visir, lmax_visir = np.nanmin(spectral_merids[:, ifilt_v, 0]), np.nanmax(spectral_merids[:, ifilt_v, 0])
                lmin_calib, lmax_calib = np.nanmin(cirs[:, ifilt_sc, 0]), np.nanmax(cirs[:, ifilt_sc, 0])
                latmin, latmax         = np.max((lmin_visir, lmin_calib, -70)), np.min((lmax_visir, lmax_calib, 70))
                visirkeep              = (spectral_merids[:, ifilt_v, 0] >= latmin) & (spectral_merids[:, ifilt_v, 0] <= latmax)            
                visirdata              = spectral_merids[visirkeep, ifilt_v, 3]
                visirmean              = np.nanmean(spectral_merids[visirkeep, ifilt_v, 3])
                # Use CIRS for N-Band
                calibkeep  = (cirs[:, ifilt_sc, 0] >= latmin) & (cirs[:, ifilt_sc, 0] <= latmax)
                calib      = cirs[:, ifilt_sc, 1]
                calibdata  = cirs[calibkeep, ifilt_sc, 1]
                calibmean  = np.nanmean(calibdata)
            else:
                # Establish shared latitudes for accurate averaging
                lmin_visir, lmax_visir = np.nanmin(spectral_merids[:, ifilt_v, 0]), np.nanmax(spectral_merids[:, ifilt_v, 0])
                lmin_calib, lmax_calib = np.nanmin(iris[:, ifilt_sc, 0]), np.nanmax(iris[:, ifilt_sc, 0])
                latmin, latmax         = np.max((lmin_visir, lmin_calib)), np.min((lmax_visir, lmax_calib))
                visirkeep              = (spectral_merids[:, ifilt_v, 0] >= latmin) & (spectral_merids[:, ifilt_v, 0] <= latmax)            
                visirdata              = spectral_merids[visirkeep, ifilt_v, 3]
                visirmean              = np.nanmean(spectral_merids[visirkeep, ifilt_v, 3])
                # Use IRIS for Q-Band
                calibkeep  = (iris[:, ifilt_sc, 0] >= latmin) & (iris[:, ifilt_sc, 0] <= latmax)
                calib      = iris[:, ifilt_sc, 1]
                calibdata  = iris[calibkeep, ifilt_sc, 1]
                calibmean  = np.nanmean(calibdata)
            # Do calibration
            calib_coeff_spectral[iwave, 0] = wave
            calib_coeff_spectral[iwave, 1] = visirmean / calibmean
            # print(ifilt_sc, visirmean, calibmean, calib_coeff_spectral[iwave, 1])

        # Calculate calibration coefficients for the single merid profiles
        print('Calibrating singles:')
        for ifile, wave in enumerate(wavenumber):
            # Get filter index for spectral profiles
            _, _, ifilt_sc, ifilt_v = set_wave(wavelength=False, wavenumber=wave)
            # Establish shared latitudes for accurate averaging
            lmin_single, lmax_single       = np.nanmin(single_merids[:, ifile, 0]), np.nanmax(single_merids[:, ifile, 0])
            lmin_spectral, lmax_spectral   = np.nanmin(spectral_merids[:, ifilt_v, 0]), np.nanmax(spectral_merids[:, ifilt_v, 0])
            latmin, latmax                 = np.max((lmin_single, lmin_spectral)), np.min((lmax_single, lmax_spectral))
            singlekeep                     = (single_merids[:, ifile, 0] >= latmin) & (single_merids[:, ifile, 0] <= latmax)            
            singledata                     = single_merids[singlekeep, ifile, 3]
            singlemean                     = np.nanmean(singledata)
            spectralkeep                   = (spectral_merids[:, ifilt_v, 0] >= latmin) & (spectral_merids[:, ifilt_v, 0] <= latmax)
            spectraldata                   = spectral_merids[spectralkeep, ifilt_v, 3]
            spectralmean                   = np.nanmean(spectraldata)
            calib_coeff_single[ifile, 0]   = ifile
            calib_coeff_single[ifile, 1]   = singlemean / spectralmean

            # print(ifile, singlemean, spectralmean, calib_coeff_single[ifile, 1])

        # Save calibration
        for ifile in range(nfiles):
            # Calibrate individual merid profiles using individual calibration coefficients
            calib_single_merids = single_merids
            calib_single_merids[:, ifile, 3] /= calib_coeff_single[ifile, 1]
            calib_single_merids[:, ifile, 4] /= calib_coeff_single[ifile, 1]
        for ifilt in range(nfilters):
            # Calibrate spectral merid profiles using spectral calibration coefficients
            calib_spectral_merids = spectral_merids
            calib_spectral_merids[:, ifilt, 3] /= calib_coeff_spectral[ifilt, 1]
            calib_spectral_merids[:, ifilt, 4] /= calib_coeff_spectral[ifilt, 1]

        # # Clear single_merids and spectral_merids arrays from from local variables
        # del locals()['single_merids']
        # del locals()['spectral_merids']

        return calib_single_merids, calib_spectral_merids, calib_coeff_single, calib_coeff_spectral

    def save_arrays(singles, spectrals):
        
        # If subdirectory does not exist, create it
        dir = 'individual_merid_profiles/'
        if not os.path.exists(dir):
            os.makedirs(dir)
        # Save individual meridional profiles
        for ifile, fname in enumerate(files):
            # Extract filename
            name = fname.split('.fits.gz')
            name = name[0].split('/')
            # Write individual mean profiles to np.array
            np.save(f"{dir}{name[1]}_merid_profile", singles[:, ifile, :])
            # Write individual mean profiles to textfile
            np.savetxt(f"{dir}{name[1]}_merid_profile.txt", singles[:, ifile, :],
                        fmt=['%4.2f', '%5.2f', '%4.2f', '%8.5e', '%8.5e', '%8.5f', '%s'],
                        header='LAT    LCM    MU    RAD    ERROR    NU    VIEW')
       
        # If subdirectory does not exist, create it
        dir = 'spectral_merid_profiles/'
        if not os.path.exists(dir):
            os.makedirs(dir)
        # Save spectral meridional profiles
        for ifilt in range(nfilters):
            # Write individual mean profiles to np.array
            filt = visir_wavenumbers(ifilt)
            np.save(f"{dir}{filt}_merid_profile", spectrals[:, ifilt, :])
            # Write individual mean profiles to textfile
            np.savetxt(f"{dir}{filt}_merid_profile.txt", spectrals[:, ifilt, :],
                        fmt=['%4.2f', '%5.2f', '%4.2f', '%8.5e', '%8.5e', '%8.5f'],
                        header='LAT    LCM    MU    RAD    ERROR    NU')

    def plot_profiles(singles, spectrals):
        a = 1
        print('Plotting')

        # plt.figure(dpi=900)
        # ax1 = plt.subplot2grid((2, 1), (0, 0))
        # ax1.plot(lat_grid, calib, color='k', lw=1, label='calib_full')
        # ax1.plot(lat_grid[calibkeep], calibdata, color='k', lw=3, label='calib_keep')
        # ax1.plot(lat_grid[visirkeep], visirdata, color='skyblue', lw=0, marker='o', markersize=3, label='visir_raw')
        # ax1.plot(lat_grid[visirkeep], visirdata/(visirmean/calibmean), color='orange', lw=0, marker='o', markersize=3, label='visir_calib')
        # ax1.set_xlim((-90, 90))
        # ax1.set_ylim((0, 20e-8))
        # ax1.legend()
        # ax2 = plt.subplot2grid((2, 1), (1, 0))
        # ax2.plot(lat_grid[singlekeep], singledata, color='black', lw=0, marker='.', markersize=5, label='single')
        # ax2.plot(lat_grid[spectralkeep], spectraldata, color='red', lw=2, label='spectral')
        # ax2.set_xlim((-90, 90))
        # ax2.set_ylim((0, 20e-8))
        # ax2.legend()
        # plt.savefig('figure.png', dpi=900)

        return a

    def write_spx(spectrals):
        b = 2
        print('spxing')

        return b
    
    ##### Define global inputs #####
    files       = find_files(mode='images')           # Point to location of all input observations
    nfiles      = len(files)                          # Number of input observations
    mu_max      = 80.0                 		          # Maximum emission angle
    lat_range   = -90, 90          			          # Latitude range for binning pixels (planetographic)
    lat_step    = 1                                   # Latitude increment for binning pixels (planetographic)
    lat_grid    = np.arange(-89.5, 90, lat_step)      # Latitude range from pole-to-pole
    n_lat_bins  = len(lat_grid)                       # Number of latitude bins in final profiles
    lon_range   = 360, 0            		          # Longitude range for binning pixels (Sys III)
    merid_width = 30                                  # Longitude range about the central meridian for averaging
    nfilters    = 13                                  # Set manually if using irregularly-sampled data
    # Flags
    calc        = 0                                   # (0) Calculate meridional profiles, (1) read stored profiles
    save        = 1                                   # (0) Do not save (1) save meridional profiles
    plot        = 0                                   # (0) Do not plot (1) plot meridional profiles
    spx         = 0                                   # (0) Do not write (1) do write spxfiles for NEMESIS input


    ##### Run code #####
    # Step 1: Read img, cmap and mufiles
    # Step 2: Geometric registration of pixel information
    # Step 3: Gather pixel information for all files
    # Step 4: Create central meridian average for each observation
    # Step 5: Create central meridian average for each wavelength
    # Step 6: Calibrate result of Step 5 to spacecraft data
    # Step 7: Calibrate individual cmaps to result of Step 6
    # Step 8: Store all cmap profiles and calibration parameters (individual and average)
    # Step 9: Plot profiles
    # Step 10: Create spectra from stored profiles
    
    if calc == 0:
        # Steps 1-3: Generate nested dictionaries containing spatial and spectral information of each cylindrical map
        print('Registering maps...')
        spectrum, wavelength, wavenumber, LCMIII = register_maps()

        # Steps 4-5: Generate average meridional profiles for each observation and each filter
        print('Calculating meridional profiles...')
        single_merids, spectral_merids = create_merid_profiles()
        
        # Steps 6-7: Generate calibrated versions of the profiles from Steps 4 and 5
        print('Calibrating meridional profiles...')
        calib_single_merids, calib_spectral_merids, calib_coeff_single, calib_coeff_spectral = calibrate_merid_profiles()

        # Step 8: Store all cmap profiles and calibration parameters
        if save == 1:
            save_arrays(singles=calib_single_merids, spectrals=calib_spectral_merids)

    # Step 9: Plot meridional profiles (optionally read stored numpy arrays from Step 8)
    if plot == 1:
        if calc == 0:
            plot_profiles(singles=calib_single_merids, spectrals=calib_spectral_merids)
        if calc == 1:
            # Point to stored individual meridional profiles
            profiles1 = find_files(mode='singles')
            profiles2 = find_files(mode='spectrals')
            plot_profiles(singles=profiles1, spectrals=profiles2)
    
    # Step 10: Generate spectral inputs for NEMESIS (optionally read stored numpy arrays from Step 8)
    if spx == 1:
        if calc == 0:
            write_spx(spectrals=calib_spectral_merids)
        if calc == 1:
            # Point to stored individual meridional profiles
            profiles = find_files(mode='spectrals')
            write_spx(spectrals=profiles)

    end = time.time()
    print(f"Elapsed time: {np.round(end-start, 3)} s")
    print(f"Time per file: {np.round(end-start, 3)/len(files)} s")

if __name__ == '__main__':
    main()