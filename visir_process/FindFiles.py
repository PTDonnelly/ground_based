def FindFiles(mode):
    """Put all input data in a single list"""

    if mode == 'images':
        # files = ['cal_wvisir_NEII_2_2018-05-24T02_45_35.7531_Jupiter_clean_withchop_calib_coeff.npy']
        # data_dir = '/Users/db496/Documents/Research/Observations/GBdata/visir/jupiter/global_maps_dataset/'
        data_dir = '/Users/ptdonnelly/Documents/Research/projects/shared/dbardet/data/'
        files = [f"{data_dir}recal_wvisir_ARIII_2018-05-25T03:45:53.1419_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_ARIII_2018-05-26T00:03:13.9569_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_ARIII_2018-05-26T05:44:15.9897_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_ARIII_2018-05-26T07:19:42.0774_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_ARIII_2018-05-26T23:29:55.9051_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_ARIII_2018-05-27T00:39:50.8993_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_ARIII_2018-05-27T02:39:55.9284_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_J7.9_2018-05-24T02:49:57.5419_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_J7.9_2018-05-24T05:51:00.6121_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_J7.9_2018-05-25T00:22:22.2779_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_J7.9_2018-05-25T04:08:22.2746_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_J7.9_2018-05-26T00:21:09.9792_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_J7.9_2018-05-26T06:02:47.9655_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_J7.9_2018-05-26T07:38:32.1062_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_J7.9_2018-05-26T23:49:49.9432_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_NEII_1_2018-05-24T06:10:42.6362_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_NEII_1_2018-05-25T00:11:20.2744_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_NEII_1_2018-05-25T03:55:29.2519_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_NEII_1_2018-05-26T00:12:26.0008_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_NEII_1_2018-05-26T05:55:47.1495_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_NEII_1_2018-05-26T07:29:24.0714_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_NEII_1_2018-05-26T23:37:12.7783_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_NEII_1_2018-05-27T00:47:16.7533_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_NEII_2_2018-05-24T02:45:35.7531_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_NEII_2_2018-05-24T03:39:18.6152_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_NEII_2_2018-05-24T05:42:53.6526_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_NEII_2_2018-05-24T07:15:35.6747_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_NEII_2_2018-05-25T00:13:56.7964_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_NEII_2_2018-05-25T04:02:03.2717_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_NEII_2_2018-05-26T00:16:46.6576_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_NEII_2_2018-05-26T06:00:13.1871_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_NEII_2_2018-05-26T23:43:36.9132_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_PAH1_2018-05-24T02:36:43.5199_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_PAH1_2018-05-24T03:27:08.2479_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_PAH1_2018-05-24T04:49:11.6224_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_PAH1_2018-05-24T05:34:02.1968_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_PAH1_2018-05-25T03:23:18.2624_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_PAH1_2018-05-25T23:44:28.6926_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_PAH1_2018-05-26T01:13:36.7987_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_PAH1_2018-05-26T23:09:18.7096_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_PAH2_2_2018-05-24T06:41:15.5878_Jupiter_cburst_withchop.fits.gz",
                f"{data_dir}recal_wvisir_PAH2_2018-05-24T06:33:21.6415_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_PAH2_2018-05-24T06:37:01.6178_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_Q1_2018-05-25T04:24:52.2863_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_Q1_2018-05-26T00:42:18.1412_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_Q1_2018-05-26T06:23:41.1581_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_Q1_2018-05-26T07:56:58.6145_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_Q1_2018-05-27T00:05:03.3551_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_Q1_2018-05-27T01:16:37.3572_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_Q1_2018-05-27T03:15:15.3243_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_Q2_2018-05-25T04:18:05.8435_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_Q2_2018-05-26T00:35:44.6206_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_Q2_2018-05-26T06:17:10.6564_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_Q2_2018-05-26T07:54:25.1460_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_Q2_2018-05-27T00:02:20.9139_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_Q2_2018-05-27T01:21:22.7250_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_Q2_2018-05-27T03:10:37.3636_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_Q3_2018-05-25T04:14:46.2779_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_Q3_2018-05-26T00:31:03.1137_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_Q3_2018-05-26T06:10:47.8855_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_Q3_2018-05-26T07:46:39.9647_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_Q3_2018-05-26T23:56:07.9176_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_Q3_2018-05-27T01:06:08.9103_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_Q3_2018-05-27T03:06:37.8961_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_SIV_1_2018-05-24T06:24:13.7864_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_SIV_1_2018-05-24T23:57:02.2345_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_SIV_1_2018-05-25T03:41:11.7568_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_SIV_1_2018-05-26T00:00:38.1179_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_SIV_1_2018-05-26T01:27:26.5086_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_SIV_1_2018-05-26T05:39:46.4224_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_SIV_1_2018-05-26T07:15:06.4006_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_SIV_1_2018-05-26T23:25:14.8980_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_SIV_1_2018-05-27T00:35:11.8896_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_SIV_1_2018-05-27T02:32:55.1965_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_SIV_2_2018-05-24T04:53:42.6259_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_SIV_2_2018-05-24T05:38:31.6363_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_SIV_2_2018-05-25T23:49:02.1191_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_SIV_2_2018-05-26T01:20:03.1395_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_SIV_2_2018-05-26T05:32:18.1273_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_SIV_2_2018-05-26T23:15:47.9167_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_SIV_2_2018-05-27T00:22:51.9127_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_SIV_2_2018-05-27T03:49:46.9099_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_SIV_2018-05-24T23:52:11.2499_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_SIV_2018-05-25T23:53:41.9626_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_SIV_2018-05-26T01:22:38.0760_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_SIV_2018-05-26T05:36:57.1244_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_SIV_2018-05-26T23:18:24.7716_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_SIV_2018-05-27T00:25:24.7290_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_SIV_2018-05-27T02:27:30.7403_Jupiter_clean_withchop.fits.gz",
                f"{data_dir}recal_wvisir_SIV_2018-05-27T03:54:48.9331_Jupiter_clean_withchop.fits.gz"]

    if mode == 'singles':
        profile_dir = '../outputs/single_merid_profiles/'
        files = [f"{profile_dir}cal_wvisir_ARIII_2018-05-25T03:45:53.1419_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_ARIII_2018-05-26T00:03:13.9569_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_ARIII_2018-05-26T05:44:15.9897_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_ARIII_2018-05-26T07:19:42.0774_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_ARIII_2018-05-26T23:29:55.9051_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_ARIII_2018-05-27T00:39:50.8993_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_ARIII_2018-05-27T02:39:55.9284_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_J7.9_2018-05-24T02:49:57.5419_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_J7.9_2018-05-24T05:51:00.6121_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_J7.9_2018-05-25T00:22:22.2779_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_J7.9_2018-05-25T04:08:22.2746_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_J7.9_2018-05-26T00:21:09.9792_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_J7.9_2018-05-26T06:02:47.9655_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_J7.9_2018-05-26T07:38:32.1062_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_J7.9_2018-05-26T23:49:49.9432_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_NEII_1_2018-05-24T06:10:42.6362_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_NEII_1_2018-05-25T00:11:20.2744_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_NEII_1_2018-05-25T03:55:29.2519_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_NEII_1_2018-05-26T00:12:26.0008_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_NEII_1_2018-05-26T05:55:47.1495_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_NEII_1_2018-05-26T07:29:24.0714_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_NEII_1_2018-05-26T23:37:12.7783_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_NEII_1_2018-05-27T00:47:16.7533_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_NEII_2_2018-05-24T02:45:35.7531_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_NEII_2_2018-05-24T03:39:18.6152_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_NEII_2_2018-05-24T05:42:53.6526_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_NEII_2_2018-05-24T07:15:35.6747_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_NEII_2_2018-05-25T00:13:56.7964_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_NEII_2_2018-05-25T04:02:03.2717_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_NEII_2_2018-05-26T00:16:46.6576_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_NEII_2_2018-05-26T06:00:13.1871_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_NEII_2_2018-05-26T23:43:36.9132_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_PAH1_2018-05-24T02:36:43.5199_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_PAH1_2018-05-24T03:27:08.2479_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_PAH1_2018-05-24T04:49:11.6224_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_PAH1_2018-05-24T05:34:02.1968_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_PAH1_2018-05-25T03:23:18.2624_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_PAH1_2018-05-25T23:44:28.6926_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_PAH1_2018-05-26T01:13:36.7987_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_PAH1_2018-05-26T23:09:18.7096_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_PAH2_2_2018-05-24T06:41:15.5878_Jupiter_cburst_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_PAH2_2018-05-24T06:33:21.6415_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_PAH2_2018-05-24T06:37:01.6178_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_Q1_2018-05-25T04:24:52.2863_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_Q1_2018-05-26T00:42:18.1412_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_Q1_2018-05-26T06:23:41.1581_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_Q1_2018-05-26T07:56:58.6145_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_Q1_2018-05-27T00:05:03.3551_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_Q1_2018-05-27T01:16:37.3572_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_Q1_2018-05-27T03:15:15.3243_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_Q2_2018-05-25T04:18:05.8435_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_Q2_2018-05-26T00:35:44.6206_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_Q2_2018-05-26T06:17:10.6564_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_Q2_2018-05-26T07:54:25.1460_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_Q2_2018-05-27T00:02:20.9139_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_Q2_2018-05-27T01:21:22.7250_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_Q2_2018-05-27T03:10:37.3636_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_Q3_2018-05-25T04:14:46.2779_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_Q3_2018-05-26T00:31:03.1137_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_Q3_2018-05-26T06:10:47.8855_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_Q3_2018-05-26T07:46:39.9647_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_Q3_2018-05-26T23:56:07.9176_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_Q3_2018-05-27T01:06:08.9103_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_Q3_2018-05-27T03:06:37.8961_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_SIV_1_2018-05-24T06:24:13.7864_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_SIV_1_2018-05-24T23:57:02.2345_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_SIV_1_2018-05-25T03:41:11.7568_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_SIV_1_2018-05-26T00:00:38.1179_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_SIV_1_2018-05-26T01:27:26.5086_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_SIV_1_2018-05-26T05:39:46.4224_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_SIV_1_2018-05-26T07:15:06.4006_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_SIV_1_2018-05-26T23:25:14.8980_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_SIV_1_2018-05-27T00:35:11.8896_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_SIV_1_2018-05-27T02:32:55.1965_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_SIV_2_2018-05-24T04:53:42.6259_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_SIV_2_2018-05-24T05:38:31.6363_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_SIV_2_2018-05-25T23:49:02.1191_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_SIV_2_2018-05-26T01:20:03.1395_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_SIV_2_2018-05-26T05:32:18.1273_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_SIV_2_2018-05-26T23:15:47.9167_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_SIV_2_2018-05-27T00:22:51.9127_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_SIV_2_2018-05-27T03:49:46.9099_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_SIV_2018-05-24T23:52:11.2499_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_SIV_2018-05-25T23:53:41.9626_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_SIV_2018-05-26T01:22:38.0760_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_SIV_2018-05-26T05:36:57.1244_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_SIV_2018-05-26T23:18:24.7716_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_SIV_2018-05-27T00:25:24.7290_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_SIV_2018-05-27T02:27:30.7403_Jupiter_clean_withchop_merid_profile.npy",
                f"{profile_dir}cal_wvisir_SIV_2018-05-27T03:54:48.9331_Jupiter_clean_withchop_merid_profile.npy"]

    if mode == 'ksingles':
        profile_dir = '../outputs/single_merid_profiles/'
        files = [f"{profile_dir}cal_wvisir_ARIII_2018-05-25T03:45:53.1419_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_ARIII_2018-05-26T00:03:13.9569_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_ARIII_2018-05-26T05:44:15.9897_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_ARIII_2018-05-26T07:19:42.0774_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_ARIII_2018-05-26T23:29:55.9051_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_ARIII_2018-05-27T00:39:50.8993_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_ARIII_2018-05-27T02:39:55.9284_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_J7.9_2018-05-24T02:49:57.5419_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_J7.9_2018-05-24T05:51:00.6121_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_J7.9_2018-05-25T00:22:22.2779_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_J7.9_2018-05-25T04:08:22.2746_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_J7.9_2018-05-26T00:21:09.9792_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_J7.9_2018-05-26T06:02:47.9655_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_J7.9_2018-05-26T07:38:32.1062_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_J7.9_2018-05-26T23:49:49.9432_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_NEII_1_2018-05-24T06:10:42.6362_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_NEII_1_2018-05-25T00:11:20.2744_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_NEII_1_2018-05-25T03:55:29.2519_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_NEII_1_2018-05-26T00:12:26.0008_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_NEII_1_2018-05-26T05:55:47.1495_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_NEII_1_2018-05-26T07:29:24.0714_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_NEII_1_2018-05-26T23:37:12.7783_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_NEII_1_2018-05-27T00:47:16.7533_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_NEII_2_2018-05-24T02:45:35.7531_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_NEII_2_2018-05-24T03:39:18.6152_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_NEII_2_2018-05-24T05:42:53.6526_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_NEII_2_2018-05-24T07:15:35.6747_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_NEII_2_2018-05-25T00:13:56.7964_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_NEII_2_2018-05-25T04:02:03.2717_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_NEII_2_2018-05-26T00:16:46.6576_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_NEII_2_2018-05-26T06:00:13.1871_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_NEII_2_2018-05-26T23:43:36.9132_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_PAH1_2018-05-24T02:36:43.5199_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_PAH1_2018-05-24T03:27:08.2479_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_PAH1_2018-05-24T04:49:11.6224_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_PAH1_2018-05-24T05:34:02.1968_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_PAH1_2018-05-25T03:23:18.2624_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_PAH1_2018-05-25T23:44:28.6926_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_PAH1_2018-05-26T01:13:36.7987_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_PAH1_2018-05-26T23:09:18.7096_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_PAH2_2_2018-05-24T06:41:15.5878_Jupiter_cburst_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_PAH2_2018-05-24T06:33:21.6415_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_PAH2_2018-05-24T06:37:01.6178_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_Q1_2018-05-25T04:24:52.2863_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_Q1_2018-05-26T00:42:18.1412_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_Q1_2018-05-26T06:23:41.1581_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_Q1_2018-05-26T07:56:58.6145_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_Q1_2018-05-27T00:05:03.3551_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_Q1_2018-05-27T01:16:37.3572_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_Q1_2018-05-27T03:15:15.3243_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_Q2_2018-05-25T04:18:05.8435_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_Q2_2018-05-26T00:35:44.6206_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_Q2_2018-05-26T06:17:10.6564_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_Q2_2018-05-26T07:54:25.1460_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_Q2_2018-05-27T00:02:20.9139_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_Q2_2018-05-27T01:21:22.7250_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_Q2_2018-05-27T03:10:37.3636_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_Q3_2018-05-25T04:14:46.2779_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_Q3_2018-05-26T00:31:03.1137_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_Q3_2018-05-26T06:10:47.8855_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_Q3_2018-05-26T07:46:39.9647_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_Q3_2018-05-26T23:56:07.9176_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_Q3_2018-05-27T01:06:08.9103_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_Q3_2018-05-27T03:06:37.8961_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_SIV_1_2018-05-24T06:24:13.7864_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_SIV_1_2018-05-24T23:57:02.2345_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_SIV_1_2018-05-25T03:41:11.7568_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_SIV_1_2018-05-26T00:00:38.1179_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_SIV_1_2018-05-26T01:27:26.5086_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_SIV_1_2018-05-26T05:39:46.4224_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_SIV_1_2018-05-26T07:15:06.4006_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_SIV_1_2018-05-26T23:25:14.8980_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_SIV_1_2018-05-27T00:35:11.8896_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_SIV_1_2018-05-27T02:32:55.1965_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_SIV_2_2018-05-24T04:53:42.6259_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_SIV_2_2018-05-24T05:38:31.6363_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_SIV_2_2018-05-25T23:49:02.1191_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_SIV_2_2018-05-26T01:20:03.1395_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_SIV_2_2018-05-26T05:32:18.1273_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_SIV_2_2018-05-26T23:15:47.9167_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_SIV_2_2018-05-27T00:22:51.9127_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_SIV_2_2018-05-27T03:49:46.9099_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_SIV_2018-05-24T23:52:11.2499_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_SIV_2018-05-25T23:53:41.9626_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_SIV_2018-05-26T01:22:38.0760_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_SIV_2018-05-26T05:36:57.1244_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_SIV_2018-05-26T23:18:24.7716_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_SIV_2018-05-27T00:25:24.7290_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_SIV_2018-05-27T02:27:30.7403_Jupiter_clean_withchop_calib_coeff.npy",
                f"{profile_dir}cal_wvisir_SIV_2018-05-27T03:54:48.9331_Jupiter_clean_withchop_calib_coeff.npy"]
    
    if mode == 'spectrals':
        profile_dir = '../outputs/spectral_merid_profiles/'
        files = [f"{profile_dir}511.97345_merid_profile.npy",
                f"{profile_dir}532.73522_merid_profile.npy",
                f"{profile_dir}563.97095_merid_profile.npy",
                f"{profile_dir}766.75227_merid_profile.npy",
                f"{profile_dir}815.7001_merid_profile.npy",
                f"{profile_dir}852.115_merid_profile.npy",
                f"{profile_dir}887.524_merid_profile.npy",
                f"{profile_dir}929.27953_merid_profile.npy",
                f"{profile_dir}956.922_merid_profile.npy",
                f"{profile_dir}1012.52_merid_profile.npy",
                f"{profile_dir}1112.15_merid_profile.npy",
                f"{profile_dir}1163.0755_merid_profile.npy",
                f"{profile_dir}1253.0625_merid_profile.npy"]
    
    if mode == 'kspectrals':
        profile_dir = '../outputs/spectral_merid_profiles/'
        files = [f"{profile_dir}766.75227_calib_coeff.npy",
                f"{profile_dir}815.7001_calib_coeff.npy",
                f"{profile_dir}852.115_calib_coeff.npy",
                f"{profile_dir}887.524_calib_coeff.npy",
                f"{profile_dir}929.27953_calib_coeff.npy",
                f"{profile_dir}956.922_calib_coeff.npy",
                f"{profile_dir}1012.52_calib_coeff.npy",
                f"{profile_dir}1112.15_calib_coeff.npy",
                f"{profile_dir}1163.0755_calib_coeff.npy",
                f"{profile_dir}1253.0625_calib_coeff.npy",
                f"{profile_dir}511.97345_calib_coeff.npy",
                f"{profile_dir}532.73522_calib_coeff.npy",
                f"{profile_dir}563.97095_calib_coeff.npy"]

    return files