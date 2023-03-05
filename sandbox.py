# @staticmethod
# def make_data_dict(filepaths: List[str]) -> List[dict]:
#     """Create dictionary to store metadata and data products"""

#     # Build dictionary template for each observation
#     dict_template = {
#         "filename": [],
#         "data_products": []  
#         }
#     # Build a list of this dictionary to contain all files in filepaths
#     data_dict = [deepcopy(dict_template) for _ in filepaths]
#     return data_dict

# @classmethod
# def append_products_to_dataset(cls, ifile: int, filepath: object, data_products: Dict[str, npt.ArrayLike], dataset: List[dict]) -> List[dict]:
#     """Add each file to an overall dictionary containing the entire dataset for a given epoch."""
#     dataset[ifile]['filename'] = filepath.stem
#     dataset[ifile]['data_products'] = data_products
#     return

# Dataset.create():

#     # Create dictionary to store metadata and data products
#     dataset = Process.make_data_dict(filepaths)

#     for ifile, filepath in enumerate(filepaths):
#         # Add data_products to the dictionary containing the entire dataset for this epoch."""
#         Process.append_products_to_dataset(ifile, filepath, data_products, dataset)


#############################################



def get_map_extent(self, x_size: int, y_size: int) ->  Tuple[int, int, int, int]:
    """xxx"""
    
    # Point to pre-defined axes limits from Config class
    latrange, lonrange = Config().latitude_range, Config().longitude_range
    
    # Calculate relative bounding indices for horizontal (x, longitude) axes of cylindrical map
    x_dim = x_size
    x_start_relative = lonrange[0] / (360 / x_size)
    x_stop_relative = lonrange[1] / (360 / x_size)
    
    # Calculate relative bounding indices for vertical (y, latitude) axes of cylindrical map
    y_dim = (y_size / 2)
    y_start_relative = latrange[0] / (180 / y_size)
    y_stop_relative = latrange[1] / (180 / y_size)
    
    # Convert to absolute bounding indices, taking into account size of cylindrical map
    x_start_absolute = int(x_dim - x_start_relative)
    x_stop_absolute = int(x_dim - x_stop_relative)
    y_start_absolute = int(y_dim + y_start_relative)
    y_stop_absolute = int(y_dim + y_stop_relative)
    return x_start_absolute, x_stop_absolute, y_start_absolute, y_stop_absolute


def find_data(self) -> List[str]:        

    filepaths = glob.glob(f"{self.data_directory}{self.epoch}/*.fits.gz")

    if (self.epoch == "2016feb"):
        filepaths = [
            f"{self.data_directory}{self.epoch}/wvisir_J7.9_2016-02-15T08:47:39.7606_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_J7.9_2016-02-15T05:21:59.7428_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_J7.9_2016-02-15T05:25:36.2404_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_J7.9_2016-02-15T06:09:03.7779_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_J7.9_2016-02-15T06:12:40.2674_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_J7.9_2016-02-15T08:51:29.2734_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_J7.9_2016-02-16T05:49:55.5712_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_J7.9_2016-02-16T05:53:34.0963_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_J7.9_2016-02-16T08:05:29.5535_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_J7.9_2016-02-16T08:09:05.0788_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_J7.9_2016-02-16T08:53:29.5943_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_J7.9_2016-02-16T08:57:06.0889_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_NEII_1_2016-02-15T05:17:17.5740_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_NEII_1_2016-02-15T05:19:16.2549_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_NEII_1_2016-02-15T06:04:19.6242_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_NEII_1_2016-02-15T06:06:18.2720_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_NEII_1_2016-02-15T08:42:53.5568_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_NEII_1_2016-02-15T08:44:51.2919_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_NEII_1_2016-02-16T05:45:09.4158_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_NEII_1_2016-02-16T05:47:07.1067_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_NEII_1_2016-02-16T08:00:51.4820_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_NEII_1_2016-02-16T08:02:48.1118_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_NEII_1_2016-02-16T08:48:57.4374_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_NEII_1_2016-02-16T08:50:51.1209_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_NEII_2_2016-02-15T04:58:05.2964_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_NEII_2_2016-02-15T05:00:03.2720_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_NEII_2_2016-02-15T05:45:13.3344_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_NEII_2_2016-02-15T05:47:10.2689_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_NEII_2_2016-02-15T08:23:39.3850_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_NEII_2_2016-02-15T08:25:35.2972_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_NEII_2_2016-02-16T05:25:13.1801_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_NEII_2_2016-02-16T05:27:11.0699_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_NEII_2_2016-02-16T07:41:17.1905_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_NEII_2_2016-02-16T07:43:12.0791_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_NEII_2_2016-02-16T08:30:13.2218_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_NEII_2_2016-02-16T08:32:08.0715_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_PAH1_2016-02-15T04:48:49.7459_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_PAH1_2016-02-15T04:50:48.2300_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_PAH1_2016-02-15T05:35:53.1955_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_PAH1_2016-02-15T05:37:53.2969_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_PAH1_2016-02-15T08:14:17.2385_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_PAH1_2016-02-15T08:16:17.2684_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_PAH1_2016-02-16T05:15:37.3243_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_PAH1_2016-02-16T05:17:44.0822_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_PAH1_2016-02-16T07:31:39.6114_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_PAH1_2016-02-16T07:33:38.1577_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_PAH1_2016-02-16T08:21:11.1104_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_PAH1_2016-02-16T08:23:09.0714_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q1_2016-02-15T05:02:43.2867_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q1_2016-02-15T05:04:43.2782_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q1_2016-02-15T05:49:49.3163_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q1_2016-02-15T05:51:50.2548_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q1_2016-02-15T08:28:13.3064_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q1_2016-02-15T08:30:15.2725_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q1_2016-02-16T05:29:59.1336_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q1_2016-02-16T05:32:08.0700_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q1_2016-02-16T07:45:57.1691_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q1_2016-02-16T07:48:01.1119_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q1_2016-02-16T08:34:41.1556_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q1_2016-02-16T08:36:44.0646_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q2_2016-02-15T05:07:27.0295_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q2_2016-02-15T05:09:28.2387_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q2_2016-02-15T05:54:33.0492_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q2_2016-02-15T05:56:34.2312_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q2_2016-02-15T08:32:57.0731_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q2_2016-02-15T08:35:01.2418_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q2_2016-02-16T05:34:58.9051_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q2_2016-02-16T05:37:05.0917_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q2_2016-02-16T07:50:38.9118_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q2_2016-02-16T07:53:14.1398_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q2_2016-02-16T08:39:20.8892_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q2_2016-02-16T08:41:20.0595_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q3_2016-02-15T05:12:15.8414_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q3_2016-02-15T05:14:25.2561_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q3_2016-02-15T05:59:21.8466_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q3_2016-02-15T06:01:31.2718_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q3_2016-02-15T08:37:49.8844_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q3_2016-02-15T08:40:01.2619_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q3_2016-02-16T05:40:01.6675_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q3_2016-02-16T05:42:12.0928_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q3_2016-02-16T07:55:51.6819_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q3_2016-02-16T07:58:00.1225_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q3_2016-02-16T08:44:01.6597_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_Q3_2016-02-16T08:46:09.0626_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_SIV_2_2016-02-15T04:53:27.2566_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_SIV_2_2016-02-15T04:55:22.2554_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_SIV_2_2016-02-15T05:40:37.3549_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_SIV_2_2016-02-15T05:42:34.2363_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_SIV_2_2016-02-15T08:19:05.3466_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_SIV_2_2016-02-15T08:21:00.2466_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_SIV_2_2016-02-16T05:20:29.2882_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_SIV_2_2016-02-16T05:22:31.1138_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_SIV_2_2016-02-16T07:36:17.2161_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_SIV_2_2016-02-16T07:38:10.1222_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_SIV_2_2016-02-16T08:25:45.1583_Jupiter_clean_withchop.fits.gz",
            f"{self.data_directory}{self.epoch}/wvisir_SIV_2_2016-02-16T08:27:38.0963_Jupiter_clean_withchop.fits.gz"
        ]
    elif (self.epoch == "may2018"):
        filepaths = [
                f"{self.data_directory}2018/2018may/wvisir_ARIII_2018-05-25T03:45:53.1419_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_ARIII_2018-05-26T00:03:13.9569_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_ARIII_2018-05-26T05:44:15.9897_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_ARIII_2018-05-26T07:19:42.0774_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_ARIII_2018-05-26T23:29:55.9051_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_ARIII_2018-05-27T00:39:50.8993_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_ARIII_2018-05-27T02:39:55.9284_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_J7.9_2018-05-24T02:49:57.5419_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_J7.9_2018-05-24T05:51:00.6121_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_J7.9_2018-05-25T00:22:22.2779_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_J7.9_2018-05-25T04:08:22.2746_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_J7.9_2018-05-26T00:21:09.9792_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_J7.9_2018-05-26T06:02:47.9655_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_J7.9_2018-05-26T07:38:32.1062_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_J7.9_2018-05-26T23:49:49.9432_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_NEII_1_2018-05-24T06:10:42.6362_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_NEII_1_2018-05-25T00:11:20.2744_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_NEII_1_2018-05-25T03:55:29.2519_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_NEII_1_2018-05-26T00:12:26.0008_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_NEII_1_2018-05-26T05:55:47.1495_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_NEII_1_2018-05-26T07:29:24.0714_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_NEII_1_2018-05-26T23:37:12.7783_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_NEII_1_2018-05-27T00:47:16.7533_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_NEII_2_2018-05-24T02:45:35.7531_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_NEII_2_2018-05-24T03:39:18.6152_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_NEII_2_2018-05-24T05:42:53.6526_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_NEII_2_2018-05-24T07:15:35.6747_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_NEII_2_2018-05-25T00:13:56.7964_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_NEII_2_2018-05-25T04:02:03.2717_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_NEII_2_2018-05-26T00:16:46.6576_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_NEII_2_2018-05-26T06:00:13.1871_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_NEII_2_2018-05-26T23:43:36.9132_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_PAH1_2018-05-24T02:36:43.5199_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_PAH1_2018-05-24T03:27:08.2479_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_PAH1_2018-05-24T04:49:11.6224_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_PAH1_2018-05-24T05:34:02.1968_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_PAH1_2018-05-25T03:23:18.2624_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_PAH1_2018-05-25T23:44:28.6926_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_PAH1_2018-05-26T01:13:36.7987_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_PAH1_2018-05-26T23:09:18.7096_Jupiter_clean_withchop.fits.gz",
                # f"{self.data_directory}2018/2018may/wvisir_PAH2_2_2018-05-24T06:41:15.5878_Jupiter_cburst_withchop.fits.gz",
                # f"{self.data_directory}2018/2018may/wvisir_PAH2_2018-05-24T06:33:21.6415_Jupiter_clean_withchop.fits.gz",
                # f"{self.data_directory}2018/2018may/wvisir_PAH2_2018-05-24T06:37:01.6178_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_Q1_2018-05-25T04:24:52.2863_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_Q1_2018-05-26T00:42:18.1412_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_Q1_2018-05-26T06:23:41.1581_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_Q1_2018-05-26T07:56:58.6145_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_Q1_2018-05-27T00:05:03.3551_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_Q1_2018-05-27T01:16:37.3572_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_Q1_2018-05-27T03:15:15.3243_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_Q2_2018-05-25T04:18:05.8435_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_Q2_2018-05-26T00:35:44.6206_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_Q2_2018-05-26T06:17:10.6564_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_Q2_2018-05-26T07:54:25.1460_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_Q2_2018-05-27T00:02:20.9139_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_Q2_2018-05-27T01:21:22.7250_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_Q2_2018-05-27T03:10:37.3636_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_Q3_2018-05-25T04:14:46.2779_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_Q3_2018-05-26T00:31:03.1137_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_Q3_2018-05-26T06:10:47.8855_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_Q3_2018-05-26T07:46:39.9647_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_Q3_2018-05-26T23:56:07.9176_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_Q3_2018-05-27T01:06:08.9103_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_Q3_2018-05-27T03:06:37.8961_Jupiter_clean_withchop.fits.gz",
                # f"{self.data_directory}2018/2018may/wvisir_SIV_1_2018-05-24T06:24:13.7864_Jupiter_clean_withchop.fits.gz",
                # f"{self.data_directory}2018/2018may/wvisir_SIV_1_2018-05-24T06:26:07.6279_Jupiter_clean_withchop.fits.gz",
                # f"{self.data_directory}2018/2018may/wvisir_SIV_1_2018-05-24T23:55:04.5471_Jupiter_clean_withchop.fits.gz",
                # f"{self.data_directory}2018/2018may/wvisir_SIV_1_2018-05-24T23:57:02.2345_Jupiter_clean_withchop.fits.gz",
                # f"{self.data_directory}2018/2018may/wvisir_SIV_1_2018-05-25T03:35:43.6183_Jupiter_clean_withchop.fits.gz",
                # f"{self.data_directory}2018/2018may/wvisir_SIV_1_2018-05-25T03:37:39.2908_Jupiter_clean_withchop.fits.gz",
                # f"{self.data_directory}2018/2018may/wvisir_SIV_1_2018-05-25T03:41:11.7568_Jupiter_clean_withchop.fits.gz",
                # f"{self.data_directory}2018/2018may/wvisir_SIV_1_2018-05-25T03:43:11.3128_Jupiter_clean_withchop.fits.gz",
                # f"{self.data_directory}2018/2018may/wvisir_SIV_1_2018-05-25T23:58:36.3950_Jupiter_clean_withchop.fits.gz",
                # f"{self.data_directory}2018/2018may/wvisir_SIV_1_2018-05-26T00:00:38.1179_Jupiter_clean_withchop.fits.gz",
                # f"{self.data_directory}2018/2018may/wvisir_SIV_1_2018-05-26T01:27:26.5086_Jupiter_clean_withchop.fits.gz",
                # f"{self.data_directory}2018/2018may/wvisir_SIV_1_2018-05-26T01:29:21.1227_Jupiter_clean_withchop.fits.gz",
                # f"{self.data_directory}2018/2018may/wvisir_SIV_1_2018-05-26T05:39:46.4224_Jupiter_clean_withchop.fits.gz",
                # f"{self.data_directory}2018/2018may/wvisir_SIV_1_2018-05-26T05:41:42.1232_Jupiter_clean_withchop.fits.gz",
                # f"{self.data_directory}2018/2018may/wvisir_SIV_1_2018-05-26T07:15:06.4006_Jupiter_clean_withchop.fits.gz",
                # f"{self.data_directory}2018/2018may/wvisir_SIV_1_2018-05-26T07:17:02.1476_Jupiter_clean_withchop.fits.gz",
                # f"{self.data_directory}2018/2018may/wvisir_SIV_1_2018-05-26T23:23:17.1633_Jupiter_clean_withchop.fits.gz",
                # f"{self.data_directory}2018/2018may/wvisir_SIV_1_2018-05-26T23:25:14.8980_Jupiter_clean_withchop.fits.gz",
                # f"{self.data_directory}2018/2018may/wvisir_SIV_1_2018-05-27T00:33:16.9997_Jupiter_clean_withchop.fits.gz",
                # f"{self.data_directory}2018/2018may/wvisir_SIV_1_2018-05-27T00:35:11.8896_Jupiter_clean_withchop.fits.gz",
                # f"{self.data_directory}2018/2018may/wvisir_SIV_1_2018-05-27T02:32:55.1965_Jupiter_clean_withchop.fits.gz",
                # f"{self.data_directory}2018/2018may/wvisir_SIV_1_2018-05-27T02:35:06.9274_Jupiter_clean_withchop.fits.gz",
                # f"{self.data_directory}2018/2018may/wvisir_SIV_1_2018-05-27T03:57:35.1621_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_SIV_2_2018-05-24T04:53:42.6259_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_SIV_2_2018-05-24T05:38:31.6363_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_SIV_2_2018-05-25T23:49:02.1191_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_SIV_2_2018-05-26T01:20:03.1395_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_SIV_2_2018-05-26T05:32:18.1273_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_SIV_2_2018-05-26T23:15:47.9167_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_SIV_2_2018-05-27T00:22:51.9127_Jupiter_clean_withchop.fits.gz",
                f"{self.data_directory}2018/2018may/wvisir_SIV_2_2018-05-27T03:49:46.9099_Jupiter_clean_withchop.fits.gz"
                # f"{self.data_directory}2018/2018may/wvisir_SIV_2018-05-24T23:52:11.2499_Jupiter_clean_withchop.fits.gz",
                # f"{self.data_directory}2018/2018may/wvisir_SIV_2018-05-25T23:53:41.9626_Jupiter_clean_withchop.fits.gz",
                # f"{self.data_directory}2018/2018may/wvisir_SIV_2018-05-26T01:22:38.0760_Jupiter_clean_withchop.fits.gz",
                # f"{self.data_directory}2018/2018may/wvisir_SIV_2018-05-26T05:36:57.1244_Jupiter_clean_withchop.fits.gz",
                # f"{self.data_directory}2018/2018may/wvisir_SIV_2018-05-26T23:18:24.7716_Jupiter_clean_withchop.fits.gz",
                # f"{self.data_directory}2018/2018may/wvisir_SIV_2018-05-27T00:25:24.7290_Jupiter_clean_withchop.fits.gz",
                # f"{self.data_directory}2018/2018may/wvisir_SIV_2018-05-27T02:27:30.7403_Jupiter_clean_withchop.fits.gz",
                # f"{self.data_directory}2018/2018may/wvisir_SIV_2018-05-27T03:54:48.9331_Jupiter_clean_withchop.fits.gz"
            ]        
    return self.pathify(filepaths)