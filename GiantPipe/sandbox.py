

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


# Calculate horizontal and vertical extent of image as absolute array indices
x_min, x_max, y_min, y_max = self.get_map_extent(x_size=x_size, y_size=y_size)