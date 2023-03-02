# ground_based
Processing ground-based planetary observations for mapping and atmospheric retrieval.

The structure of GiantPipe version 2 uses a mix of Object-Oriented and Functional (OOF) design principles. It reproduces the functionality of version 1 but has the primary goal of being is intended to be more readable and user-friendly, as well as allowing straightforward future code expansion and maintenance. Increases in speed and memory efficiency are expected, but are a secondary to readability and code robustness.

#### Note: 

This code is still currently coupled quite heavily to the overall structure of the VISIR images and NASA/DRM cylindrical map outputs.

# Config class

### The pathing for pointing to the observations assumes that your data is in a hierarchy like:
    /root/data/visir/2016feb/wvisir_J7.9_2016-02-15T08:47:39.7606_Jupiter_clean_withchop.fits.gz

    /root/data/visir/2016feb//wvisir_Q1_2016-02-15T05:02:43.2867_Jupiter_clean_withchop.fits.gz

    ...

    /root/data/visir/2018may/wvisir_NEII_1_2018-05-24T06:10:42.6362_Jupiter_clean_withchop.fits.gz

    /root/data/visir/2018may/wvisir_SIV_2_2018-05-24T04:53:42.6259_Jupiter_clean_withchop.fits.gz

    so for example:

    data_directory = Config.data_directory = "/root/data/visir/"
    epoch = Config.epoch = "2016feb"
    filepath = f"{data_directory}{epoch}/wvisir_J7.9_2016-02-15T08:47:39.7606_Jupiter_clean_withchop.fits.gz"


# Code Design

## Classes

Classes are either data-focused (containing mostly/all attributes) or method-focused (containing mostly/all functions). It is normally clear which is which, but it is always stated explicitly. Data-focused classes used the @dataclass decorator for simplicity and moethod-focused classes are defined as usual.

## Class names

1) Should be a noun.
2) Always capitalised.
3) If multiple words, no underscore and successive words are capitalised.

Examples:

```class Datamap:```

```class CalibratedDatamap:```

## Function names

1) Should be a verb.
2) Always lower case
3) If multiple words, an underscore separates each word.
4) Prefix establishes what the function does

Examples:

  ```def read_thing():``` Used when reading files and their information
  
  ```def get_thing():``` Used when calculating from or manipulating existing variables (e.g. from data files)
  
  ```def make_thing():``` Used when generating new variables or changing existing ones
 
