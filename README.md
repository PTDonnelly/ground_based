# ground_based
Processing ground-based planetary observations for mapping and atmospheric retrieval.

The structure of GiantPipe version 2 uses a mix of Object-Oriented and Functional (OOF) design principles. It reproduces the functionality of version 1 but has the primary goal of being is intended to be more readable and user-friendly, as well as allowing straightforward future code expansion and maintenance. Increases in speed and memory efficiency are expected, but are a secondary to readability and code robustness.

#### Note: 

This code is still currently coupled quite heavily to the overall structure of the VISIR images and NASA/DRM cylindrical map outputs.

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
  
  ```def get_thing():``` Used when calculating manipulating existing variables (e.g. from data files)
  
  ```def make_thing():``` Used when generating new variables or changing existing ones
 
