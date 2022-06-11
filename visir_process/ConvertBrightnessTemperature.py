import numpy as np

def ConvertBrightnessTemperature(data, wavelength):
    # Function to convert radiance in brightness temperature
    rad = data *1.e-7
    h = 6.626e-34 # Planck constant
    c = 2.9979e8  # speed of light
    k = 1.3806e-23 # Boltzmann constant
    v = (1e4/wavelength)*100. # in m-1
    rad = rad * 100. # in W m-2 sr-1 (m-1)-1
    rad[rad==0.] = np.nan
    c1 = 2. * h * c * c
    c2 = (h*c)/k
    a = c1 * v * v * v / rad
    temperature = c2 * v / np.log(a+1)
    return temperature