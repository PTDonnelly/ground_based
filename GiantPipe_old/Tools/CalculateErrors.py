import numpy as np
import matplotlib.pyplot as plt

def CalculateErrors(image, view):
    """Calculate radiance error from the variability background sky flux"""

    calc_sd_sky = False

    if calc_sd_sky:
        # Only use unaffected hemisphere
        ny, nx = np.shape(image)
        img = image[int(ny/2):-1, :] if view == 1 else image
        img = image[0:int(ny/2)-1, :] if view == -1 else image
        # Set a radiance threshold to separate "planet" from "sky" flux
        thresh = 0.1
        # Calculate mean radiance value of the image
        mean_flux = np.mean(img)
        # Isolate sky
        keep  = (img < thresh * mean_flux)
        sky   = img[keep]
        error = np.std(sky)
    else:
        # Set to a constant typical value
        error = 0.05

    return error