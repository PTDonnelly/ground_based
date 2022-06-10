import numpy as np
import matplotlib.pyplot as plt

def CalculateErrors(image, view):
    """Calculate radiance error from the variability background sky flux"""

    # # Set to a constant typical value
    # error = 0.05

    # Only use unaffected hemisphere
    ny, nx = np.shape(image)
    if view == 1:
        y1, y2 = int(ny/2), -1
        img = image[y1:y2, :]
    if view == -1:
        y1, y2 = 0, int(ny/2)-1
        img = image[y1:y2, :]
    # Set a radiance threshold to separate "planet" from "sky" flux
    thresh = 0.25
    # Calculate mean radiance value of the image
    mean_flux = np.mean(img)
    # Isolate sky
    keep  = (img < thresh * mean_flux)
    sky   = img[keep]
    error = np.std(sky)
    # print(f"STDDEV: {error}")

    # sky  = img[np.ix_(keep[0], keep[1])]
    # plt.figure
    # ax = plt.subplot2grid((1, 2), (0, 0))
    # ax.imshow(img, origin="lower")
    # ax = plt.subplot2grid((1, 2), (0, 1))
    # ax.imshow(sky, origin="lower")
    # plt.show()
    
    return error