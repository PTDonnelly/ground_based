import numpy as np

def ReadZonalWind(filename):
    """ DB: Function to load Jupiter zonal wind data from 
            Porco et al., Science 299, 1541-1547 (2003) """
    # Load Jupiter zonal jets data to determine belts and zones location
    jets_lines = np.loadtxt(filename)
    latpc = jets_lines[:,1]
    latpg = jets_lines[:,2]
    speed = jets_lines[:,4]
    cond = np.where(speed>0)
    ejets_c = latpc[cond]
    nejet = ejets_c.size
    cond = np.where(speed<0)
    wjets_c = latpc[cond]
    nwjet = wjets_c.size
    return ejets_c, wjets_c, nejet, nwjet