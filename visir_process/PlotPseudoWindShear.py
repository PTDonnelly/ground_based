import os

import matplotlib.pyplot as plt
import numpy as np
from BinningInputs import BinningInputs
from VisirWavenumbers import VisirWavenumbers
from ReadZonalWind import ReadZonalWind


def PlotPseudoWindShear(windshear):
    #  Load Jupiter zonal jets data to determine belts and zones location
    ejets_c, wjets_c, nejet, nwjet = ReadZonalWind("2003porco_zonalu.dat")

    print('Plotting pseudo wind shear...')
    # If subdirectory does not exist, create it
    dir = 'pseudo_wind_shear_profiles_figures/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    for ifilt in range(BinningInputs.nfilters):

        lat=BinningInputs.latgrid
        pdata=windshear[ifilt,:]
        plt.plot(lat[np.where(lat <-5)],pdata[np.where(lat <-5)],linewidth=3.0,color="black")
        plt.plot(lat[np.where(lat >5)],pdata[np.where(lat >5)],linewidth=3.0,color="black")
        plt.plot(lat[np.where((lat <-5) & (pdata<0))],pdata[np.where((lat <-5) & (pdata<0))],"ro")
        plt.plot(lat[np.where((lat <-5) & (pdata>0))],pdata[np.where((lat <-5) & (pdata>0))],"bo")
        plt.plot(lat[np.where((lat >5) & (pdata<0))],pdata[np.where((lat >5) & (pdata<0))],"ro")
        plt.plot(lat[np.where((lat >5) & (pdata>0))],pdata[np.where((lat >5) & (pdata>0))],"bo")
        plt.legend(loc="upper right")
        plt.set_xlim(-90,90)
        plt.set_ylim([-0.7,0.7])
        for i in range(0,nejet):
            plt[ifilt].plot([ejets_c[i],ejets_c[i]],[-15,15],color='black',linestyle="dashed")
        for i in range(0,nwjet):
            plt[ifilt].plot([wjets_c[i],wjets_c[i]],[-15,15],color='black',linestyle="dotted")
        plt[ifilt].plot([-90,90],[0,0],linewidth=1.0,color="grey")
    
        
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        #plt.title("Jupiter VISIR Thermal Windshear", size=15)    
        plt.xlabel('Planetocentric Latitude',size=25)
        plt.ylabel('du/dz [m s$^{-1}$ km$^{-1}$]',size=25)

        # Save figure showing calibation method 
        filt = VisirWavenumbers(ifilt)
        plt.savefig(f"{dir}{filt}_pseudo_wind_shear.png", dpi=900)
        plt.savefig(f"{dir}{filt}_pseudo_wind_shear.eps", dpi=900)