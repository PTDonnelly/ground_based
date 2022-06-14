import numpy as np
from BinningInputs import BinningInputs

def WriteSpx(spectrals):
    b = 2
    print('Creating spectra...')
    print(np.shape(spectrals))
    # for ilat in range(BinningInputs.Nlatbins):
    #     lats     = spectrals[ilat, :, 0]
    #     # LCMs     = spectrals[ilat, :, 1]
    #     mus      = spectrals[ilat, :, 2]
    #     # rads     = spectrals[ilat, :, 3]
    #     # rad_errs = spectrals[ilat, :, 4]
    #     # wavenums = spectrals[ilat, :, 5]

    #     print(ilat, lats, mus)

    #     # for imu, mu in enumerate(mus):


    # nconv=mean(data_av[igeom].spts)

    # if (data_av[igeom].wn_start gt 0 and data_av[igeom].emm lt emm_max) then begin

    #     nav=1.0

    #     printf,lun,nconv,format=‘(i10)’
    #     printf,lun,nav,format=‘(i10)’

    #     flat=data_av[igeom].latitude
    #     flon=data_av[igeom].longitude
    #     sol_ang=data_av[igeom].sol
    #     emiss_ang=data_av[igeom].emm
    #     azi_ang= data_av[igeom].azi
    #     wgeom=1.0

    #     printf, flat, flon, sol_ang, emiss_ang, azi_ang, wgeom, format=‘(f12.5,2x,f12.5,2x,f12.5,2x,f12.5,2x,f12.5,2x,f12.5)’
   
    #     out=fltarr(3,nconv)
    #     out(0,*)=data_av[igeom].wn_start
    #     out(1,*)=data_av[igeom].spec
    #     out(2,*)=data_av[igeom].spec_err
   
    #     printf, lun,out,format=‘(f10.4,2x,e15.6,2x,e15.6)’