import xlrd
import numpy as np

def ReadExcelFiles(fpath):

    workbook = xlrd.open_workbook_xls(fpath)

    c2h2sheet = workbook.sheet_by_index(2)
    c2h6sheet = workbook.sheet_by_index(4)

    nrows = c2h2sheet.nrows
    ncols = c2h2sheet.ncols

    nlevel = nrows-1
    nlat = ncols-2

    # Fill pressure array
    pressure = np.empty((nlevel))
    pressure.fill(np.nan)
    for ilev in range(nlevel):
        pressure[ilev] = c2h2sheet.cell_value(ilev+1,0)
    # Fill latitude array
    latitude = np.empty((nlat))
    latitude.fill(np.nan)
    for ilat in range(nlat-1):
        latitude[ilat] = c2h2sheet.cell_value(0,ilat+2)
    
    # Fill C2H2 array
    c2h2 = np.empty((nlevel, nlat))
    c2h2.fill(np.nan)
    for ilev in range (nlevel-1):
        for ilat in range (nlat-1):
            c2h2[ilev, ilat] = c2h2sheet.cell_value(ilev+1,ilat+2)
    # Fill C2H6 array
    c2h6 = np.empty((nlevel, nlat))
    c2h6.fill(np.nan)
    for ilev in range (nlevel-1):
        for ilat in range (nlat-1):
            c2h6[ilev, ilat] = c2h6sheet.cell_value(ilev+1,ilat+2)
    
    return pressure, latitude, c2h2, c2h6