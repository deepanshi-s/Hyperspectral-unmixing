import scipy.io as spio
import pysptools.eea as eea
import pysptools.abundance_maps as amp
import numpy as np
import xlrd
import math

# Give the location of the file
loc = ("End members.xlsx")
 
# To open Workbook
wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)

#read endmembers from excel file
veg = np.array(list(sheet.row_values(4)[1:261]))
soil = np.array(list(sheet.row_values(1)[1:]))

#resample soil endmember
soil_ind = np.array(list(sheet.row_values(0)[1:]))
soil_ind = soil_ind.astype(int)

veg_ind = np.ceil(np.array(list(sheet.row_values(3)[1:261])))
veg_ind = veg_ind.astype(int)

soil_sampled = np.zeros(len(veg))

for i in range(len(veg)):
  soil_sampled[i] = soil[np.where(soil_ind == veg_ind[i])]

#combine both the endmembers arrays
soil_sampled = soil_sampled.reshape((soil_sampled.shape[0],1))
veg = veg.reshape((veg.shape[0],1))
endmem = np.concatenate((soil_sampled,veg),axis = -1)
endmem = endmem.reshape((endmem.shape[1], endmem.shape[0]))
print(endmem.shape)

#read the HSIcube from excel file
loc1 = ("AVIRIS spectra.xlsx")
wb1 = xlrd.open_workbook(loc1)
sheet1 = wb1.sheet_by_index(0)
print(sheet1.nrows, sheet1.ncols)

hsi_cube = np.zeros((sheet1.nrows-2, sheet1.ncols-1))

for i in range(sheet1.nrows-2):
  hsi_cube[i,:] = np.array(list(sheet1.row_values(i+2)[1:]))

hsi_cube_reshaped = hsi_cube.reshape((hsi_cube.shape[0],1,hsi_cube.shape[1]))
print(hsi_cube_reshaped.shape)

#fcls unmixing
fcls = amp.FCLS()
amap2 = fcls.map(hsi_cube_reshaped, endmem, normalize=False, mask=None)
print(amap2.shape)
print(amap2)
amap2=amap2.reshape((amap2.shape[0],amap2.shape[2]))

#write into a csv file
import pandas as pd

DF = pd.DataFrame(amap2)

DF.columns = ['soil', 'vegetation']
# save the dataframe as a csv file
DF.to_csv("data2.csv")

  hsi_cube[i,:] = np.array(list(sheet1.row_values(i+2)[1:]))
