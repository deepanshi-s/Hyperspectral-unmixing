#reading endmembers from text files and generating abundance maps


import scipy.io as spio
import pysptools.eea as eea
import pysptools.abundance_maps as amp
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
from spectral import *


import spectral.io.envi as envi
img = envi.open('subset_045728.hdr', 'subset_045728')

arr = img.load()
arr = np.array(arr)



#read endmember 1 from txt file 
file1 = open('residue_em.txt', 'r') 
Lines = file1.readlines() 
import numpy as np
count = 0
# Strips the newline character 
i=0
a1 = np.zeros((2151))
for i in range(2154): 
  line = Lines[i]
  if(i>2):
    line = line.split("\t")
    line = line[-1]
    line = line[:-1]
    line = float(line)
    a1[i-3] = line
file1.close

#read endmember 2
file1 = open('soil_em.txt', 'r') 
Lines = file1.readlines() 
import numpy as np
count = 0
# Strips the newline character 
i=0
a2 = np.zeros((2151))
for i in range(2154): 
  line = Lines[i]
  if(i>2):
    line = line.split("\t")
    line = line[-1]
    line = line[:-1]
    line = float(line)
    a2[i-3] = line
file1.close



#to match the wavelengths from the excel file and txt file of endmembers
import xlrd
import numpy as np

# Give the location of the file
loc = ("AVIRISNG.xlsx")
 
# To open Workbook
wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)
band = np.zeros((425))
# For row 0 and column 0
for i in range(1,sheet.nrows):
  band[i-1] = float(sheet.cell_value(i,1))

#map the wavelengths from excel sheet and txt file
residue = np.zeros((425))
for i in range(425):
  residue[i] = a1[int(band[i]-350)]

soil = np.zeros((425))
for i in range(425):
  soil[i] = a2[int(band[i]-350)]

#concatenate the endmembers 
residue = residue.reshape((1,425))
soil = soil.reshape((1,425))
M = np.concatenate((residue,soil))
print(M.shape)


#remove noisy bands 
m1 = M[:,:195]
m2 = M[:,212:281]
m3 = M[:,316:]
print(m1.shape,m2.shape,m3.shape)
h = np.hstack((m1,m2,m3))
h = h[:,1:]
print(h.shape)


