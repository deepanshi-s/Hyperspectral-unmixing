#writing endmember files

import scipy.io as spio
import pysptools.eea as eea
import pysptools.abundance_maps as amp
import matplotlib.pyplot as plt
import numpy as np
import cvxopt
import scipy.io as spio
from sklearn import *
from sklearn.neural_network import  *
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from spectral import *


import spectral.io.envi as envi
img = envi.open('subset_045728.hdr', 'subset_045728')

arr = img.load()
arr = np.array(arr)

#generate endmembers
ppi = eea.NFINDR()
U = ppi.extract(arr,1,normalize=True, mask=None)
print(U.shape)
print(str(ppi))
print('  End members indexes:', ppi.get_idx())
index = ppi.get_idx()
ppi.display()

#writing endmember files, to write the wavelengths replace random
f= open("em.txt","w+")
f.write("Endmember file \n")
f.write("Column 1: Wavelength \n")
f.write("Column 2: Soil \n")
for i in range(U.shape[1]):
     f.write('random   %f\r\n '%U[0,i])

f.close()
