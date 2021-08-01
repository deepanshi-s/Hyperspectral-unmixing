import scipy.io as spio
import pysptools.eea as eea
import pysptools.abundance_maps as amp
import matplotlib.pyplot as plt
import numpy as np


#input is a matlab file
mat = spio.loadmat('C:\\Users\\Deepanshi\\Documents\\IISc_internship\\dataset_indian_pines\\Indian_pines_corrected.mat', squeeze_me=True)       #spio.loadmat returns a dictionary type output                    
mat_gt = spio.loadmat('C:\\Users\\Deepanshi\\Documents\\IISc_internship\\dataset_indian_pines\\Indian_pines_gt.mat', squeeze_me=True)
x1 = mat['indian_pines_corrected']                               #x1 is of numpy array type with dimensions (145,145,200)
x2 = mat_gt['indian_pines_gt']
print(x2.shape)
print(x2[0])

'''
indexes = np.zeros((16,2))
plt.imshow(x2, cmap = 'gray')
plt.show()
for k in range (0,16,1):
    i = index[k]//145
    j = index[k]-145*i
    print(i,j)
    indexes[k] = i,j
print(indexes)
print("okay")
'''
#pixel purity index"
ppi = eea.PPI()
U = ppi.extract(x1, 16, numSkewers=10000, normalize=False, mask=None)                   #U is a numpy array with dimensions (16.200)
indexes =  ppi.get_idx()
print(indexes)
#ppi.display()
print("okay")
for i in indexes:
    print(i, x2[i])


nfindr = eea.NFINDR()
V = nfindr.extract(x1, 16, normalize =False, mask = None)
print(V.shape)
ind = nfindr.get_idx()
print('  End members indexes:', nfindr.get_idx())

for i in ind:
    print(i,x2[i])
#nfindr.plot('C:\\Users\\Deepanshi\\Documents\\IISc_internship', suffix='test1')
'''

ucls = amp.UCLS()
amap = ucls.map(x1, V, normalize = True, mask= None)
print(amap.shape)
ucls.display(mask=None, interpolation='none', colorMap='jet', columns=None, suffix=None)

nnls = amp.NNLS()
amap1 = nnls.map(x1, V, normalize = True, mask= None)
print(amap1.shape)
nnls.display(mask=None, interpolation='none', colorMap='jet', columns=6, suffix=None)

fcls = amp.FCLS()
amap2 = fcls.map(x1, V, normalize=False, mask=None)
print(amap2.shape)
fcls.display(mask=None, interpolation='none', colorMap='jet', columns=6, suffix=None)
'''

