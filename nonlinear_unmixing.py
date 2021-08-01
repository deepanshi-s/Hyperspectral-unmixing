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
import cv2
from sklearn.metrics import f1_score
import sklearn.metrics
from spectral import *
import spectral.io.envi as envi

#load the envi files
img = envi.open('samson.hdr', 'samson_1.img')
print(img)
arr = img.load()
print(arr.__class__)
arr = np.array(arr)

#endmember extraction using NFINDR algorithm
end_member = eea.NFINDR()
U = end_member.extract(arr,3 ,normalize=False, mask=None)
print('  End members indexes:', end_member.get_idx())
end_member.display()

#reshaping the endmember array to account for the ground truth array(i.e. the order of the endmembers in ground truth is different from what si returned by the function
U1 = np.zeros((3,156))
U1[0] = U[1]
U1[1] = U[0]
U1[2] = U[2]

#FCLS based abundance estimation
fcls = amp.FCLS()
amap2 = fcls.map(arr, U1, normalize=False, mask=None)
print(amap2.shape)
fcls.display(mask=None, interpolation='none', colorMap='jet', columns=10, suffix=None)
U1 = amap2

#rehaping and resizing the input image
U1 = U1.reshape((95,95,3))
flip = cv2.flip(U1,0)
flip = cv2.rotate(flip, cv2.ROTATE_90_CLOCKWISE)
U1  = flip.reshape((flip.shape[0]*flip.shape[1]), flip.shape[2])

#load the ground truth array
mat = spio.loadmat('end3.mat')
print(mat.keys())
gt = mat['A']
gt = gt.T
print("gt shpae-",gt.shape)

#one hot encoding the data
gt1 = np.zeros((9025,3))
for i in range(0,9025,1):
  x = np.argmax(gt[i,:])
  gt_score[i] = x+1
  gt1[i,x] =1

#train test split
X_train, X_test, Y_train, Y_test= train_test_split(U1, gt1, test_size=0.25, random_state=42)
print(X_test.shape)
print(X_train.shape)
print(Y_test.shape)
print(Y_train.shape)

#train the classifier
mlp = MLPClassifier()
model = mlp.fit(X_train,Y_train)

#prediction
predicted = model.predict(U1)
print("predicted array shape-",predicted.shape)

#reshaping and post processing the output for plotting
predicted1 = predicted.reshape((95,95,3))
img = np.zeros((95,95,3))
for i in range(0,95,1):
  for j in range(0,95,1):
    if(predicted1[i,j,0] == 1):
      img[i,j,:] =  [255,0,0]
    if(predicted1[i,j,1] == 1):
      img[i,j,:] = [0,255,0]
    if(predicted1[i,j,2] == 1):
      img[i,j,:] = [0,0,255]

plt.imshow(img)
plt.show()

gtx = gt.reshape((95,95,3))
plt.imshow(gtx)
plt.show()

predicted_score = np.zeros((9025,1))

for i in range(0,9025,1):
  if(predicted[i,0] == 1):
    predicted_score[i] = 1

  elif(predicted[i,1] == 1):
    predicted_score[i] = 2

  elif(predicted[i,2] == 1):
    predicted_score[i] = 3

  else:
    predicted_score[i] =0

print('f1.score-',sklearn.metrics.f1_score(gt_score, predicted_score, average='weighted'))
print('jaccard score-',sklearn.metrics.jaccard_score(gt_score, predicted_score, average='weighted'))

