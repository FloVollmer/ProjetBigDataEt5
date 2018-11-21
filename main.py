import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pretraitement import PreProc
from classifieur import Classifieur

train_data = loadmat('train_32x32.mat')
test_data = loadmat('test_32x32.mat')

testSize = 150
classe1 = train_data['y'].squeeze(1)[:testSize]==2 # label a tester
train_data['X'] = train_data['X'][:, :, :, :testSize] # taille d'echantillon a tester


preProc = PreProc(train_data, testSize)
preProc.pretraiteDatas()



classifieur = Classifieur(train_data, testSize)
#imgTest = classifieur.barycentreX(2)
#classifieur.distanceEntre2images(imgTest.astype("int"), train_data['X'][:, :, :, 3])
classifieur.tester()

plt.imshow(train_data['X'][:, :, :, 3])
#print('Label', train_data['y'][3])
plt.show()
