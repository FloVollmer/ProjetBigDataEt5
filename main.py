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
preProc.pretraiteDatas();


#print(classe1.shape)
#print(train_data['X'].shape)
imgTest = np.mean(train_data['X'][:, :, :, classe1], axis=3)
#print(imgTest.shape)
classifieur = Classifieur()
classifieur.distanceEntre2images(imgTest.astype("int"), train_data['X'][:, :, :, 3])
#plt.imshow(imgTest.astype("int"))
plt.imshow(train_data['X'][:, :, :, 3])
#plt.savefig('foo.png')
#print('Label', train_data['y'][3])
plt.show()
