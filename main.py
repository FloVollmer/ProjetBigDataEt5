import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pretraitement import PreProc
from classifieur import Classifieur
from classifieurVecteur import ClassifieurVecteur
import sklearn.decomposition as SK

train_data = loadmat('train_32x32.mat')
test_data = loadmat('test_32x32.mat')

testSize = np.size(train_data['X'][0, 0, 0, :])
testSize =5000
classe1 = train_data['y'].squeeze(1)[:testSize]==2 # label a tester
train_data['X'] = train_data['X'][:, :, :, :testSize] # taille d'echantillon a tester


preProc = PreProc(train_data, testSize)
train_data_vecteur = preProc.pretraiteDatas()
classeEnsemble = train_data['y']


classifieur = Classifieur(train_data, testSize)
#imgTest = classifieur.barycentreX(2)
#classifieur.distanceEntre2images(imgTest.astype("int"), train_data['X'][:, :, :, 3])
classifieur.tester()
# plt.imshow(train_data['X'][:, :, :, 3])
#print('Label', train_data['y'][3])
# plt.show()
