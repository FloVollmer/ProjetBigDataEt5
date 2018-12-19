import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pretraitement import PreProc
from classifieur import Classifieur
from classifieurSVM import ClassifieurSVM
from classifieurVoisins import ClassifieurVoisins
import sklearn.decomposition as SK

train_data = loadmat('train_32x32.mat')
test_data = loadmat('test_32x32.mat')

testSize = np.size(train_data['X'][0, 0, 0, :])
testSize = 1000
train_data['X'] = train_data['X'][:, :, :, :testSize] # taille d'echantillon a tester
train_data['y'] = train_data['y'][:testSize] # taille d'echantillon a tester
test_data['X'] = test_data['X'][:, :, :, :testSize] # taille d'echantillon a tester
test_data['y'] = test_data['y'][:testSize] # taille d'echantillon a tester


preProc = PreProc(train_data, test_data, testSize)
preProc.pretraiteDatas()

#classifieurVoisins = ClassifieurVoisins(train_data, test_data)
#classifieurVoisins.tester()

classifieurSVM = ClassifieurSVM(train_data, test_data, testSize)
classifieurSVM.calibrer()
classifieurSVM.tester()

#classifieur = Classifieur(train_data, test_data, testSize)
#classifieur.tester()

# plt.imshow(train_data['X'][:, :, :, 3])
#print('Label', train_data['y'][3])
# plt.show()
