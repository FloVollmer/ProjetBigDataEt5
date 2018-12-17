import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pretraitement import PreProc
from classifieur import Classifieur
import sklearn.decomposition as SK
from sklearn import svm

train_data = loadmat('train_32x32.mat')
test_data = loadmat('test_32x32.mat')

testSize = np.size(train_data['X'][0, 0, 0, :])
testSize = 300
train_data['X'] = train_data['X'][:, :, :, :testSize] # taille d'echantillon a tester
train_data['y'] = train_data['y'][:testSize] # taille d'echantillon a tester
test_data['X'] = test_data['X'][:, :, :, :testSize] # taille d'echantillon a tester
test_data['y'] = test_data['y'][:testSize] # taille d'echantillon a tester


preProc = PreProc(train_data, test_data, testSize)
preProc.pretraiteDatas()

#X = [[0, 0], [1, 1], [2, 2]]
#y = [0, 1, 'bonjour']
clf = svm.SVC(gamma='scale', decision_function_shape='ovr')
clf.fit(train_data['X'], train_data['y'].ravel())  
predictions = clf.predict(test_data['X'])
print(str(predictions))
print(str(test_data['y'].ravel()))
print(np.where(test_data['y'].ravel()==predictions, 1, 0)) # 1 = prediction juste

#classifieur = Classifieur(train_data, test_data, testSize)
#classifieur.tester()

# plt.imshow(train_data['X'][:, :, :, 3])
#print('Label', train_data['y'][3])
# plt.show()
