import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from pretraitement import PreProc
from classifieur import Classifieur
from classifieurSVM import ClassifieurSVM
from classifieurVoisins import ClassifieurVoisins
import sklearn.decomposition as SK

def toutTester(trainSize, testSize, valPCA) :

	# Lecture et pretraitement des donnees
	train_data = loadmat('train_32x32.mat')
	test_data = loadmat('test_32x32.mat')
	train_data['X'] = train_data['X'][:, :, :, :trainSize]
	train_data['y'] = train_data['y'][:trainSize]
	test_data['X'] = test_data['X'][:, :, :, :testSize]
	test_data['y'] = test_data['y'][:testSize]

	preProc = PreProc(train_data, test_data, trainSize, testSize)
	preProc.pretraiteDatas(20)
	
	# test du classieur proches voisins
	classifieurVoisins = ClassifieurVoisins(train_data, test_data)
	classifieurVoisins.tester()

	# test du classieur SVM
	classifieurSVM = ClassifieurSVM(train_data, test_data, trainSize, testSize)
	classifieurSVM.calibrer()
	classifieurSVM.tester()
	
	# test de notre classifieur a distances
	classifieur = Classifieur(train_data, test_data, trainSize, testSize)
	classifieur.tester()

	print('yes')


toutTester(2000, 1000, 20)

# plt.imshow(train_data['X'][:, :, :, 3])
# print('Label', train_data['y'][3])
# plt.show()