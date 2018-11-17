import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
#wher


def negatif (couche) :
	return 255 - couche
	
def isohelie (img_id) :

	minCouche = [255, 255, 255]
	maxCouche = [0, 0, 0]
	
	# On trouve les maximum et minimum de chaque couche
	for j in range(32) :
		for i in range(32) :
			for k in range(3) :
				if train_data['X'][j, i, k, img_id] < minCouche[k] :
					minCouche[k] = train_data['X'][j, i, k, img_id]
				if train_data['X'][j, i, k, img_id] > maxCouche[k] :
					maxCouche[k] = train_data['X'][j, i, k, img_id]
					
	# On etend le spectre des couleurs et on fait une isohelie
	for j in range(32) :
		for i in range(32) :
			for k in range(3) :
				train_data['X'][j, i, k, img_id] -= minCouche[k]
				train_data['X'][j, i, k, img_id] *= 255 / (maxCouche[k] - minCouche[k])
				if train_data['X'][j, i, k, img_id] > 127 :
					train_data['X'][j, i, k, img_id] = 255
				else :
					train_data['X'][j, i, k, img_id] = 0
	

	
def noirSurBlanc (img_id) :	
	
	# On regarde pour chaque couche si les bordures sont plutot claires ou foncees
	borduresClaires = [0, 0, 0]
	for k in range(3) :
		for i in range(32) :
			borduresClaires[k] += train_data['X'][0, i, k, img_id]
			borduresClaires[k] += train_data['X'][31, i, k, img_id]
			borduresClaires[k] += train_data['X'][i, 0, k, img_id]
			borduresClaires[k] += train_data['X'][i, 31, k, img_id]
		#print("borduresClaires[" + str(k) + "] = " + str(borduresClaires[k]))
		for j in range(8) :
			for i in range(8) :
				#train_data['X'][j+12, i+12, k, img_id] = k*80
				borduresClaires[k] -= train_data['X'][j+12, i+12, k, img_id]*2;
		#print("borduresClaires[" + str(k) + "] = " + str(borduresClaires[k]))
	
	# Si elles sont foncees on fait le negatif de cette couche car on veut des nombres en noir sur blanc
	for k in range(3) :	
		if borduresClaires[k] < 0 :
			for j in range(32) :
				for i in range(32) :
					train_data['X'][j, i, k, img_id] = negatif(train_data['X'][j, i, k, img_id])
					
					
	
def pretraiteImg (img_id) :
	
	#isohelie(img_id)
	
	noirSurBlanc(img_id)
	
	
	
def pretraitement(testSize):
	for i in range(testSize) :
		pretraiteImg(i);
	

	
train_data = loadmat('train_32x32.mat')
test_data = loadmat('test_32x32.mat')

testSize = 50
classe1 = train_data['y'].squeeze(1)[:testSize]==2 # label a tester
train_data['X'] = train_data['X'][:, :, :, :testSize] # taille d'echantillon a tester

pretraitement(testSize)

#print(classe1.shape)
#print(train_data['X'].shape)
imgTest = np.mean(train_data['X'][:, :, :, classe1], axis=3)
#plt.imshow(imgTest)
plt.imshow(train_data['X'][:, :, :, 2])
#print('Label', train_data['y'][3])
plt.show()
