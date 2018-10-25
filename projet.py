import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
#wher


def negatif (couche) :
	return 255 - couche
	
def pretraitement (image_idx) :

	minCouche = [255, 255, 255]
	maxCouche = [0, 0, 0]
	
	# On trouve les maximum et minimum de chaque couche
	for j in range(32) :
		for i in range(32) :
			for k in range(3) :
				if train_data['X'][j, i, k, image_idx] < minCouche[k] :
					minCouche[k] = train_data['X'][j, i, k, image_idx]
				if train_data['X'][j, i, k, image_idx] > maxCouche[k] :
					maxCouche[k] = train_data['X'][j, i, k, image_idx]
					
	# On etend le spectre des couleurs et on fait une isohelie
	for j in range(32) :
		for i in range(32) :
			for k in range(3) :
				train_data['X'][j, i, k, image_idx] -= minCouche[k]
				train_data['X'][j, i, k, image_idx] *= 255 / (maxCouche[k] - minCouche[k])
				if train_data['X'][j, i, k, image_idx] > 127 :
					train_data['X'][j, i, k, image_idx] = 255
				else :
					train_data['X'][j, i, k, image_idx] = 0
	
	# On regarde pour chaque couche si les bordures sont plutot claires ou foncees
	sommeBordureCouche = [0, 0, 0]
	for i in range(32) :
		for k in range(3) :
			if train_data['X'][0, i, k, image_idx] > 127 : 
				sommeBordureCouche[k] += 1
			if train_data['X'][31, i, k, image_idx] > 127 : 
				sommeBordureCouche[k] += 1
			if train_data['X'][i, 0, k, image_idx] > 127 : 
				sommeBordureCouche[k] += 1
			if train_data['X'][i, 31, k, image_idx] > 127 : 
				sommeBordureCouche[k] += 1
				
	# Si elles sont foncees on fait le negatif de cette couche car on veut des nombres en noir sur blanc
	for k in range(3) :	
		if sommeBordureCouche[k] < 32*4/2 :
			for j in range(32) :
				for i in range(32) :
					train_data['X'][j, i, k, image_idx] = negatif(train_data['X'][j, i, k, image_idx])
			
	print('Label', train_data['y'][image_idx])
	
	barycentre = [0, 0, 0, 0]
	for j in range(32) :
		for i in range(32) :
			for k in range(3) :
	
	plt.imshow(train_data['X'][:, :, :, image_idx])
	plt.show()


train_data = loadmat('train_32x32.mat')
test_data = loadmat('test_32x32.mat')

pretraitement(20);