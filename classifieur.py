import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

class Classifieur :

	def distanceEntre2images(self, imageBary, imageCible):
		distanceParcouru = 0
		for k in range(3) :
			for i in range(32) :
				for j in range(32) : 
					distanceParcouru += (imageBary[i,j,k]-imageCible[i,j,k])*(imageBary[i,j,k]-imageCible[i,j,k])
		distanceParcouru = np.sqrt(distanceParcouru)
		print(distanceParcouru)
		
	def barycentreX(self, train_data, label) : 
		testSize = 150
		tableBool = train_data['y'].squeeze(1)[:testSize]==label
		barycentre = np.mean(train_data['X'][:, :, :, tableBool], axis=3)
		return barycentre