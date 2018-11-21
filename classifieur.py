import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.misc
from PIL import Image

class Classifieur :

	barycentres = []

	def __init__(self, train_data, testSize) :
		self.td = train_data
		self.testSize = testSize
		for i in range(1, 11) :
			self.barycentreX(i)

	def distanceEntre2images(self, imageBary, imageCible):
		distanceParcouru = 0
		for k in range(3) :
			for i in range(32) :
				for j in range(32) : 
					distanceParcouru += (imageBary[i,j,k]-imageCible[i,j,k])*(imageBary[i,j,k]-imageCible[i,j,k])
		distanceParcouru = np.sqrt(distanceParcouru)
		print(distanceParcouru)
		
	def barycentreX(self, label) :
		
		try :
			img = Image.open('barycentre' + str(label) + '.png')
		except FileNotFoundError :
			tableBool = self.td['y'].squeeze(1)[:self.testSize]==label
			barycentre = np.mean(self.td['X'][:, :, :, tableBool], axis=3)
			scipy.misc.imsave('barycentre' + str(label) + '.png', barycentre)
			return barycentre
		else :
			return np.array(img)
		
		
		