import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.misc
from PIL import Image

class Classifieur :

	barycentres = []

	def __init__(self, train_data, test_data, testSize) :
		self.td = train_data
		self.test = test_data
		self.testSize = testSize
		for i in range(1, 11) :
			self.barycentres.append(self.barycentreX(i))
		self.barycentres = np.reshape(self.barycentres, (10, len(self.barycentres[1])))
			
	def tester(self) :
		tauxExact = 0
		for i in range(0, self.testSize) :
			labelDeter = self.determinerLabel(self.td['X'][i, :])
			# print(str(self.td['y'][i]) + ' -> ' + str(labelDeter))
			if (labelDeter == self.td['y'][i]) :
				tauxExact += 1
			
		tauxExact /= self.testSize;
		print('tauxExact = ' + str(tauxExact))
	
	def determinerLabel(self, image) :
		dMin = self.distanceEntre2vecteurs(self.barycentres[0], image)
		labelMin = 1
		
		for i in range (1, 10) :
			dBary = self.distanceEntre2vecteurs(self.barycentres[i], image)
			if dBary < dMin :
				dMin = dBary
				labelMin = i+1
				
		return labelMin
		

	def distanceEntre2images(self, img1, img2) :
		distance = 0.0
		for k in range(3) :
			for i in range(32) :
				for j in range(32) :
					distance += (img1[i,j,k]-img2[i,j,k])*(img1[i,j,k]-img2[i,j,k])
		distance = np.sqrt(distance)
		#print(distance)
		return distance
		
	def distanceEntre2vecteurs(self, vect1, vect2) :
		distance = 0.0
		for i in range(vect1.size) :
			distance += (vect1[i]-vect2[i])*(vect1[i]-vect2[i])
		distance = np.sqrt(distance)
		return distance
		
	def barycentreX(self, label) :
			tableBool = self.td['y'].squeeze(1)[:self.testSize]==label
			barycentre = np.mean(self.td['X'][tableBool, :], axis=0)
			return barycentre
		
	# def barycentreX(self, label) :
		# try :
			# img = Image.open('barycentre' + str(label) + '.png')
		# except FileNotFoundError :
			# tableBool = self.td['y'].squeeze(1)[:self.testSize]==label
			# barycentre = np.mean(self.td['X'][:, :, :, tableBool], axis=3)
			# scipy.misc.imsave('barycentre' + str(label) + '.png', barycentre)
			# return barycentre
		# else :
			# return np.array(img)	
		