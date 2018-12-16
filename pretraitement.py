import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import sklearn.decomposition as SK

class PreProc :
	
	def __init__(self, train_data, testSize) :
		self.td = train_data
		self.testSize = testSize
		
	def PCAVecteur(self, train_data, taille_pca) :
		print(train_data['X'].shape)
		temp = train_data['X']
		temp = np.transpose(temp, (3, 0, 1, 2))
		print(temp.shape)
		size = temp.shape[0]
		temp = np.reshape(temp, (size, 32*32*3))
		print(temp.shape)
		PCA = SK.PCA(n_components=taille_pca)
		result = PCA.fit_transform(temp)
		print(result.shape)
		return result
	
	def negatif (self, couche) :
		return 255 - couche
	
	def isohelie (self, img_id) :

		minCouche = [255, 255, 255]
		maxCouche = [0, 0, 0]
		
		# On trouve les maximum et minimum de chaque couche
		for j in range(32) :
			for i in range(32) :
				for k in range(3) :
					if self.td['X'][j, i, k, img_id] < minCouche[k] :
						minCouche[k] = self.td['X'][j, i, k, img_id]
					if self.td['X'][j, i, k, img_id] > maxCouche[k] :
						maxCouche[k] = self.td['X'][j, i, k, img_id]
						
		# On etend le spectre des couleurs et on fait une isohelie
		for j in range(32) :
			for i in range(32) :
				for k in range(3) :
					self.td['X'][j, i, k, img_id] -= minCouche[k]
					self.td['X'][j, i, k, img_id] *= 255 / (maxCouche[k] - minCouche[k])
					if self.td['X'][j, i, k, img_id] > 127 :
						self.td['X'][j, i, k, img_id] = 255
					else :
						self.td['X'][j, i, k, img_id] = 0
		

	def noirSurBlanc (self, img_id) :	
		
		# On regarde pour chaque couche si les bordures sont plutot claires ou foncees
		borduresClaires = [0, 0, 0]
		for k in range(3) :
			for i in range(32) :
				borduresClaires[k] += self.td['X'][0, i, k, img_id]
				borduresClaires[k] += self.td['X'][31, i, k, img_id]
				borduresClaires[k] += self.td['X'][i, 0, k, img_id]
				borduresClaires[k] += self.td['X'][i, 31, k, img_id]
			#print("borduresClaires[" + str(k) + "] = " + str(borduresClaires[k]))
			for j in range(8) :
				for i in range(8) :
					#self.td['X'][j+12, i+12, k, img_id] = k*80
					borduresClaires[k] -= self.td['X'][j+12, i+12, k, img_id]*2;
			#print("borduresClaires[" + str(k) + "] = " + str(borduresClaires[k]))
		
		# Si elles sont foncees on fait le negatif de cette couche car on veut des nombres en noir sur blanc
		for k in range(3) :	
			if borduresClaires[k] < 0 :
				for j in range(32) :
					for i in range(32) :
						self.td['X'][j, i, k, img_id] = self.negatif(self.td['X'][j, i, k, img_id])
	
	def noirSurBlanc2 (self, img_id) :	
		
		# On regarde pour chaque couche si les bordures sont plutot claires ou foncees
		comptePixelBlanc = 0
		comptePixelNoir = 0
		for k in range(3) :
			for i in range(32) :
				for j in range(32) : 
					if self.td['X'][i, j, k, img_id] >= 255 :
						comptePixelBlanc += 1
					else :
						comptePixelNoir += 1
			if comptePixelNoir > comptePixelBlanc :
				for i in range(32) :
					for j in range(32) : 
						self.td['X'][i, j, k, img_id] = self.negatif(self.td['X'][i, j, k, img_id])
			comptePixelBlanc = 0
			comptePixelNoir = 0
	
	def noirSurBlanc3 (self, img_id) :	
		
		# On regarde pour chaque couche si les bordures sont plutot claires ou foncees
		comptePixelBlanc = 0
		comptePixelNoir = 0
		for k in range(3) :
			for i in range(32) :
				for j in range(32) : 
					if self.td['X'][i, j, k, img_id] >= 255 :
						comptePixelBlanc += 1
					else :
						comptePixelNoir += 1
			if comptePixelNoir > comptePixelBlanc :
				for i in range(32) :
					for j in range(32) : 
						if self.td['X'][i, j, k, img_id] <= 0 :
							self.td['X'][i, j, k, img_id] = 255
						elif self.td['X'][i, j, k, img_id] >= 255 :
							self.td['X'][i, j, k, img_id] = 0
			comptePixelBlanc = 0
			comptePixelNoir = 0
							
							
		
	def pretraiteImg (self, img_id) :
		self.isohelie(img_id)
		self.noirSurBlanc(img_id)
		
	
	def pretraiteDatas (self) :
		for i in range(self.testSize) :
			self.pretraiteImg(i);
		return self.PCAVecteur(self.td, 6)
		

	

