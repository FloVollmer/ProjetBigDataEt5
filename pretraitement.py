import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import sklearn.decomposition as SK

class PreProc :
	
	def __init__(self, train_data, test_data, testSize) :
		self.td = train_data
		self.test = test_data
		self.testSize = testSize
		
	def applatirDonnees(self) :
	
		print(self.td['X'].shape)
		self.td['X'] = np.transpose(self.td['X'], (3, 0, 1, 2))
		self.td['X'] = np.reshape(self.td['X'], (self.td['X'].shape[0], 32*32*3))
		print(self.td['X'].shape)
		
		print(self.test['X'].shape)
		self.test['X'] = np.transpose(self.test['X'], (3, 0, 1, 2))
		self.test['X'] = np.reshape(self.test['X'], (self.test['X'].shape[0], 32*32*3))
		print(self.test['X'].shape)
	
		
	def PCAVecteur(self, taille_pca) :
		PCA = SK.PCA(n_components=taille_pca)
		result = PCA.fit_transform(self.td['X'])
		print(result.shape)
		self.td['X'] = result
		result = PCA.fit_transform(self.test['X'])
		print(result.shape)
		self.test['X'] = result
	
	def negatif (self, couche) :
		return 255 - couche
	
	def isohelie (self, img_id, img_set) :

		minCouche = [255, 255, 255]
		maxCouche = [0, 0, 0]
		
		# On trouve les maximum et minimum de chaque couche
		for j in range(32) :
			for i in range(32) :
				for k in range(3) :
					if img_set[j, i, k, img_id] < minCouche[k] :
						minCouche[k] = img_set[j, i, k, img_id]
					if img_set[j, i, k, img_id] > maxCouche[k] :
						maxCouche[k] = img_set[j, i, k, img_id]
						
		# On etend le spectre des couleurs et on fait une isohelie
		for j in range(32) :
			for i in range(32) :
				for k in range(3) :
					img_set[j, i, k, img_id] -= minCouche[k]
					img_set[j, i, k, img_id] *= 255 / (maxCouche[k] - minCouche[k])
					if img_set[j, i, k, img_id] > 127 :
						img_set[j, i, k, img_id] = 255
					else :
						img_set[j, i, k, img_id] = 0
		

	def noirSurBlanc (self, img_id, img_set) :	
		
		# On regarde pour chaque couche si les bordures sont plutot claires ou foncees
		borduresClaires = [0, 0, 0]
		for k in range(3) :
			for i in range(32) :
				borduresClaires[k] += img_set[0, i, k, img_id]
				borduresClaires[k] += img_set[31, i, k, img_id]
				borduresClaires[k] += img_set[i, 0, k, img_id]
				borduresClaires[k] += img_set[i, 31, k, img_id]
			#print("borduresClaires[" + str(k) + "] = " + str(borduresClaires[k]))
			for j in range(8) :
				for i in range(8) :
					borduresClaires[k] -= img_set[j+12, i+12, k, img_id]*2;
			#print("borduresClaires[" + str(k) + "] = " + str(borduresClaires[k]))
		
		# Si elles sont foncees on fait le negatif de cette couche car on veut des nombres en noir sur blanc
		for k in range(3) :	
			if borduresClaires[k] < 0 :
				for j in range(32) :
					for i in range(32) :
						img_set[j, i, k, img_id] = self.negatif(img_set[j, i, k, img_id])
	
	def noirSurBlanc2 (self, img_id, img_set) :	
		
		# On regarde pour chaque couche si les bordures sont plutot claires ou foncees
		comptePixelBlanc = 0
		comptePixelNoir = 0
		for k in range(3) :
			for i in range(32) :
				for j in range(32) : 
					if img_set[i, j, k, img_id] >= 255 :
						comptePixelBlanc += 1
					else :
						comptePixelNoir += 1
			if comptePixelNoir > comptePixelBlanc :
				for i in range(32) :
					for j in range(32) : 
						img_set[i, j, k, img_id] = self.negatif(img_set[i, j, k, img_id])
			comptePixelBlanc = 0
			comptePixelNoir = 0
	
	def noirSurBlanc3 (self, img_id, img_set) :	
		
		# On regarde pour chaque couche si les bordures sont plutot claires ou foncees
		comptePixelBlanc = 0
		comptePixelNoir = 0
		for k in range(3) :
			for i in range(32) :
				for j in range(32) : 
					if img_set[i, j, k, img_id] >= 255 :
						comptePixelBlanc += 1
					else :
						comptePixelNoir += 1
			if comptePixelNoir > comptePixelBlanc :
				for i in range(32) :
					for j in range(32) : 
						if img_set[i, j, k, img_id] <= 0 :
							img_set[i, j, k, img_id] = 255
						elif img_set[i, j, k, img_id] >= 255 :
							img_set[i, j, k, img_id] = 0
			comptePixelBlanc = 0
			comptePixelNoir = 0
							
							
		
	def pretraiteImg (self, img_id, img_set) :
		self.isohelie(img_id, img_set)
		self.noirSurBlanc(img_id, img_set)
		
	
	def pretraiteDatas (self) :
		for i in range(self.testSize) :
			self.pretraiteImg(i, self.td['X']);
			self.pretraiteImg(i, self.test['X']);
		self.applatirDonnees();
		self.PCAVecteur(20)
		

	

