import numpy as np
from sklearn import svm

class ClassifieurSVM :

	def __init__(self, train_data, test_data, testSize) :
		self.td = train_data
		self.test = test_data
		self.testSize = testSize
		self.meilleurC = -1

	def calibrer(self) :
		miniN = int(self.testSize/2)
		miniTdX = self.td['X'][:miniN];
		miniTdy = self.td['y'][:miniN];
		miniTestX = self.td['X'][-miniN:];
		miniTesty = self.td['y'][-miniN:];

		print(str(self.test['y'].ravel()))
		
		# On recherche les meilleures valeurs dans une grille logarithmique base 10
		C_range = np.logspace(-2, 10, 13)
		gamma_range = np.logspace(-9, 3, 13)
		self.meilleurC = -1
		self.meilleurGamma = -1
		meilleurTaux = 0
		for C in C_range:
			for gamma in gamma_range:
				#print('C = ' + str(C))
				#print('gamma = ' + str(gamma))
				clf = svm.SVC(gamma=gamma, C=C)
				clf.fit(miniTdX, miniTdy.ravel())  
				predictions = clf.predict(miniTestX)
				#print(predictions[:70])
				tauxExact = np.mean(np.where(miniTesty.ravel()==predictions, 1, 0))
				#print('tauxExact = ' + str(tauxExact))
				if tauxExact > meilleurTaux :
					meilleurTaux = tauxExact
					self.meilleurC = C
					self.meilleurGamma = gamma


		print('meilleurTaux intermediaire = ' + str(meilleurTaux))
		print('meilleurC intermediaire = ' + str(self.meilleurC))
		print('meilleurGamma intermediaire = ' + str(self.meilleurGamma))


		# Pour plus de precision, on cherche ensuite dans une grille logarithmique base 2
		C_range = np.logspace(-3, 3, base=2, num=13)*self.meilleurC
		gamma_range = np.logspace(-3, 3, base=2, num=13)*self.meilleurGamma
		self.meilleurC = -1
		self.meilleurGamma = -1
		meilleurTaux = 0
		for C in C_range:
			for gamma in gamma_range:
				#print('C = ' + str(C))
				#print('gamma = ' + str(gamma))
				clf = svm.SVC(gamma=gamma, C=C)
				clf.fit(miniTdX, miniTdy.ravel())  
				predictions = clf.predict(miniTestX)
				#print(predictions[:70])
				tauxExact = np.mean(np.where(miniTesty.ravel()==predictions, 1, 0))
				#print('tauxExact = ' + str(tauxExact))
				if tauxExact > meilleurTaux :
					meilleurTaux = tauxExact
					self.meilleurC = C
					self.meilleurGamma = gamma


		print('meilleurTaux = ' + str(meilleurTaux))
		print('meilleurC = ' + str(self.meilleurC))
		print('meilleurGamma = ' + str(self.meilleurGamma))


	def tester(self) :
		clf = svm.SVC(gamma=self.meilleurGamma, C=self.meilleurC)
		clf.fit(self.td['X'], self.td['y'].ravel())  
		predictions = clf.predict(self.test['X'])
		print(predictions)
		tauxExact = np.mean(np.where(self.test['y'].ravel()==predictions, 1, 0))
		print('tauxExact = ' + str(tauxExact))