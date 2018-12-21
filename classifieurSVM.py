import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix

class ClassifieurSVM :

	def __init__(self, train_data, test_data, trainSize, testSize) :
		self.td = train_data
		self.test = test_data
		self.trainSize = trainSize
		self.testSize = testSize
		self.meilleurC = -1
		with open('Resultats_SVM.txt', 'w+') as fichier :
			print('Initialisation du classifieur SVM', file=fichier)
			print(str(self.test['y'].ravel()[:70]) + ' ...', file=fichier)

	def calibrer(self) :
		if self.trainSize <= 2000 :
			miniN = int(self.trainSize/2)
		else :
			miniN = 1000
		miniTdX = self.td['X'][:miniN];
		miniTdy = self.td['y'][:miniN];
		miniTestX = self.td['X'][-miniN:];
		miniTesty = self.td['y'][-miniN:];
		
		with open('Resultats_SVM.txt', 'a+') as fichier :
			print('\nCalibrage du classifieur SVM', file=fichier)
		
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

		with open("Resultats_SVM.txt", "a+") as fichier :
			print('meilleurTaux intermediaire = ' + str(meilleurTaux), file=fichier)
			print('meilleurC intermediaire = ' + str(self.meilleurC), file=fichier)
			print('meilleurGamma intermediaire = ' + str(self.meilleurGamma), file=fichier)


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

		with open('Resultats_SVM.txt', 'a+') as fichier :
			print('meilleurTaux = ' + str(meilleurTaux), file=fichier)
			print('meilleurC = ' + str(self.meilleurC), file=fichier)
			print('meilleurGamma = ' + str(self.meilleurGamma), file=fichier)


	def tester(self) :
		clf = svm.SVC(gamma=self.meilleurGamma, C=self.meilleurC)
		clf.fit(self.td['X'], self.td['y'].ravel())  
		predictions = clf.predict(self.test['X'])
		with open('Resultats_SVM.txt', 'a+') as fichier :
			print('\nPredictions du classifieur SVM', file=fichier)
			print(str(predictions[:70]) + ' ...', file=fichier)
			tauxExact = np.mean(np.where(self.test['y'].ravel()==predictions, 1, 0))
			print('tauxExact = ' + str(tauxExact), file=fichier)
			print('Matrice de confusion :' + str(tauxExact), file=fichier)
			print(confusion_matrix(self.test['y'].ravel(), predictions), file=fichier)