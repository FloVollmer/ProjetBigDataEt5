import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

class ClassifieurVoisins :

	def __init__(self, train_data, test_data) :
		self.td = train_data
		self.test = test_data
		with open('Resultats_Voisins.txt', 'w+') as fichier :
			print('Initialisation du classifieur Voisins', file=fichier)
			print(str(self.test['y'].ravel()[:70]) + ' ...', file=fichier)

	def tester(self, nbVoisins) :

		clf = KNeighborsClassifier(n_neighbors=nbVoisins)
		clf.fit(self.td['X'], self.td['y'].ravel())  
		predictions = clf.predict(self.test['X'])
		with open('Resultats_Voisins.txt', 'a+') as fichier :
			print('\nPredictions du classifieur Voisins', file=fichier)
			print(str(predictions[:70]) + ' ...', file=fichier)
			tauxExact = np.mean(np.where(self.test['y'].ravel()==predictions, 1, 0))
			print('tauxExact = ' + str(tauxExact), file=fichier)
			print('Matrice de confusion :' + str(tauxExact), file=fichier)
			print(confusion_matrix(self.test['y'].ravel(), predictions), file=fichier)