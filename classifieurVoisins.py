import numpy as np
from sklearn.neighbors.nearest_centroid import NearestCentroid

class ClassifieurVoisins :

	def __init__(self, train_data, test_data) :
		self.td = train_data
		self.test = test_data
		with open('Resultats_Voisins.txt', 'w+') as fichier :
			print('Initialisation du classifieur Voisins', file=fichier)
			print(str(self.test['y'].ravel()[:70]) + ' ...', file=fichier)

	def tester(self) :

		clf = NearestCentroid()
		clf.fit(self.td['X'], self.td['y'].ravel())  
		predictions = clf.predict(self.test['X'])
		with open('Resultats_Voisins.txt', 'a+') as fichier :
			print('\nPredictions du classifieur Voisins', file=fichier)
			print(str(predictions[:70]) + ' ...', file=fichier)
			print(str(np.where(self.test['y'].ravel()==predictions, 1, 0)[:70]) + ' ...', file=fichier)
			tauxExact = np.mean(np.where(self.test['y'].ravel()==predictions, 1, 0))
			print('tauxExact = ' + str(tauxExact), file=fichier)