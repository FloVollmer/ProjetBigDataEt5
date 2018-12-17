import numpy as np
from sklearn.neighbors.nearest_centroid import NearestCentroid

class ClassifieurVoisins :

	def __init__(self, train_data, test_data) :
		self.td = train_data
		self.test = test_data

	def tester(self) :

		clf = NearestCentroid()
		clf.fit(self.td['X'], self.td['y'].ravel())  
		predictions = clf.predict(self.test['X'])
		print(predictions)
		tauxExact = np.mean(np.where(self.test['y'].ravel()==predictions, 1, 0))
		print('tauxExact = ' + str(tauxExact))