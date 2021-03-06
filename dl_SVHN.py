import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.io import loadmat
import matplotlib.pyplot as plt
torch.manual_seed(0)

class CNN(nn.Module):

	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = nn.Conv2d(3, 20, 5)
		self.conv2 = nn.Conv2d(20, 50, 5)
		self.fc1 = nn.Linear(1250, 500)
		self.fc2 = nn.Linear(500, 10)

	def forward(self, x):
		x = F.relu(F.max_pool2d(self.conv1(x), 2))
		x = F.relu(F.max_pool2d(self.conv2(x), 2))
		x = x.view(x.shape[0], -1) # Flatten the tensor
		x = F.relu(self.fc1(x))
		x = F.log_softmax(self.fc2(x), dim=1)
		return x
		
def precision(valReelles, predictions) :
	taux = 0
	for i in range(len(predictions)) :
		if predictions[i] == valReelles[i] :
			taux += 1
	return taux/len(predictions)

if __name__ == '__main__':

	# Load the dataset
	train_data = loadmat('train_32x32.mat')
	test_data = loadmat('test_32x32.mat')

	train_label = train_data['y'][:100]
	train_label = np.where(train_label==10, 0, train_label)
	train_label = torch.from_numpy(train_label.astype('int')).squeeze(1)
	train_label = train_label.type(torch.LongTensor)
	train_data = torch.from_numpy(train_data['X'].astype('float32')).permute(3, 2, 0, 1)[:100]

	test_label = test_data['y'][:1000]
	test_label = np.where(test_label==10, 0, test_label)
	test_label = torch.from_numpy(test_label.astype('int')).squeeze(1)
	test_label = test_label.type(torch.LongTensor)
	test_data = torch.from_numpy(test_data['X'].astype('float32')).permute(3, 2, 0, 1)[:1000]

	# Hyperparameters
	epoch_nbr = 10
	batch_size = 5
	learning_rate = 1e-4
	
	axeX = np.arange(0, epoch_nbr, 1)
	axeY = []

	net = CNN()
	optimizer = optim.SGD(net.parameters(), lr=learning_rate)
	for e in range(epoch_nbr):
		print("Epoch", e)
		tauxEpoch = 0.0
		for i in range(0, train_data.shape[0], batch_size):
			optimizer.zero_grad() # Reset all gradients to 0
			predictions_train = net(train_data[i:i+batch_size])
			_, class_predicted = torch.max(predictions_train, 1)
			loss = F.nll_loss(predictions_train, train_label[i:i+batch_size])
			loss.backward()
			optimizer.step() # Perform the weights update
			tauxExact = precision(class_predicted, train_label[i:i+batch_size])
			# print('precision = ' + str(tauxExact) + ' %')
			tauxEpoch += tauxExact;
		tauxEpoch /= (train_data.shape[0]/batch_size);
		print('tauxEpoch = ' + str(tauxEpoch))
		axeY.append(tauxEpoch)
			
	plt.plot(axeX, axeY)
	plt.gca().set_ylim([0,1])
	plt.gca().set_xlim([0,epoch_nbr-1])
	plt.show()
