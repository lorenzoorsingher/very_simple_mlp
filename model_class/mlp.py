# import the necessary packages
from collections import OrderedDict
import torch.nn as nn
def get_training_model(inFeatures=8, hiddenDim=16, nbClasses=1):
	# construct a shallow, sequential neural network
	mlpModel = nn.Sequential(OrderedDict([
		("hidden_layer_1", nn.Linear(inFeatures, hiddenDim)),
		("activation_1", nn.ReLU()),
		("hidden_layer_2", nn.Linear(hiddenDim, 32)),
		("activation_2", nn.ReLU()),
		("hidden_layer_3", nn.Linear(32, hiddenDim)),
		("activation_3", nn.ReLU()),
		("output_layer", nn.Linear(hiddenDim, nbClasses))
		#("map_layer", nn.Sigmoid())
	]))
	# return the sequential model
	return mlpModel