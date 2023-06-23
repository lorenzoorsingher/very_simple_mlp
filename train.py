# import the necessary packages
from model_class import mlp
from model_class.prepare_dataset import load_data, polish_data, split_train_test
from torch.optim import SGD
import torch.nn as nn
import torch
import pdb
import numpy as np
from random import randint
import cv2 as cv
def next_batch(inputs, targets, batchSize):
	# loop over the dataset
	for i in range(0, inputs.shape[0], batchSize):
		# yield a tuple of the current batched data and labels
		yield (inputs[i:i + batchSize], targets[i:i + batchSize])

# specify our batch size, number of epochs, and learning rate
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-4
# determine the device we will be using for training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] training using {}...".format(DEVICE))

data_path = "dataset/diabetes_prediction_dataset.csv"

raw_data = load_data(data_path)
X,y = polish_data(raw_data)
trainX, testX, trainY, testY = split_train_test(X,y,test_size=0.15)
trainX = torch.from_numpy(trainX).float()
testX = torch.from_numpy(testX).float()
trainY = torch.from_numpy(trainY).float()
testY = torch.from_numpy(testY).float()
# initialize our model and display its architecture
mlp = mlp.get_training_model().to(DEVICE)
print(mlp)
# initialize optimizer and loss function
opt = SGD(mlp.parameters(), lr=LR)
lossFunc = nn.HingeEmbeddingLoss()

from math import isnan
# create a template to summarize current training progress
trainTemplate = "epoch: {} test loss: {:.3f} test accuracy: {:.3f}"
# loop through the epochs
for epoch in range(0, EPOCHS):
	# initialize tracker variables and set our model to trainable
	print("[INFO] epoch: {}...".format(epoch + 1))
	trainLoss = 0
	trainAcc = 0
	samples = 0
	mlp.train()
	# loop over the current batch of data
	for (batchX, batchY) in next_batch(trainX, trainY, BATCH_SIZE):
		# flash data to the current device, run it through our
		# model, and calculate loss
		(batchX, batchY) = (batchX.to(DEVICE), batchY.to(DEVICE))
		predictions = mlp(batchX)
		#breakpoint()
		loss = lossFunc(predictions, torch.tensor(np.expand_dims(batchY.long(),axis=1)).float())
		
		# zero the gradients accumulated from the previous steps,
		# perform backpropagation, and update model parameters
		opt.zero_grad()
		loss.backward()
		opt.step()
		# update training loss, accuracy, and the number of samples
		# visited
		trainLoss += loss.item() * batchY.size(0)
		if isnan(trainLoss):
			print("IS NAN")
			breakpoint()
		#print(trainLoss)
		trainAcc += (predictions.max(1)[1] == batchY).sum().item()
		samples += batchY.size(0)
	#breakpoint()
	# display model progress on the current training batch
	trainTemplate = "epoch: {} train loss: {:.3f} train accuracy: {:.3f}"
	print(trainTemplate.format(epoch + 1, (trainLoss / samples),
		(trainAcc / samples)))
	

	# initialize tracker variables for testing, then set our model to
	# evaluation mode
	testLoss = 0
	testAcc = 0
	samples = 0
	mlp.eval()
	# initialize a no-gradient context
	with torch.no_grad():
		# loop over the current batch of test data
		for (batchX, batchY) in next_batch(testX, testY, BATCH_SIZE):
			# flash the data to the current device
			(batchX, batchY) = (batchX.to(DEVICE), batchY.to(DEVICE))
			# run data through our model and calculate loss
			predictions = mlp(batchX)
			loss = lossFunc(predictions, batchY.long())
			# update test loss, accuracy, and the number of
			# samples visited
			testLoss += loss.item() * batchY.size(0)
			testAcc += (predictions.max(1)[1] == batchY).sum().item()
			samples += batchY.size(0)
		# display model progress on the current test batch
		testTemplate = "epoch: {} test loss: {:.3f} test accuracy: {:.3f}"
		print(testTemplate.format(epoch + 1, (testLoss / samples),
			(testAcc / samples)))
		print("")

print("SINGLE TRY")
################################
raw_data= load_data(data_path)
X,y = polish_data(raw_data)
while True:
	
	X = torch.from_numpy(np.asarray(X,dtype=np.float32)).float()
	y =torch.from_numpy(np.asarray(y,dtype=np.float32)).float()

	idx = randint(0,4000)

	(bX,bY)=(X[idx],y[idx])


	prediction = mlp(bX)
	print(prediction)
	if(y[idx] > 0.5):
		print("diabetic")
		breakpoint()
	else:
		print("not diabetic")

	

##############################