# import the necessary packages
from model_class import mlp
from model_class.prepare_dataset import load_data, polish_data, split_train_test, balance_data
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
EPOCHS = 200
LR = 1e-3
# determine the device we will be using for training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("[INFO] training using {}...".format(DEVICE))

data_path = "dataset/diabetes_prediction_dataset.csv"
raw_data = load_data(data_path)
ttdata = raw_data[:round(len(raw_data)-len(raw_data)*.10)]
valdata = raw_data[round(len(raw_data)-len(raw_data)*.10):]
X,y = polish_data(ttdata)
X,y = balance_data(X,y)
trainX, testX, trainY, testY = split_train_test(X,y,test_size=0.015)
trainX = torch.from_numpy(trainX).float()
testX = torch.from_numpy(testX).float()
trainY = torch.from_numpy(trainY).float()
testY = torch.from_numpy(testY).float()

Xv, yv = polish_data(valdata)
valX, _, valY, _ = split_train_test(Xv,yv,test_size=0)
valX = torch.from_numpy(valX).float()
valY = torch.from_numpy(valY).float()

#breakpoint()
# initialize our model and display its architecture
mlp = mlp.get_training_model().to(DEVICE)
print(mlp)
# initialize optimizer and loss function
opt = SGD(mlp.parameters(), lr=LR)
lossFunc = nn.BCEWithLogitsLoss(reduction="mean")


def get_info(predictions, batchY):
	truePos = (torch.logical_and((predictions-.5)>0, (batchY-.5) > 0)).sum().item()
	falseNeg = (torch.logical_and((predictions-.5)<0, (batchY-.5) > 0)).sum().item()
	totRetr = ((predictions-.5) > 0).sum().item()
	if totRetr == 0:
		precision = 0
	else:
		precision = (truePos/totRetr)* batchY.size(0)
	if (truePos+falseNeg) == 0:
		recall = 0
	else:
		recall = (truePos/(truePos+falseNeg))* batchY.size(0)
	
	accuracy = ((predictions-.5) * (batchY-.5) > 0).sum().item()

	return precision, recall, accuracy

from math import isnan
# create a template to summarize current training progress
trainTemplate = "epoch: {} test loss: {:.3f} test accuracy: {:.3f}"
# loop through the epochs
for epoch in range(0, EPOCHS):
	# initialize tracker variables and set our model to trainable
	print("[INFO] epoch: {}...".format(epoch + 1))
	trainLoss = 0
	trainAcc = 0
	trainPrecision=0
	trainRecall=0
	samples = 0
	
	mlp.train()
	# loop over the current batch of data
	for (batchX, batchY) in next_batch(trainX, trainY, BATCH_SIZE):
		# flash data to the current device, run it through our
		# model, and calculate loss
		(batchX, batchY) = (batchX.to(DEVICE), batchY.to(DEVICE))
		predictions = mlp(batchX)
		#print(predictions)
		loss = lossFunc(predictions, batchY)
		#breakpoint()
		# zero the gradients accumulated from the previous steps,
		# perform backpropagation, and update model parameters
		opt.zero_grad()
		loss.backward()
		opt.step()
		# update training loss, accuracy, and the number of samples
		# visited

		trainLoss += loss.item() * batchY.size(0)
		p,r,a = get_info(predictions,batchY)
		trainPrecision += p
		trainRecall += r
		trainAcc += a
		samples += batchY.size(0)

	#breakpoint()
	# display model progress on the current training batch
	trainTemplate = "epoch: {} train loss: {:.3f} train accuracy: {:.3f} train precision: {:.3f} train recall: {:.3f}"
	print(trainTemplate.format(epoch + 1, (trainLoss /samples ),
		(trainAcc / samples), (trainPrecision/samples), (trainRecall/samples)))
	

	# initialize tracker variables for testing, then set our model to
	# evaluation mode
	testLoss = 0
	testAcc = 0
	testPrecision=0
	testRecall=0
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
			loss = lossFunc(predictions, batchY)
			# update test loss, accuracy, and the number of
			# samples visited

			testLoss += loss.item() * batchY.size(0)
			p,r,a = get_info(predictions,batchY)
			testPrecision += p
			testRecall += r
			testAcc += a
			samples += batchY.size(0)

		
		#display model progress on the current test batch
		testTemplate = "epoch: {} test loss: {:.3f} test accuracy: {:.3f} test precision: {:.3f} test recall: {:.3f}"
		print(testTemplate.format(epoch + 1, (testLoss / samples),
			(testAcc / samples), (testPrecision/samples), (testRecall/samples)))
		print("")

print("[VALIDATION]")

with torch.no_grad():
	# flash the data to the current device
	(batchX, batchY) = (valX.to(DEVICE), valY.to(DEVICE))
	# run data through our model and calculate loss
	predictions = mlp(batchX)
	loss = lossFunc(predictions, batchY)
	# update test loss, accuracy, and the number of
	# samples visited

	truePos = (torch.logical_and((predictions-.5)>0, (batchY-.5) > 0)).sum().item()
	falseNeg = (torch.logical_and((predictions-.5)<0, (batchY-.5) > 0)).sum().item()
	totRetr = ((predictions-.5) > 0).sum().item()
	

	valLoss = loss.item() 
	valPrecision,valRecall,valAcc = get_info(predictions,batchY)

	#display model progress on the current test batch
	valTemplate = " val loss: {:.3f} val accuracy: {:.3f} val precision: {:.3f} val recall: {:.3f}"
	print(valTemplate.format(valLoss,valAcc/len(valY), valPrecision, valRecall))
	print("")
