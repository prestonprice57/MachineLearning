from sklearn import datasets
import numpy as np
import random

iris = datasets.load_iris() # Load dataset

dataArray = [] # index 0=data 1=target

# Load data into array
for i, data in enumerate(iris.data):
	target = i/50
	array = [data, target]
	dataArray.append(array)

random.shuffle(dataArray) # Randomize dataset

trainData = dataArray[:105] # 70%=train data
testData = dataArray[105:] # 30%=test data

trainDataNP = np.array(trainData)
testDataNP = np.array(testData)

class HardCoded:

	def train(self, data):
		return

	def predict(self, data):
		for dataPoint in data:
			#runMachineLearningAlgorithm(dataPoint)
			print dataPoint[1]
		

hardCoded = HardCoded()
hardCoded.train(trainDataNP)
hardCoded.predict(testDataNP)
