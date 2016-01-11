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


trainData = dataArray[0:105] # 70%=train data
testData = dataArray[105:150] # 30%=test data

trainDataNP = np.array(trainData)
testDataNP = np.array(testData)

class HardCoded:

	def train(self, data):
		return

	def predict(self, data):
		predictions = []
		correctPrediction = 0.0
		for i, dataPoint in enumerate(data):
			#runMachineLearningAlgorithm(dataPoint)
			predictedNumber = 1
			predictions.append(predictedNumber)

			if predictions[i] == dataPoint[1]:
				correctPrediction += 1
		predictPercent = ((correctPrediction/len(data)*100))
		print "Method Percentage = " + str(predictPercent) + "%"



		

hardCoded = HardCoded()
hardCoded.train(trainDataNP)
hardCoded.predict(testDataNP)
