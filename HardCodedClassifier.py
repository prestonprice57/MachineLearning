from sklearn import datasets
import numpy as np
import random

iris = datasets.load_iris()

iris_x = iris.data
iris_y = iris.target

# np.random.seed(5) uncomment this to seed random 
# generator and obtain constant random results
indices = np.random.permutation(len(iris_x))

iris_x_train = iris_x[indices[:105]]
iris_y_train = iris_y[indices[:105]]
iris_x_test = iris_x[indices[105:]]
iris_y_test = iris_y[indices[105:]]

class HardCoded:

	def train(self, data, figure):
		return

	def predict(self, data):
		predictions = []
		correctPrediction = 0.0
		for i, dataPoint in enumerate(data):
			#runMachineLearningAlgorithm(dataPoint)
			predictedNumber = 1
			predictions.append(predictedNumber)

			if predictions[i] == dataPoint:
				correctPrediction += 1
		predictPercent = ((correctPrediction/len(data)*100))
		print "Method Percentage = " + str(predictPercent) + "%"




hardCoded = HardCoded()
hardCoded.train(iris_x_train, iris_y_train)
hardCoded.predict(iris_y_test)
