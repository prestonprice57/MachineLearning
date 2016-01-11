from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import random

iris = datasets.load_iris()

iris_x = iris.data
iris_y = iris.target

np.random.seed(10)
indices = np.random.permutation(len(iris_x))

iris_x_train = iris_x[indices[:105]]
iris_y_train = iris_y[indices[:105]]
iris_x_test = iris_x[indices[105:]]
iris_y_test = iris_y[indices[105:]]

knn = KNeighborsClassifier()

knn.fit(iris_x_train, iris_y_train)

prediction = knn.predict(iris_x_test)


correctPrediction = 0.0
for index in range(len(prediction)):
	if prediction[index] == iris_y_test[index]:
		correctPrediction+=1


print "Percentage: ", (100*correctPrediction/len(prediction))
print "Correct Predictions: ", correctPrediction
print "Total Predictions: ", len(prediction)

class HardCoded:

	def train(self, data):
		return

	def predict(self, data):
		#nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(trainData)
		#distances, indices = nbrs.kneighbors(trainData)
		#print distances
		#print indices
		return



		

hardCoded = HardCoded()

