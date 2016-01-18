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


'''
knn = KNeighborsClassifier()

knn.fit(iris_x_train, iris_y_train)

prediction = knn.predict(iris_x_test)


correct_prediction = 0.0
for index in range(len(prediction)):
	if prediction[index] == iris_y_test[index]:
		correct_prediction+=1


print "Percentage: ", (100*correct_prediction/len(prediction))
print "Correct Predictions: ", correct_prediction
print "Total Predictions: ", len(prediction)
'''


class kNNAlgorithm:

	def __init__(self):
		self.data_points = []
		self.target = []
		self.pred_targets = []

	def train(self, train_data, target):
		self.data_points = train_data
		self.target = target
		# ALGORITHM 1. get z-score (x-x_mean)/standard_dev
		# 2. use euclidean distance formula to get distance
		# 3. find k smallest points
		# 4. the winner is what is use to classify
		

	def predict(self, predict_data, k):

		z_score = self.data_points
		np_z_score = np.array(z_score)
		z_score_predict = predict_data
		firstTimeThrough = True

		z_mean = []
		z_std_dev = []
		# 1. get z-score (x-x_mean)/standard_dev
		for z in z_score:
			for j in xrange(0, len(z)):
				# calculate z-scorw

				if firstTimeThrough == True:
					column = [row[j] for row in z_score]
					np_column = np.array(column)
					z_mean.append(np.mean(column))
					z_std_dev.append(np.std(column))
				
				mean = z_mean[j]
				std_dev = z_std_dev[j]

				z[j] = (z[j]-mean)/std_dev
			firstTimeThrough = False

		# get z-score of predict elements
		for z in z_score_predict:
			for j in xrange(0, len(z)):
				mean = z_mean[j]
				std_dev = z_std_dev[j]

				z[j] = (z[j]-mean)/std_dev

		# 2. use euclidean distance formula to get distance
		# formula: d(p,q) = (q1-p1)^2+(q2-p2)^2.......til qn pn
		for predict in z_score_predict:

			distances = []
			for data in z_score:
				distance = 0
				for i in xrange(0, len(data)):
					distance += (predict[i]-data[i])**2
				distances.append(distance)

			smallestArray = self.findSmallestN(distances, k)			
			predictedTargetArr = self.target[smallestArray]
			predTarget = self.mostCommonElement(predictedTargetArr)
			self.pred_targets.append(predTarget)


	# 3. find k smallest points
	def findSmallestN(self, arr, n):
		smallestNums = [100000000000000000]*n
		smallestIndex = [0]*n
		for index1, i in enumerate(arr):
			for j, num in enumerate(smallestNums):
				if i < num:
					smallestNums[j] = i
					smallestIndex[j] = index1
					break

		return smallestIndex

	# 4. the winner is what is use to classify
	def mostCommonElement(self, arr):
		num_counter = {}
		for num in arr:
			if num in num_counter:
				num_counter[num] += 1
			else:
				num_counter[num] = 1

		common_num = sorted(num_counter, key = num_counter.get, reverse = True)

		return common_num[0]

	def checkTestData(self, test_targets):
		total = 0.0
		correct = 0.0

		for i, target in enumerate(self.pred_targets):
			if int(target) == test_targets[i]:
				correct += 1
			total +=1

		print "Predicted Correct:", int(correct)
		print "Total:            ", int(total)
		print("Percentage:        %.2f%%" % ((correct/total)*100))

		

knn = kNNAlgorithm()
knn.train(iris_x_train, iris_y_train)
knn.predict(iris_x_test, 5)
knn.checkTestData(iris_y_test)

