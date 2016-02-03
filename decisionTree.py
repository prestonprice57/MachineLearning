from sklearn import datasets
import numpy as np
from collections import Counter
from math import log
from sets import Set
import csv

# Load Iris Dataset and bin it appropriately

iris = datasets.load_iris()

iris_x = iris.data
iris_target = iris.target

firstD = [i[0] for i in iris_x]

iris_x_bins = np.array(iris_x)

minVals = np.amin(iris_x_bins, axis=0)
maxVals = np.amax(iris_x_bins, axis=0)

for (i,j) in enumerate(iris_x[0]):
	# the + and - 0.01 is so that the min and max values do not fall
	# in their own bin alone
	bins = np.linspace(minVals[i]-0.01,maxVals[i]+0.01, num=12) #changing the binning process may result in better or worse models
	iris_x_bins[:,i] = np.digitize([index[i] for index in iris_x_bins],bins)


# Separate Iris data into train and test sets
SEED_NUMBER = 123456789
np.random.seed(SEED_NUMBER)
indices = np.random.permutation(len(iris_x_bins))

iris_x_train = iris_x_bins[indices[:105]]
iris_target_train = iris_target[indices[:105]]
iris_x_test = iris_x_bins[indices[105:]]
iris_target_test = iris_target[indices[105:]]



cars_x = []
cars_target = []
reader = csv.reader(open("cars.csv", "rb"), skipinitialspace=False)
for r in reader:
	cars_x.append(r[:6])
	cars_target.append(r[6])

cars_x_np = np.array(cars_x)
cars_target_np = np.array(cars_target)

np.random.seed(SEED_NUMBER)
indices2 = np.random.permutation(len(cars_x))

cars_x_train = cars_x_np[indices2[:1209]]
cars_target_train = cars_target_np[indices2[:1209]]
cars_x_test = cars_x_np[indices2[1209:]]
cars_target_test = cars_target_np[indices2[1209:]]





lenses_x = []
lenses_target = []
reader = csv.reader(open("lenses.csv", "rb"), skipinitialspace=False)
for r in reader:
	lenses_x.append(r[:4])
	lenses_target.append(r[4])

lenses_x_np = np.array(lenses_x)
lenses_target_np = np.array(lenses_target)

np.random.seed(SEED_NUMBER)
indices3 = np.random.permutation(len(lenses_x))

lenses_x_train = lenses_x_np[indices3[:17]]
lenses_target_train = lenses_target_np[indices3[:17]]
lenses_x_test = lenses_x_np[indices3[17:]]
lenses_target_test = lenses_target_np[indices3[17:]]



votes_x = []
votes_target = []
reader = csv.reader(open("votes.csv", "rb"), skipinitialspace=False)
for r in reader:
	votes_x.append(r[1:])
	votes_target.append(r[0])

votes_x_np = np.array(votes_x)
votes_target_np = np.array(votes_target)

np.random.seed(SEED_NUMBER)
indices4 = np.random.permutation(len(votes_x))

votes_x_train = votes_x_np[indices4[:304]]
votes_target_train = votes_target_np[indices4[:304]]
votes_x_test = votes_x_np[indices4[304:]]
votes_target_test = votes_target_np[indices4[304:]]



class DecisionTree:

	def __init__(self, train_data, train_target, test_data, test_target):
		self.train_data = train_data
		self.train_target = train_target
		self.test_data = test_data
		self.test_target = test_target
		self.tree = {}

	def calculateEntropy(self,arr):
		count = dict(Counter(arr))

		totalNodes = sum(count.values())
		entropy = 0
		for (key, i) in count.iteritems():
			fraction = float(i)/float(totalNodes)
			entropy += (-fraction)*log(fraction,2)

		return entropy

	def calculateEntropyOfDict(self, dictionary):
		entropyArr = [1000]*len(self.train_data[0])
		entropySum = 0
		firstDictIndex = dictionary.itervalues().next()
		numberOfItems = sum(len(v) for v in firstDictIndex.itervalues())
		
		for i, key in dictionary.iteritems():
			for j, key2 in dictionary[i].iteritems():
				numInArr = len(dictionary[i][j])
				entropySum += (numInArr/float(numberOfItems))*self.calculateEntropy(dictionary[i][j].values())
			
			entropyArr[i] = entropySum
			entropySum = 0

		return entropyArr


	# DATA = inputs, CLASSES = outputs, FEATURES = training data
	def buildTree(self, inputs=None, attributesLeft=None, features=None):

		if features is None:
			features = self.train_target

		if inputs is None:
			inputs = self.train_data

		if attributesLeft is None:
			attributesLeft = [i for i in xrange(len(self.train_data[0]))]


		examples = dict(Counter(features))

		if len(examples) == 1:
			return features[0]  # Leaf of label
		elif len(attributesLeft) == 0:
			return max(examples, key=lambda i: examples[i]) # Node with value of most frequent output in features
		elif len(inputs) == 0:
			return -1 # Node not found
		else:
			columns = {}
			for col, i in enumerate(inputs[0]):
				if col in attributesLeft:
					#columns.append(inputs[:,col])
					columns[col] = inputs[:,col]


			separateData = {}
			for i in attributesLeft:
				separateData[i] = {}
				for j, data in enumerate(columns[i]):
					if columns[i][j] in separateData[i]:
						separateData[i][columns[i][j]][j] = features[j]
					else:
						separateData[i][columns[i][j]] = {j: features[j]}

			#print separateData, "\n\n"
			entropyArr = self.calculateEntropyOfDict(separateData)
			smallEntropy = entropyArr.index(min(entropyArr))

			
			tree = {}
			tree[smallEntropy] = {}
			for key in separateData[smallEntropy].keys():
				tree[smallEntropy].update({key: {}})
				
			#print tree
			num = 1.0

			attributesLeft.remove(smallEntropy)
			#print attributesLeft

			for branch in separateData[smallEntropy]:
				#print branch, separateData[smallEntropy][branch].keys()

				indices = separateData[smallEntropy][branch].keys()
				newInputs = self.train_data[indices]
				newFeatures = self.train_target[indices]

				subtree = self.buildTree(newInputs, attributesLeft, newFeatures)

				tree[smallEntropy][branch] = subtree

			#print tree
			self.tree = tree
			return tree

	def testPredict(self):
		#print self.test_data[3]
		#print self.tree
		correctPrediction = 0
		for i, data in enumerate(self.test_data):
			predictedClass = self.prediction(data, self.tree)

			if predictedClass == self.test_target[i]:
				correctPrediction+=1

		print "Accuracy:", (float(correctPrediction)/len(self.test_target))*100, "%"

		return

	def prediction(self, data, subtree):

		for node in subtree:

			if data[node] in subtree[node]:
				if isinstance(subtree[node][data[node]], dict):
					return self.prediction(data,subtree[node][data[node]])
				else:
					return subtree[node][data[node]]

			else:
				count = dict(Counter(self.train_target))
				return max(count, key=(lambda i: count[i]))


print "Iris:"
dTree = DecisionTree(iris_x_train, iris_target_train, iris_x_test, iris_target_test)
dTree.buildTree()
dTree.testPredict()

print "\nCars:"
dTree2 = DecisionTree(cars_x_train, cars_target_train, cars_x_test, cars_target_test)
dTree2.buildTree()
dTree2.testPredict()

print "\nLenses"
dTree3 = DecisionTree(lenses_x_train, lenses_target_train, lenses_x_test, lenses_target_test)
dTree3.buildTree()
dTree3.testPredict()

print "\nVotes"
dTree4 = DecisionTree(votes_x_train, votes_target_train, votes_x_test, votes_target_test)
dTree4.buildTree()
dTree4.testPredict()



