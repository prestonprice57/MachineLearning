import numpy as np
import math
from sklearn import datasets


# Load Iris Dataset and bin it appropriately

iris = datasets.load_iris()

iris_x = iris.data
iris_target = iris.target

# Separate Iris data into train and test sets
SEED_NUMBER = 5
np.random.seed(SEED_NUMBER)
indices = np.random.permutation(len(iris_x))

iris_x_train = iris_x[indices[:105]]
iris_target_train = iris_target[indices[:105]]
iris_x_test = iris_x[indices[105:]]
iris_target_test = iris_target[indices[105:]]

train_data = iris_x_train.tolist()

for i, row in enumerate(train_data):
	for j, item in enumerate(row):
		train_data[i][j]/=10

def createNeuralNetwork(data, nodesInLayer):
	nodes = {}
	weights = [0]*(len(nodesInLayer)+1)
	epochNum = 5
	bias = -1
	n = 0.1

	layer = 'layer0'
	for row in data:
		row.insert(0,bias)

	nodes[layer] = { 'values': data[0], 'weights': [], }


	# Add data to 1st layer
	for i in xrange(nodesInLayer[0]):
		nodes[layer]['weights'].append(np.random.uniform(-0.5, 0.5, (len(data[0]))))

	# Add the rest of the layers
	for i in xrange(1,len(nodesInLayer)+1):
		if i < len(nodesInLayer):
			newLayer = 'layer' + str(i)
			nodes[newLayer] = { 'values': [bias], 'weights': [], 'error': [] }
			for j in xrange(nodesInLayer[i]):
				nodes[newLayer]['weights'].append(np.random.uniform(-0.5, 0.5, (nodesInLayer[i-1]+1)))
		else:
			newLayer = 'layer' + str(i)
			nodes[newLayer] = { 'values': [], 'error': [] }

	count = 0
	for z in xrange(epochNum):
		# Start Feed forward and back propagation epoch
		for d in data:

			#Reset Values
			nodes['layer0']['values'] = d
			for i in xrange(1,len(nodesInLayer)+1):
				resetLayer = 'layer' + str(i)
				if i < len(nodesInLayer):
					nodes[resetLayer]['values'] = [bias]
					nodes[resetLayer]['error'] = []
				else:
					nodes[resetLayer]['values'] = []
					nodes[resetLayer]['error'] = []
					nodes[resetLayer]['outputs'] = []
			# Feed Forward
			for i in xrange(1,len(nodesInLayer)+2):
				iLayer = 'layer' + str(i-1)
				jLayer = 'layer' + str(i)
				total = 0
				
				if i < len(nodesInLayer)+1:			
					for arr in nodes[iLayer]['weights']:
						aVal = 0
						for j, num in enumerate(arr):
							aVal += (num*nodes[iLayer]['values'][j])
						aVal = 1/(1+(math.exp(-aVal)))
						nodes[jLayer]['values'].append(aVal)

			# Calculate output
			outLayerNum = len(nodesInLayer)
			outLayer = 'layer' + str(outLayerNum)
			maxVal = max(nodes[outLayer]['values'])
			nodes[outLayer]['outputs'] = []
			
			for num in nodes[outLayer]['values']:
				if num == maxVal:
					nodes[outLayer]['outputs'].append(1)
				else:
					nodes[outLayer]['outputs'].append(0)



			# Back Propagation

			# Output Nodes
			errorj = 0
			prevLayer = 'layer' + str(len(nodesInLayer)-1)
			for j, aj in enumerate(nodes[outLayer]['values']):
				errorj = aj*(1-aj)*(aj-nodes[outLayer]['outputs'][j])
				nodes[outLayer]['error'].append(errorj)


			for i, node in enumerate(nodes[prevLayer]['weights']):
				for j, wij in enumerate(node):
					node[j] = wij-(n*nodes[outLayer]['error'][i]*nodes[prevLayer]['values'][i])

			# Hidden Nodes

			for i in xrange(len(nodesInLayer)-1,0,-1):
				layerk = 'layer' + str(i+1)
				layerj = 'layer' + str(i)
				layeri = 'layer' + str(i-1)
				for j, aj in enumerate(nodes[layerj]['values']):
					sigmaWeight = 0	
					for k, weight in enumerate(nodes[layerj]['weights']):
						for w in weight:
							sigmaWeight += w*nodes[layerk]['error'][k]
					errorj = aj*(1-aj)*sigmaWeight
					nodes[layerj]['error'].append(errorj)

				for j, node in enumerate(nodes[layeri]['weights']):
					for k, wij in enumerate(node):
						node[k] = wij-(n*nodes[layerj]['error'][j]*nodes[layeri]['values'][j])

			
			print nodes
			print
			print



			outLayerNum = len(nodesInLayer)
			outLayer = 'layer' + str(outLayerNum)
			if nodes[outLayer]['outputs'] == [0,0,1]:
				count+=1
	print count
	

createNeuralNetwork(train_data, [3,3,3])
