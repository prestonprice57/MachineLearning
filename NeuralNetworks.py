import numpy as np


data = [5.1,3.5,1.4,0.2]


def createNeuralNetwork(numNodes):
	nodes = {}
	weights = np.random.uniform(-0.5, 0.5, len(data)*(numNodes+1))
	
	k = 0
	for i in xrange(numNodes):
		if not i in nodes.keys():
			nodes[i] = []
		for j in xrange(len(data)+1):
			nodes[i].append(weights[k])
			k+=1

	print weights
	print nodes

createNeuralNetwork(2)
