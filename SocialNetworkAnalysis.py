import networkx as nx
import csv
import matplotlib.pyplot as plt

G = nx.Graph()

reader = csv.reader(open("Mommy_blogLinks.csv", "rb"), skipinitialspace=False)
for r in reader:
	G.add_edge(r[0], r[1], color='red')
	G.add_node(r[0], weight=3)
	G.add_node(r[1], weight=4)
	

# Degree Centrality Calculation
degree = nx.degree_centrality(G)
degree_keys_sorted = sorted(degree, key=degree.get, reverse=True)
#print "\nTop 5 degree centrality"
#for i in xrange(0,5):
#	print str(i+1) + ". " + degree_keys_sorted[i] + ": " + str(degree[degree_keys_sorted[i]])
'''
# Closeness Centrality Calculation
closeness = nx.closeness_centrality(G)
closeness_keys_sorted = sorted(closeness, key=closeness.get, reverse=True)
#print "\nTop 5 closeness centrality"
#for i in xrange(0,5):
#	print str(i+1) + ". " + closeness_keys_sorted[i] + ": " + str(closeness[closeness_keys_sorted[i]])


# Betweenness Centrality Calculation
betweenness = nx.betweenness_centrality(G)
betweenness_keys_sorted = sorted(betweenness, key=betweenness.get, reverse=True)
#print "\nTop 5 betweenness centrality"
#for i in xrange(0,5):
#	print str(i+1) + ". " + betweenness_keys_sorted[i] + ": " + str(betweenness[betweenness_keys_sorted[i]])
'''
#nx.draw(G, node_size=50)
#plt.show()

#G2 = nx.Graph()

string = 'bedat8.com'
new, tmp = string.split('.')
print new

reader = csv.reader(open("Mommy_twitterMentions.csv", "rb"), skipinitialspace=False)
for r in reader:
	G.add_node(r[0], color='red')
	G.add_node(r[1], color='red')
	G.add_edge(r[0], r[1], color='blue')
	

# Degree Centrality Calculation
degree2 = nx.degree_centrality(G)
degree2_keys_sorted = sorted(degree2, key=degree2.get, reverse=True)
#print "\nTop 5 degree centrality"
#for i in xrange(0,5):
#	print str(i+1) + ". " + degree2_keys_sorted[i] + ": " + str(degree2[degree2_keys_sorted[i]])

'''
# Closeness Centrality Calculation
closeness2 = nx.closeness_centrality(G)
closeness2_keys_sorted = sorted(closeness2, key=closeness2.get, reverse=True)
#print "\nTop 5 closeness centrality"
#for i in xrange(0,5):
#	print str(i+1) + ". " + closeness2_keys_sorted[i] + ": " + str(closeness2[closeness2_keys_sorted[i]])


# Betweenness Centrality Calculation
betweenness2 = nx.betweenness_centrality(G)
betweenness2_keys_sorted = sorted(betweenness2, key=betweenness2.get, reverse=True)
#print "\nTop 5 betweenness centrality"
#for i in xrange(0,5):
#	print str(i+1) + ". " + betweenness2_keys_sorted[i] + ": " + str(betweenness2[betweenness2_keys_sorted[i]])
'''
blog_and_twitter = []
for item in degree_keys_sorted:
	new = item.split('.',1)[0]
	for sub in degree2_keys_sorted:
		if new == sub:
			G.add_edge(sub,item, color='green')
			break


edges = G.edges()
edge_colors = [G[u][v]['color'] for u,v in edges]

nodes = G.nodes()
#node_colors = [G[key]['colors'] for key in nodes]

nx.draw(G, node_size=50, edges=edges, edge_color=edge_colors)

plt.show()


