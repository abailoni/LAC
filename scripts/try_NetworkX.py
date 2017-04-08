"""
This file should contain the graphs and data structure
"""
from utils.graph_utils import init_pixel_graph, contract_nodes


G = init_pixel_graph(3,2)

print G.edges(0)
print G[0]


contract_nodes(G, 0, 1)
contract_nodes(G, 3, 4)
print 1 in G
print "Edges from 0:", G.edges(0)
print "Node 3: ", G[3]
print "edge?", G[3][0]
print G.edges(0,2)
# [Node1][Node2][Num_edge][DictKey]
print G[0][2][0]



