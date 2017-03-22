"""
This file should contain the graphs and data structure
"""
import sys

from utils.graph_utils import init_pixel_graph, contract_nodes


G = init_pixel_graph(3,2)

print G.edges(0)
print G[0]


contract_nodes(G, 0, 1)
contract_nodes(G, 3, 4)
print G.edges(0)
print G[0]
print G[3][0]

print G[0]



