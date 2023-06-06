import networkx as nx
import numpy as np

def filter_graph(points, edges, s):
    # Create a graph
    G = nx.Graph()

    # Add nodes
    for idx, point in enumerate(points):
        G.add_node(idx, coordinate=point)

    # Add edges
    for edge in edges:
        G.add_edge(*edge)

    # List of nodes to be removed
    nodes_to_remove = [node for node, degree in dict(G.degree()).items() if degree < s]

    # Remove nodes
    G.remove_nodes_from(nodes_to_remove)

    # Create a dictionary of remaining nodes and their coordinates
    remaining_points = np.array([data['coordinate'] for node, data in G.nodes(data=True)])
    remaining_edges = np.array(list(G.edges()))
    return G, remaining_points, remaining_edges

# Sample input
points = [(1,2), (3,4), (5,6),(7,8),(9,10)]
edges = [(0, 1), (0, 2),(1,2),(3,0),(3,1),(3,2),(0,4),(2,4)]

filtered_graph, remaining_points, remaining_edges = filter_graph(points, edges, s=4)

print(filtered_graph.nodes())
print("filtered points = \n",remaining_points)
print(filtered_graph.edges())
print(remaining_edges)

