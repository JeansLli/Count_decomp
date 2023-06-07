import numpy as np
import matplotlib.pyplot as plt
import gudhi
import pdb
import networkx as nx
import gudhi as gd

from persistable import Persistable
from persistable.persistable import _HierarchicalClustering, _MetricSpace
from persistable.signed_betti_numbers import signed_betti

def creat_random_points(x_range,y_range,N):
    # output is a (N,2) array
    points = np.random.rand(N,2)
    points[:,0]=points[:,0] * x_range
    points[:,1]=points[:,1] * y_range
    return points


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
    
    # Create a map of old indices to new indices
    old_to_new_indices = {old_idx: new_idx for new_idx, old_idx in enumerate(G.nodes)}

    # Create a numpy array of remaining edges with new indices
    remaining_edges = np.array([(old_to_new_indices[u], old_to_new_indices[v]) for u, v in G.edges()])
    return remaining_points, remaining_edges




# Create the data
num_points = 15
x_range = 10
y_range = 10
points = creat_random_points(x_range,y_range,num_points)

# Plot the data
plt.figure(figsize=(10,10))
plt.scatter(points[:,0], points[:,1], alpha=0.5)
plt.show()


# Compute the hilbert function
ss = [0, 0.5, 1, 1.5, 2, 2.5, 3] #radius_scale
ks = [1, 3 / 4, 1 / 2, 1/4, 0] #degree

p = Persistable(points)
bf = p._bifiltration
hf = bf._hilbert_function(ss, ks, reduced=False, n_jobs=4)
print("hilbert function=\n", hf)

# Compute the signed Betti barcode
sb = signed_betti(hf)
print("signed Betti barcode=\n",sb)

check = np.zeros(hf.shape)
for i in range(hf.shape[0]):
    for j in range(hf.shape[1]):
        for i_ in range(0, i + 1):
            for j_ in range(0, j + 1):
                check[i, j] += sb[i_, j_]

np.testing.assert_equal(check, hf)

num_pos_bar = sb[sb>0].sum()
num_neg_bar = -sb[sb<0].sum()
num_bars = num_pos_bar + num_neg_bar



# Compute the maximal simplicial complex
max_radius = ss[-1]
min_degree = ks[-1]*num_points-1
ms = _MetricSpace(points, "minkowski")
s_neighbors = ms._nn_tree.query_radius(ms._points, max_radius)

edges = []

for i in range(ms.size()):
    for j in s_neighbors[i]:
        if j > i:
            edges.append([i, j])

edges = np.array(edges, dtype=int) # edges.shape=(E,2); edges[i]=(p_id,q_id)
#print("edges =",edges.shape)
#print("the radius is ",max_radius)


filtered_points, filtered_edges =  filter_graph(points, edges, min_degree)
#print("filtered_points\n",filtered_points)
#print("filtered_edges\n", filtered_edges)

num_filtered_points = filtered_points.shape[0]



simplex_tree = gd.SimplexTree()
for edge in filtered_edges:
    simplex_tree.insert(edge)
simplex_tree.expansion(filtered_points.shape[0])


fmt = '%s -> %.2f'
#print("simplex tree is ")
#for filtered_value in simplex_tree.get_filtration():
#    print(fmt % tuple(filtered_value))



result_str = 'Rips complex is of dimension ' + repr(simplex_tree.dimension()) + ' - ' + \
    repr(simplex_tree.num_simplices()) + ' simplices - ' + \
    repr(simplex_tree.num_vertices()) + ' vertices.'
print(result_str)
print("filtered_points.shape ",filtered_points.shape)
print("filtered_edges.shape ",filtered_edges.shape)
cnt=0
for simplex in simplex_tree.get_skeleton(1):
    if len(simplex[0]) == 2:  # only print simplices with 2 vertices (edges)
        cnt+=1
print("num edges = ",cnt)




#print("filtered_edges = \n",filtered_edges)
#rint("filtered_edges length ", filtered_edges.shape[0])
#print("cnt=",cnt-num_filtered_points)

#print("number of positive bars in the signed Betti barcode =",num_pos_bar)
#print("number of negative bars in the signed Betti barcode =",num_neg_bar)
print("number of bars in the signed Betti barcode =",num_bars)
print("#simplices / #bars =",simplex_tree.num_simplices()/num_bars)