import numpy as np
import matplotlib.pyplot as plt
import gudhi
import pdb
import networkx as nx

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


def run_experiments(n_batch, x_range, y_range, num_points, ss, ks):
    n_filter_pts = []
    n_signed_bars = []
    n_simplies = []

    for batch_id in range(n_batch):
        # Create random points
        points = creat_random_points(x_range,y_range,num_points)

        # Compute the signed Betti barcode
        p = Persistable(points)
        bf = p._bifiltration
        hf = bf._hilbert_function(ss, ks, reduced=False, n_jobs=4)
        sb = signed_betti(hf)
        num_pos_bar = sb[sb>0].sum()
        num_neg_bar = -sb[sb<0].sum()
        num_bars = num_pos_bar + num_neg_bar
        n_signed_bars.append(num_bars)

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

        filtered_points, _ =  filter_graph(points, edges, min_degree)
        num_filtered_points = filtered_points.shape[0]
        rips_complex = gudhi.RipsComplex(points=filtered_points,max_edge_length=max_radius) # max_edge_length is the radius of the ball
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=num_filtered_points)

        n_filter_pts.append(simplex_tree.num_vertices())
        n_simplies.append(simplex_tree.num_simplices())
    return n_filter_pts, n_simplies, n_signed_bars
    


x_range=10
y_range=10
num_points=20
ss = [0, 0.5, 1, 1.5, 2, 2.5, 3] #radius_scale
ks = [1, 3 / 4, 1 / 2, 1/4] #degree
n_filter_pts, n_simplies, n_signed_bars = run_experiments(1000, x_range, y_range, num_points, ss, ks)

#print("n_filter_pts",n_filter_pts)
#print("n_simplies",n_simplies)
#print("n_signed_bars",n_signed_bars)

# Create a figure and a set of subplots
fig, axs = plt.subplots(2, figsize=(10, 8)) 

# Plot y1 against x on the first subplot
axs[0].scatter(n_filter_pts, n_signed_bars, color='blue')
#axs[0].set_title('n_signed_bars: n_pts')
axs[0].set_xlabel('n_pts')
axs[0].set_ylabel('n_signed_bars')

# Plot y2 against x on the second subplot
axs[1].scatter(n_simplies, n_signed_bars, color='orange')
#axs[1].set_title('n_signed_bars: n_simplies')
axs[1].set_xlabel('n_simplies')
axs[1].set_ylabel('n_signed_bars')


plt.show()