import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
import pdb
import networkx as nx

from persistable import Persistable
from persistable.persistable import _HierarchicalClustering, _MetricSpace
from persistable.signed_betti_numbers import signed_betti
from datetime import datetime



def create_random_points(x_range,y_range,N):
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
    max_radius = ss[-1]
    min_degree = ks[-1]*num_points-1

    for batch_id in range(n_batch):
        # Create random points
        points = create_random_points(x_range,y_range,num_points)

        # Compute the signed Betti barcode
        p = Persistable(points,metric="minkowski")
        bf = p._bifiltration
        hf = bf._hilbert_function(ss, ks, reduced=False, n_jobs=4)
        sb = signed_betti(hf)
        num_pos_bar = sb[sb>0].sum()
        num_neg_bar = -sb[sb<0].sum()
        num_bars = num_pos_bar + num_neg_bar
        n_signed_bars.append(num_bars)

        # Compute the maximal simplicial complex
        
        ms = _MetricSpace(points, "minkowski")
        s_neighbors = ms._nn_tree.query_radius(ms._points, max_radius)
        edges = []

        for i in range(ms.size()):
            for j in s_neighbors[i]:
                if j > i:
                    edges.append([i, j])

        edges = np.array(edges, dtype=int) # edges.shape=(E,2); edges[i]=(p_id,q_id)

        filtered_points, filtered_edges =  filter_graph(points, edges, min_degree)
        num_filtered_points = filtered_points.shape[0]
        simplex_tree = gd.SimplexTree()
        # first add points
        for pts_id in range(num_filtered_points):
            simplex_tree.insert([pts_id])
        # then add edges
        for edge in filtered_edges:
            simplex_tree.insert(edge)
        
        simplex_tree.expansion(num_filtered_points) # expand edges to Rips Complex
        #if simplex_tree.num_simplices()==0:
        #    pdb.set_trace()
        n_filter_pts.append(num_filtered_points)
        n_simplies.append(simplex_tree.num_simplices())
    return n_filter_pts, n_simplies, n_signed_bars
    

def draw_plot(x,y,x_name,y_name,line_k,time,title_name):
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y)

    # Find min and max values
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    # Set x and y axis titles
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    # Draw the line y=x
    if (ymax-ymin)<(xmax-xmin):
        line_x = np.linspace(ymin/line_k, ymax/line_k, 100)
    else:
        line_x = np.linspace(ymin/line_k, xmax, 100)
    plt.plot(line_x, line_k*line_x, 'k--', color='red', label='y = '+str(line_k)+'x')
    plt.legend()

    plt.title(title_name)

    #plt.grid(True)
    fig_name = time+'_'+x_name+'_'+y_name
    plt.savefig("../experiment_result_v1/"+fig_name)
    plt.show()

x_range = 10
y_range = 10
num_points = 15
ss = [0, 1, 1.5, 2, 2.5, 3, 3.5] #radius_scale
ks = [1, 3 / 4, 1 / 2, 1 / 4, 0] #degree
n_filter_pts, n_simplices, n_signed_bars = run_experiments(50, x_range, y_range, num_points, ss, ks)
print("n_simplices = ",n_simplices)
print("n_signed_bars = ",n_signed_bars)
print("n_filter_pts",n_filter_pts)



max_radius = ss[-1]
min_degree = ks[-1]*num_points-1

current_time = datetime.now().time()
# Convert the time to a string
time_string = current_time.strftime('%H:%M:%S')
title_name = str(num_points)+" initial points"+", "+ \
             "region size="+str(x_range)+"*"+str(y_range)+", "+ \
             "max_radius="+str(max_radius)+", "+\
             "min_degree="+str(min_degree)
print("title_name=",title_name)



draw_plot(n_simplices,n_signed_bars,'n_simplices','n_bars',1, time_string,title_name)
draw_plot(n_filter_pts, n_signed_bars, 'n_pts','n_bars',3, time_string,title_name)

n_simplices_array = np.array(n_simplices)
log_n_simplices = np.log(n_simplices_array)
log_n_simplices_list = list(log_n_simplices)

#draw_plot(log_n_simplices_list,n_signed_bars,'log(n_simplices)','n_bars',5, time_string,title_name)

