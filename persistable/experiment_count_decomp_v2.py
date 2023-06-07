import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
import pdb
import networkx as nx

from persistable import Persistable
from persistable.persistable import _HierarchicalClustering, _MetricSpace
from persistable.signed_betti_numbers import signed_betti
from datetime import datetime
from collections import Counter
import math



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
    n_signed_bars = []

    for _ in range(n_batch):
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

    return n_signed_bars
    

def draw_plot(x,y,x_name,y_name,line_k,time,title_name):
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y)

    points = list(zip(x, y))
    counter = Counter(points)
    for point, count in counter.items():
        plt.text(point[0], point[1], str(count))

    # Find min and max values
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    # Set x and y axis titles
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    # Draw the line y=x
    #if (ymax-ymin)<(xmax-xmin):
    #    line_x = np.linspace(ymin, ymax, 100)
    #else:
    #    line_x = np.linspace(ymin/line_k, xmax, 100)
    #plt.plot(line_x, line_k*line_x, 'k--', color='red', label='y = '+str(line_k)+'x')
    #plt.legend()

    plt.title(title_name)

    plt.grid(True)
    fig_name = time+'_'+x_name+'_'+y_name
    plt.savefig("../experiment_result_v2/"+fig_name)
    plt.show()

def draw_histogram(n_signed_bars, num_points, n_simplices,time,n_batch,ss,ks):
    # Create a histogram of the data
    plt.figure(figsize=(10, 10))
    counts, bins, patches = plt.hist(n_signed_bars, bins=range(min(n_signed_bars), max(n_signed_bars) + 2), align='left', edgecolor='black')

    # Add counts above the two bars
    for count, bin, patch in zip(counts, bins, patches):
        plt.text(patch.get_x() + patch.get_width()/2, patch.get_height()+0.3, int(count), 
                ha='center', va='bottom')

    
    title_name = 'Histogram for #bars, ' +str(num_points)+ \
                 ' vertices, ' + str(n_simplices)+ \
                 ' simplices, ' + str(n_batch)+ \
                    ' bifiltrations.\n' + 'radius.len='+str(ss.shape[0]) + '\n'+\
                 ' normalized degree.len='+str(ks.shape[0])
    plt.title(title_name)

    plt.xlabel('#bars')
    plt.ylabel('Frequency')

    fig_name = time+'_'+str(n_simplices)+' simplices'
    plt.savefig("../experiment_result_v2/"+fig_name)
    plt.show()


x_range=10
y_range=10
num_points=10
n_batch=500
#ss = [0, 0.5, 1 ,2, 3, 5,6, 8, 10,11, 13,15,17, 18, 20] #radius_scale
#ks = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1,0] #degree

ss = np.linspace(0, 20, 80)
ks = np.linspace(1, 0, 5)
n_signed_bars = run_experiments(n_batch, x_range, y_range, num_points, ss, ks)
#print("n_filter_pts=",n_filter_pts)
#print("n_simplices=",n_simplices)
#print("n_signed_bars=",n_signed_bars)



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

# Create a Counter object
counter = Counter(n_signed_bars)

# Print elements with counts
for element, count in counter.items():
    print(f"#bars: {element}, Count: {count}")

#draw_plot(n_simplices,n_signed_bars,'n_simplices','n_bars',1, time_string,title_name)
n_simplices = int(math.pow(2,num_points)-1)
n_filter_pts = np.ones(n_batch,dtype=int) * n_simplices
#draw_plot(n_filter_pts, n_signed_bars, 'n_pts','n_bars',3, time_string,title_name)
draw_histogram(n_signed_bars, num_points, n_simplices, time_string,n_batch,ss,ks)


#n_simplices_array = np.array(n_simplices)
#log_n_simplices = np.log(n_simplices_array)
#log_n_simplices_list = list(log_n_simplices)

#draw_plot(log_n_simplices_list,n_signed_bars,'log(n_simplices)','n_bars',5, time_string,title_name)

