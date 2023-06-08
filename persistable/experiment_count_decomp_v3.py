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
from sklearn.metrics import DistanceMetric



def generate_torus_points(N, R, r, noise_std_dev):
    # Generate N uniform random angles for the circle and tube
    theta = 2 * np.pi * np.random.rand(N)  # angle for the circle
    phi = 2 * np.pi * np.random.rand(N)  # angle for the tube

    # Convert the toroidal coordinates to Cartesian coordinates
    x = (R + r * np.cos(theta)) * np.cos(phi)
    y = (R + r * np.cos(theta)) * np.sin(phi)
    z = r * np.sin(theta)

    # Add some Gaussian noise to the points
    x += np.random.normal(scale=noise_std_dev, size=N)
    y += np.random.normal(scale=noise_std_dev, size=N)
    z += np.random.normal(scale=noise_std_dev, size=N)
    
    # Scale the coordinates to fit in a [0, 1] x [0, 1] x [0, 1] cube
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    #z = (z - z.min()) / (z.max() - z.min())
    z = z + 0.5

    return np.column_stack([x, y, z])

def generate_clustered_points(N, d, c, cluster_std_dev):
    # Initialize an empty array to hold the points
    points = np.empty((0, d))

    # Calculate the number of points per cluster
    points_per_cluster = N // c

    for _ in range(c):
        # Generate a random center for this cluster
        center = np.random.rand(d)

        # Generate points around this center with Gaussian noise
        cluster_points = np.random.normal(loc=center, scale=cluster_std_dev, size=(points_per_cluster, d))

        # Append the points to the array
        points = np.vstack([points, cluster_points])

    min_values = np.min(points)
    max_values = np.max(points)
    #print("min_values=",min_values)
    #print("max_values=",max_values)
    # Normalize the points so they fall within the range [0, 1]
    normalized_points = (points - min_values) / (max_values - min_values)

    return normalized_points

def create_uniformly_distributed_points(dim,pts_range,N):
    # output is a (N,2) array
    points = np.random.rand(N,dim)*pts_range
    return points

def create_torus_points(pts_range,N):
    points = generate_torus_points(N, 1, 0.1, 0.05)
    points = points * pts_range
    return points

def create_Gaussion_points(dim,pts_range,N,n_clusters):
    points = generate_clustered_points(N, dim, n_clusters, 0.05)
    points = points * pts_range
    return points



def run_experiments(n_batch, dim_point, pts_range, num_points, data_type, n_clusters, ss, ks):
    n_signed_bars = []
    n_simplices = []
    n_vertices = []

    for num_point in num_points:
        for _ in range(n_batch):

            # Create points
            if data_type=="uniform":
                points = create_uniformly_distributed_points(dim_point,pts_range, num_point)
            elif data_type=="torus":
                points = create_torus_points(pts_range, num_point)
            elif data_type=="gaussion":
                points = create_Gaussion_points(dim_point,pts_range, num_point,n_clusters)


            # Compute the signed Betti barcode
            p = Persistable(points,metric="minkowski")
            bf = p._bifiltration
            hf = bf._hilbert_function(ss, ks, reduced=False, n_jobs=4)
            sb = signed_betti(hf)
            num_pos_bar = sb[sb>0].sum()
            num_neg_bar = -sb[sb<0].sum()
            num_bars = num_pos_bar + num_neg_bar
            n_signed_bars.append(num_bars)
            n_simplices.append(num_point+num_point*(num_point)/2)
            n_vertices.append(num_point)

    return n_signed_bars, n_simplices, n_vertices
    

def draw_plot(x,y,x_name,y_name,title_name,fig_name):
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y)

    # Set x and y axis titles
    plt.xlabel(x_name)
    plt.ylabel(y_name)

    plt.title(title_name)

    plt.grid(True)
    plt.savefig("../experiment_result_v3/"+fig_name)
    #plt.show()


##### User Define
n_batch = 2
pts_range = 10
data_type = "uniform"
max_n_clusters = 8
max_dim_points = 8
num_points=[10,20,30,40,60,80,100,200,400,600,800,1000,1400,1800,2000,2300,2500,3000,4000,6000,8000]
#num_points=[10,20,30,40,60,70,80,90,100,110,120]
######

dist = DistanceMetric.get_metric('minkowski')
#pdb.set_trace()

n_clusters = 0 

if data_type == "torus":
    dim_point = 3
    max_radius = (dist.pairwise([np.zeros(dim_point),np.ones(dim_point)*pts_range])[0,1]+1)/2

    ss = np.linspace(0, max_radius, 10) #radius
    ks = np.linspace(1, 0, 10) #degree

    n_signed_bars, n_simplices, n_vertices = run_experiments(n_batch, dim_point, pts_range, num_points, data_type, n_clusters,ss, ks)

    title_type = data_type + " distributed points in " + str(dim_point) + "D: "
    title_name1 = title_type + "#bars v.s. #simplices"
    title_name2 = title_type + "#vertices v.s. #simplices"
    title_name3 = title_type + "#vertices v.s. sqrt(#simplices)"

    fig_name1 = data_type + "/" + data_type+"_" + str(dim_point) + "D_bars-simplices"
    fig_name2 = data_type + "/" + data_type+"_" + str(dim_point) + "D_bars-vertices"
    fig_name3 = data_type +"/" + data_type+"_" + str(dim_point) + "D_bars-sqrt(n_simplices)"


    n_simplices_array = np.array(n_simplices)
    sqrt_n_simplices = np.sqrt(n_simplices_array)
    sqrt_n_simplices_list = list(sqrt_n_simplices)

    draw_plot(n_simplices,n_signed_bars,'n_simplices','n_Hilbert_bars',title_name1,fig_name1)
    draw_plot(n_vertices, n_signed_bars, 'n_pts','n_Hilbert_bars',title_name2,fig_name2)
    draw_plot(sqrt_n_simplices_list, n_signed_bars, 'sqrt_n_simplices','n_Hilbert_bars',title_name3,fig_name3)


elif data_type== "gaussion":
    for n_clusters in range(1,max_n_clusters):
        for dim_point in range(1,max_dim_points):
            max_radius = (dist.pairwise([np.zeros(dim_point),np.ones(dim_point)*pts_range])[0,1]+1)/2
            ss = np.linspace(0, max_radius, 10) #radius
            ks = np.linspace(1, 0, 10) #degree

            n_signed_bars, n_simplices, n_vertices = run_experiments(n_batch, dim_point, pts_range, num_points, data_type, n_clusters,ss, ks)
            title_type = data_type + " distributed points in " + str(dim_point) + "D: "
            title_type = str(n_clusters) + " clusters " + title_type
            title_name1 = title_type + "#bars v.s. #simplices"
            title_name2 = title_type + "#vertices v.s. #simplices"
            title_name3 = title_type + "#vertices v.s. sqrt(#simplices)"
            fig_name1 = data_type +"/" + data_type+"_" + str(dim_point) + "D_"+str(n_clusters)+"_clusters_bars-simplices"
            fig_name2 = data_type +"/" + data_type+"_" + str(dim_point) + "D_"+str(n_clusters)+"_clusters_bars-vertices"
            fig_name3 = data_type +"/" + data_type+"_" + str(dim_point) + "D_"+str(n_clusters)+"_clusters_bars-sqrt(n_simplices)"
            
            n_simplices_array = np.array(n_simplices)
            sqrt_n_simplices = np.sqrt(n_simplices_array)
            sqrt_n_simplices_list = list(sqrt_n_simplices)

            draw_plot(n_simplices,n_signed_bars,'n_simplices','n_Hilbert_bars',title_name1,fig_name1)
            draw_plot(n_vertices, n_signed_bars, 'n_pts','n_Hilbert_bars',title_name2,fig_name2)
            draw_plot(sqrt_n_simplices_list, n_signed_bars, 'sqrt_n_simplices','n_Hilbert_bars',title_name3,fig_name3)

             
elif data_type=="uniform":
    for dim_point in range(1,max_dim_points):
        max_radius = (dist.pairwise([np.zeros(dim_point),np.ones(dim_point)*pts_range])[0,1]+1)/2
        ss = np.linspace(0, max_radius, 10) #radius
        ks = np.linspace(1, 0, 10) #degree

        n_signed_bars, n_simplices, n_vertices = run_experiments(n_batch, dim_point, pts_range, num_points, data_type, n_clusters,ss, ks)
        title_type = data_type + " distributed points in " + str(dim_point) + "D: "
        title_name1 = title_type + "#bars v.s. #simplices"
        title_name2 = title_type + "#vertices v.s. #simplices"
        title_name3 = title_type + "#vertices v.s. sqrt(#simplices)"
        fig_name1 = data_type +"/" + data_type+"_" + str(dim_point) + "D_bars-simplices"
        fig_name2 = data_type +"/" + data_type+"_" + str(dim_point) + "D_bars-vertices"
        fig_name3 = data_type +"/" + data_type+"_" + str(dim_point) + "D_bars-sqrt(n_simplices)"

        n_simplices_array = np.array(n_simplices)
        sqrt_n_simplices = np.sqrt(n_simplices_array)
        sqrt_n_simplices_list = list(sqrt_n_simplices)
        draw_plot(n_simplices,n_signed_bars,'n_simplices','n_Hilbert_bars',title_name1,fig_name1)
        draw_plot(n_vertices, n_signed_bars, 'n_pts','n_Hilbert_bars',title_name2,fig_name2)
        draw_plot(sqrt_n_simplices_list, n_signed_bars, 'sqrt_n_simplices','n_Hilbert_bars',title_name3,fig_name3)
