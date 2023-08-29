import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
import pdb
import networkx as nx

import persistable
from persistable import Persistable
from persistable.persistable import _HierarchicalClustering, _MetricSpace
from persistable.signed_betti_numbers import signed_betti
from datetime import datetime
from collections import Counter
import math
from sklearn.metrics import DistanceMetric



def generate_torus_points(N, R, r, noise_std_dev=0.1):
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

def generate_hypersphere_points(N, k, radius=1, noise_std_dev=0.1):
    # Generate N k-dimensional Gaussian random vectors
    points = np.random.normal(size=(N, k))

    # Normalize each point to the hypersphere of given radius
    points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
    points *= radius

    # Add Gaussian noise to the points
    noise = np.random.normal(scale=noise_std_dev, size=(N, k))
    points += noise

    # Normalize the points so they fall within the range [0, 1]
    min_values = np.min(points, axis=0)
    max_values = np.max(points, axis=0)
    normalized_points = (points - min_values) / (max_values - min_values)

    return normalized_points



def create_uniformly_distributed_points(dim,pts_range,N):
    # output is a (N,2) array
    points = np.random.rand(N,dim)*pts_range
    return points

def create_torus_points(pts_range,N):
    points = generate_torus_points(N, 1, 0.1, 0)
    points = points * pts_range

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    #plt.show()

    return points

def create_Gaussian_points(dim,pts_range,N,n_clusters):
    points = generate_clustered_points(N, dim, n_clusters, 0.05)
    points = points * pts_range
    return points

def create_sphere_points(dim,pts_range,N):
    points = generate_hypersphere_points(N,dim)
    points = points * pts_range
    return points


def hf_degree_rips(
    distance_matrix,
    min_rips_value,
    max_rips_value,
    max_normalized_degree,
    min_normalized_degree,
    grid_granularity,
    max_homological_dimension,
    subsample_size = None,
):
    if subsample_size == None:
        p = persistable.Persistable(distance_matrix, metric="precomputed")
    else:
        p = persistable.Persistable(distance_matrix, metric="precomputed", subsample=subsample_size)

    rips_values, normalized_degree_values, hilbert_functions, minimal_hilbert_decompositions = p._hilbert_function(
        min_rips_value,
        max_rips_value,
        max_normalized_degree,
        min_normalized_degree,
        grid_granularity,
        homological_dimension=max_homological_dimension,
    )
    #print("higher degree hilbert_functions:\n",hilbert_functions)
    #print("higher degree minimal_hilbert_decompositions:\n",minimal_hilbert_decompositions)
    return rips_values, normalized_degree_values, hilbert_functions, minimal_hilbert_decompositions



    
def run_experiments_higher_homology(n_batch, dim_point, pts_range, num_points, data_type, n_clusters, min_radius,max_radius, max_degree,min_degree,grid_granularity,max_homological_dimension):
    n_signed_bars = []
    n_simplices = []
    n_vertices = []

    for num_point in num_points:
        for _ in range(n_batch):
            print("processing "+str(num_point) +" points")
            # Create points
            if data_type=="uniform":
                points = create_uniformly_distributed_points(dim_point,pts_range, num_point)
            elif data_type=="torus":
                points = create_torus_points(pts_range, num_point)
            elif data_type=="gaussian":
                points = create_Gaussian_points(dim_point,pts_range, num_point,n_clusters)
            elif data_type=="sphere":
                points = create_sphere_points(dim_point,pts_range,num_point)

            dist = DistanceMetric.get_metric('minkowski')
            distance_matrix = dist.pairwise(points)
            # Compute the signed Betti barcode
            rips_values, normalized_degree_values, hilbert_functions, minimal_hilbert_decompositions=hf_degree_rips(distance_matrix, min_radius,max_radius,max_degree,min_degree,grid_granularity,max_homological_dimension)
            #pdb.set_trace()
            sb = minimal_hilbert_decompositions[max_homological_dimension]
            #hf = hilbert_functions[max_homological_dimension]
            
            num_pos_bar = sb[sb>0].sum()
            num_neg_bar = -sb[sb<0].sum()
            num_bars = num_pos_bar + num_neg_bar

            num_simplice = 0
            temp=0
            for k in range(max_homological_dimension+1, max_homological_dimension+3):
                temp=math.comb(num_point, k)
                num_simplice+=temp
            num_simplice+=temp

            n_signed_bars.append(num_bars)    
            n_simplices.append(num_simplice)
            n_vertices.append(num_point)

    return np.array(n_signed_bars), np.array(n_simplices), np.array(n_vertices)

def draw_plot(x,y,x_name,y_name,title_name,fig_name):
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y)

    # Set x and y axis titles
    plt.xlabel(x_name)
    plt.ylabel(y_name)

    plt.title(title_name)

    plt.grid(True)
    plt.savefig("../experiment_result_v5/"+fig_name)
    #plt.show()


def draw_plot_ratio(x,y,x_name,y_name,title_name,fig_name):
    plt.figure(figsize=(10, 8))
    plt.scatter(x, y/x)

    # Set x and y axis titles
    plt.xlabel(x_name)
    plt.ylabel(y_name+'/'+x_name)
    #plt.axhline(y = 1, color = 'r', linestyle = '--')

    plt.title(title_name)

    plt.grid(True)
    plt.savefig("../experiment_result_v5/"+fig_name+"_ratio")
    #plt.show()


##### User Define
pts_range = 10
data_type = "uniform"
max_n_clusters = 5
max_dim_points = 5
max_degree = 1
min_degree = 0
min_radius = 0
grid_granularity = 10
max_homological_dimension = 0
#num_points=[10,20,30,40,60,80,100,200,400,600,800,1000,1400,1800]
#num_points=[10,20,30,40,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
#num_points=np.arange(100,1000,50)

#num_points=np.arange(5,105,5)
#n_batch = 5

#num_points=np.arange(2,11,1)???
#n_batch = 500

num_points=np.arange(3,11,1)
n_batch = 500
print("num_points=",num_points)
######

dist = DistanceMetric.get_metric('minkowski')
#pdb.set_trace()

n_clusters = 0 

if data_type == "torus":
    dim_point = 3
    max_radius = (dist.pairwise([np.zeros(dim_point),np.ones(dim_point)*pts_range])[0,1]+1)/2

    n_signed_bars, n_simplices, n_vertices = run_experiments_higher_homology(n_batch, dim_point, pts_range, num_points, data_type, n_clusters, min_radius,max_radius, max_degree,min_degree,grid_granularity,max_homological_dimension)

    title_type = data_type + " distributed points in " + str(dim_point) + "D: "
    title_name1 = title_type + "#bars v.s. #simplices, degree "+str(max_homological_dimension)
    title_name2 = title_type + "#bars v.s. #vertices, degree "+str(max_homological_dimension)
    title_name3 = title_type + "#bars v.s. sqrt(#simplices), degree "+str(max_homological_dimension)

    fig_name1 = data_type + "/" + data_type+"_" + str(dim_point) + "D_bars-simplices_degree_"+str(max_homological_dimension)
    fig_name2 = data_type + "/" + data_type+"_" + str(dim_point) + "D_bars-vertices_degree_"+str(max_homological_dimension)
    fig_name3 = data_type +"/" + data_type+"_" + str(dim_point) + "D_bars-sqrt(n_simplices)_degree_"+str(max_homological_dimension)

    sbrt_n_simplices = np.cbrt(n_simplices)
    cbrt_n_simplices_list = list(sbrt_n_simplices)

    draw_plot(n_simplices, n_signed_bars, 'n_k+2n_(k+1)','n_Hilbert_bars',title_name1,fig_name1)
    draw_plot_ratio(n_simplices, n_signed_bars, 'n_k+2n_(k+1)','n_Hilbert_bars',title_name1,fig_name1)



elif data_type== "gaussian":
    for n_clusters in range(1,max_n_clusters):
        for dim_point in range(1,max_dim_points):
            max_radius = (dist.pairwise([np.zeros(dim_point),np.ones(dim_point)*pts_range])[0,1]+1)/2

            n_signed_bars, n_simplices, n_vertices = run_experiments_higher_homology(n_batch, dim_point, pts_range, num_points, data_type, n_clusters, min_radius,max_radius, max_degree,min_degree,grid_granularity,max_homological_dimension)
            title_type = data_type + " distributed points in " + str(dim_point) + "D: "
            title_type = str(n_clusters) + " clusters " + title_type
            title_name1 = title_type + "#bars v.s. #simplices, degree "+str(max_homological_dimension)
            title_name2 = title_type + "#bars v.s. #vertices, degree "+str(max_homological_dimension)
            title_name3 = title_type + "#bars v.s. sqrt(#simplices), degree "+str(max_homological_dimension)
            fig_name1 = data_type +"/" + data_type+"_" + str(dim_point) + "D_"+str(n_clusters)+"_clusters_bars-simplices_degree_"+str(max_homological_dimension)
            fig_name2 = data_type +"/" + data_type+"_" + str(dim_point) + "D_"+str(n_clusters)+"_clusters_bars-vertices_degree_"+str(max_homological_dimension)
            fig_name3 = data_type +"/" + data_type+"_" + str(dim_point) + "D_"+str(n_clusters)+"_clusters_bars-sqrt(n_simplices)_degree_"+str(max_homological_dimension)
            
            sqrt_n_simplices = np.sqrt(n_simplices)
            sqrt_n_simplices_list = list(sqrt_n_simplices)

            if max_homological_dimension==0:
                draw_plot(n_vertices, n_signed_bars, 'n_pts','n_Hilbert_bars',title_name2,fig_name2)
            elif max_homological_dimension==1:
                n_edges = n_vertices*(n_vertices-1)
                title_name = title_type + "#bars v.s. #edges, degree "+str(max_homological_dimension)
                fig_name = data_type +"/" + data_type+"_" + str(dim_point) + "D_bars-edges_degree_"+str(max_homological_dimension)
                draw_plot(n_edges, n_signed_bars, 'n_edges','n_Hilbert_bars',title_name,fig_name)
            
             
elif data_type=="uniform" or "sphere":
    for dim_point in range(4,max_dim_points):
        print("dim_point=",dim_point)
        max_radius = (dist.pairwise([np.zeros(dim_point),np.ones(dim_point)*pts_range])[0,1]+1)/2

        n_signed_bars, n_simplices, n_vertices = run_experiments_higher_homology(n_batch, dim_point, pts_range, num_points, data_type, n_clusters, min_radius,max_radius, max_degree,min_degree,grid_granularity,max_homological_dimension)
        title_type = data_type + " distributed points in " + str(dim_point) + "D: "
        title_name1 = title_type + "#bars v.s. #simplices, degree "+str(max_homological_dimension)
        title_name2 = title_type + "#bars v.s. #vertices, degree "+str(max_homological_dimension)
        title_name3 = title_type + "#bars v.s. sqrt(#simplices), degree "+str(max_homological_dimension)
        fig_name1 = data_type +"/" + data_type+"_" + str(dim_point) + "D_bars-simplices_degree_"+str(max_homological_dimension)
        fig_name2 = data_type +"/" + data_type+"_" + str(dim_point) + "D_bars-vertices_degree_"+str(max_homological_dimension)
        fig_name3 = data_type +"/" + data_type+"_" + str(dim_point) + "D_bars-sqrt(n_simplices)_degree_"+str(max_homological_dimension)

        n_edges = n_vertices*(n_vertices-1)
        draw_plot(n_simplices, n_signed_bars, 'n_k+2n_(k+1)','n_Hilbert_bars',title_name1,fig_name1)
        draw_plot_ratio(n_simplices, n_signed_bars, 'n_k+2n_(k+1)','n_Hilbert_bars',title_name1,fig_name1)

        #if max_homological_dimension==0:
        #    draw_plot(n_vertices, n_signed_bars, 'n_pts','n_Hilbert_bars',title_name2,fig_name2)
        #    draw_plot_ratio(n_vertices, n_signed_bars, 'n_pts','n_Hilbert_bars',title_name2,fig_name2)
        #elif max_homological_dimension==1:    
        #    title_name = title_type + "#bars v.s. #edges, degree "+str(max_homological_dimension)
        #    fig_name = data_type +"/" + data_type+"_" + str(dim_point) + "D_bars-edges_degree_"+str(max_homological_dimension)
        #    draw_plot(n_edges, n_signed_bars, 'n_edges','n_Hilbert_bars',title_name,fig_name)
        #    draw_plot_ratio(n_edges, n_signed_bars, 'n_edges','n_Hilbert_bars/n_simplicies',title_name,fig_name)




