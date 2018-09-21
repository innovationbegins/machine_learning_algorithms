# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 15:07:45 2017

@author: righttodevelopteam

"""
from __future__ import print_function
from sklearn.metrics import pairwise_distances
import numpy as np


def get_initial_centroids(data, k, seed=None):
    '''Randomly choose k data points as initial centroids'''
    if seed is not None:
#useful for obtaining consistent results.
        np.random.seed(seed)
    n = data.shape[0] 
#number of data points
        
#Pick K indices from range [0, N).
    rand_indices = np.random.randint(0, n, k)
    centroids = data[rand_indices,:]
    
    return centroids

def centroid_pairwise_dist(X,centroids):
    return pairwise_distances(X,centroids,metric='euclidean')

def assign_clusters(data, centroids):
    
#Compute distances between each data point and the set of centroids.

    distances_from_centroids = centroid_pairwise_dist(data,centroids)
    
#Compute cluster assignments for each data point.
    
    cluster_assignment = np.argmin(distances_from_centroids,axis=1)
    
    return cluster_assignment

def revise_centroids(data, k, cluster_assignment):
    new_centroids = []
    for i in range(k):
#Select all data points that belong to cluster i.
        member_data_points = data[cluster_assignment==i]
#Compute the mean of the data points.
        centroid = member_data_points.mean(axis=0)
        new_centroids.append(centroid)
    new_centroids = np.array(new_centroids)
    
    return new_centroids

def compute_heterogeneity(data, k, centroids, cluster_assignment):
    
    heterogeneity = 0.0
    for i in range(k):
        
#Select all data points that belong to cluster i. 
        
        member_data_points = data[cluster_assignment==i, :]
        
        if member_data_points.shape[0] > 0: #check if i-th cluster is non-empty.
#Compute distances from centroid to data points.
            distances = pairwise_distances(member_data_points, [centroids[i]], metric='euclidean')
            squared_distances = distances**2
            heterogeneity += np.sum(squared_distances)
        
    return heterogeneity

from matplotlib import pyplot as plt
def plot_heterogeneity(heterogeneity, k):
    plt.figure(figsize=(7,4))
    plt.plot(heterogeneity, linewidth=4)
    plt.xlabel('#Iterations')
    plt.ylabel('Heterogeneity')
    plt.title('Heterogeneity of clustering over time, K={0:d}'.format(k))
    plt.rcParams.update({'font.size': 16})
    plt.show()

def kmeans(data, k, initial_centroids, maxiter=500, record_heterogeneity=None, verbose=False):

#This function runs k-means on given data and initial set of centroids.
#maxiter: maximum number of iterations to run.(default set to=500).
#record_heterogeneity: (optional) a list, to store the history of heterogeneity as function of iterations.
                             
    centroids = initial_centroids[:]
    prev_cluster_assignment = None
    
    for itr in range(maxiter):        
        if verbose:
            print(itr, end='')
        
#Make cluster assignments using nearest centroids.
        cluster_assignment = assign_clusters(data,centroids)
            
#Compute a new centroid for each of the k clusters, averaging all data points assigned to that cluster.
        
        centroids = revise_centroids(data,k, cluster_assignment)
            
#Check for convergence.
        if prev_cluster_assignment is not None and \
          (prev_cluster_assignment==cluster_assignment).all():
            break
        
        # Print number of new assignments 
        if prev_cluster_assignment is not None:
            num_changed = np.sum(prev_cluster_assignment!=cluster_assignment)
            if verbose:
                print('    {0:5d} elements changed their cluster assignment.'.format(num_changed))   
        
        # Record heterogeneity convergence metric
        if record_heterogeneity is not None:
            # YOUR CODE HERE
            score = compute_heterogeneity(data,k,centroids,cluster_assignment)
            record_heterogeneity.append(score)
        
        prev_cluster_assignment = cluster_assignment[:]
        
    return centroids, cluster_assignment


if False: 
#change to true to run this test case.
#using iris_dataset
    import sklearn.datasets as ds
    dataset = ds.load_iris()
    k = 3
    heterogeneity = []
    initial_centroids = get_initial_centroids(dataset['data'], k, seed=0)
    centroids, cluster_assignment = kmeans(dataset['data'], k, initial_centroids, maxiter=400,
                                        record_heterogeneity=heterogeneity, verbose=True)
    plot_heterogeneity(heterogeneity, k)

