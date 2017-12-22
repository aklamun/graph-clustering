import numpy as np
import scipy
import scipy.cluster
import networkx as nx


def spec_clust(G, n_clust):
    """
Created on Sat Nov 04 22:33:41 2017
Computes eigenvector embedding of a graph and performs spectral clustering via
k-means.
@author: ajh326
    """
    L = nx.normalized_laplacian_matrix(G)
    d = np.array(nx.degree(G))
    D = np.sqrt(d[:, 1])
    vals, pre_scaled_vecs = scipy.sparse.linalg.eigsh(L, k=n_clust+1, which='SM')
    vecs = np.array([np.divide(pre_scaled_vecs[:,i], D) for i in range(1, n_clust+1)])
#    vecs = pre_scaled_vecs[:,1:(n_clust+1)]
#    vecs = [vecs[i, :]/np.linalg.norm(vecs[i, :]) for i in range(len(vecs[:,1]))]
    codebook, dist = scipy.cluster.vq.kmeans(np.transpose(vecs), n_clust)
    clust, dist = scipy.cluster.vq.vq(np.transpose(vecs), codebook)
    
    return clust

def code2clust(code):
    k = max(code)+1
    clust = []
    for i in range(k):
        clust.append([j for j in range(len(code)) if code[j] == i])
    return clust
