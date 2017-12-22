# -*- coding: utf-8 -*-
"""
Created on Wed Nov 08 18:31:41 2017

@author: aak228
"""

'''
Planted Parition (stochastic block) model: generate undirected unweighted graph with kn nodes
k built-in communities, each of size n
between any pair of nodes from the same community, edge exists with probability p
between any pair of nodes from different communities, edge exists with probabiliy q
fix n=50, k=10, p=0.5
take values of q s.t. mu = [(k-1)q]/[p+(k-1)q] varies from 0.1 to 0.9
mu measures community mixing level
these parameters come from Leskovec Local Higher-Order Graph Clustering paper
'''

import snap
import numpy as np
import networkx as nx
from sklearn.metrics import f1_score

def create_planted_partition(q, n=50, k=10, p=0.5):
    #returns adjacency matrix
    #G = snap.TUNGraph.New()
    A = np.random.binomial(1,q,(k*n,k*n))
    for i in range(k):
        A[i*n:(i+1)*n, i*n:(i+1)*n] = np.random.binomial(1,p,(n,n))
    A = A - np.diag(np.diag(A)) #no diagonal entries in graph
    A = np.triu(A) #want undirected (symmetric) graph
    A = A + np.transpose(A)
    return A

def create_nx_graph(nodes, A):
    G = nx.Graph()
    for i in range(len(nodes)):
        G.add_node(i)
    for i in range(len(nodes)):
        for j in range(i+1,len(nodes)):
            if A[i,j] > 0:
                G.add_edge(i,j)
    return G

def nx2snap(G):
    H = snap.TUNGraph.New()
    for node in G.nodes():
        H.AddNode(node)
    for edge in G.edges():
        H.AddEdge(edge[0],edge[1])
    return H

def avg_f1_score(G, clusters, true_clusters):
    n = len(G.nodes())
    k = len(clusters)
    assert k == len(true_clusters)
    final_scores = []
    for clus in clusters:
        c_assign = [1 if i in clus else 0 for i in range(1,n+1)]
        pre_scores = []
        for tclus in true_clusters:
            t_assign = [1 if i in tclus else 0 for i in range(1,n+1)]
            pre_scores.append(f1_score(t_assign,c_assign))
        final_scores.append(np.max(pre_scores))
    return np.average(final_scores)
