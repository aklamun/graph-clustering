# -*- coding: utf-8 -*-
"""
Created on Wed Dec 06 17:12:57 2017

@author: aak228
"""

import numpy as np
import networkx as nx
from collections import deque

'''
This code implements approximate personalized pagerank clustering from
"Local Graph Partitioning using PageRank Vectors"
G = simple unweighted undirected graph
'''

def node_degrees(G):
    degrees = np.zeros(len(G.nodes()))
    for i in G.nodes():
        degrees[i] = G.degree(i)
    return degrees

def push(G,u,p,r,a, eps, degrees, queue):
    # u = vertex to perform push on
    # p = distribution
    # r = residual distribution
    pp = p[:]
    rr = r[:]
    pp[u] = p[u] + a*r[u]
    rr[u] = (1-a)*r[u]/2
    assert degrees[u] > 0
    if rr[u]/degrees[u] > eps:
        queue.append(u)
    for i in G.neighbors(u):
        rr[i] = r[i] + (1-a)*r[u]/(2*degrees[u])
        if rr[i]/degrees[i] > eps:
            queue.append(i)
    return pp, rr

def approx_pprank(G,v,a,eps, degrees):
    p = np.zeros(len(G.nodes()))
    r = np.zeros(len(G.nodes()))
    r[v] = 1.
    
    queue = deque([v])
    while len(queue) > 0:
        i = queue.popleft()
        p,r = push(G,i,p,r,a, eps, degrees, queue)
    return p

def sweep(G,p, phi):
    supp_p = [i for i in G.nodes() if p[i]>0]
    degrees = node_degrees(G)
    p_norm = np.divide(p,degrees)
    p_norm_nz = [p_norm[i] for i in supp_p]
    sweep_list = np.argsort(p_norm_nz)[::-1]
    sweep_list = [supp_p[i] for i in sweep_list]
    conductances = [nx.algorithms.cuts.conductance(G, sweep_list[0:i+1]) if nx.algorithms.cuts.volume(G, sweep_list[0:i+1])>0  and nx.algorithms.cuts.volume(G, list(set(G.nodes())-set(sweep_list[0:i+1])))>0 else 1 for i in range(len(supp_p))]
    min_ind = np.argmin(conductances)
    if conductances[min_ind] < phi:
        return sweep_list[0:min_ind]
    else:
        return []

def spread_p(p,k, sweep_list, vol_sweep):
    if k in vol_sweep:
        ind = vol_sweep.index(k)
        S_i = sweep_list[0:ind]
        return np.sum([p[i] for i in S_i])
    else:
        upper_ind = next(x[0] for x in enumerate(vol_sweep) if x[1]>k)
        lower_ind = upper_ind - 1
        upper_val = np.sum([p[i] for i in sweep_list[0:upper_ind]])
        lower_val = np.sum([p[i] for i in sweep_list[0:lower_ind]])
        slope = float(upper_val - lower_val)/(vol_sweep[upper_ind] - vol_sweep[lower_ind])
        return slope*(k-vol_sweep[lower_ind]) + lower_val

def sweep_pagerank(G, p,b,phi):
    supp_p = [i for i in G.nodes() if p[i]>0]
    degrees = node_degrees(G)
    p_norm = np.divide(p,degrees)
    p_norm_nz = [p_norm[i] for i in supp_p]
    sweep_list = np.argsort(p_norm_nz)[::-1]
    sweep_list = [supp_p[i] for i in sweep_list]
    volG = 2*len(G.edges())
    vol_sweep = [nx.algorithms.cuts.volume(G, sweep_list[0:i]) for i in range(len(supp_p))]
#    if spread_p(p, 2**b, sweep_list,vol_sweep) - spread_p(p, 2**(b-1), sweep_list,vol_sweep) > 1./(48*B):
    for i in range(len(supp_p)):
        S_i = sweep_list[0:i]
        if vol_sweep[i] > 2**(b-1) and vol_sweep[i] < 2/3*volG:
            cond = nx.algorithms.cuts.conductance(G, S_i)
            if cond < phi:
                return S_i
#    else:
    return []

def pagerank_nibble(G, v, phi, b, degrees):
    m = len(G.edges())
    B = np.ceil(np.log2(m))
    if phi < 0 or phi > 1 or b < 1 or b > B:
        raise Exception("PageRank-Nibble: phi or b outside range")
    
    a = phi**2/(225*np.log(100*np.sqrt(m)))
    p = approx_pprank(G,v,a, 2**(-b)/(48*B), degrees)
    #return sweep_pagerank(p,b,phi)
    return sweep(G,p,phi)

def stationary_distribution(G, degrees):
    #set up random selection of nodes by binning
    m = nx.number_of_edges(G)
    bin_lens = [degrees[i]/(2.*m) for i in G.nodes()]
    bins = np.zeros(len(G.nodes))
    bins[0] = bin_lens[0]
    for i in range(1,len(bins)):
        bins[i] = bin_lens[i] + bins[i-1]
    bins[-1] = 1.
    return bins

def b_distribution(m):
    b_vals = [i+1 for i in range(int(np.ceil(np.log2(m))))]
    bin_lens = [2.**(-i)/(1-2**(-np.ceil(np.log2(m)))) for i in b_vals]
    bins = np.zeros(len(b_vals))
    try:
        bins[0] = bin_lens[0]
    except:
        print m, len(bins), len(bin_lens)
    for i in range(1,len(bins)):
        bins[i] = bin_lens[i] + bins[i-1]
    bins[-1] = 1.
    return bins

def random_pagerank_nibble(G, phi, bins_node, bins_b, degrees):
    #bins_node = stationary distribution for selecting nodes
    #bins_b = distribution for selecting b
    rand1,rand2 = np.random.uniform(0,1,2)
    #find which bin number falls into
    ind = next(i for i,x in enumerate(bins_node) if x >= rand1)
    v = list(G.nodes())[ind]
    b = next(i for i,x in enumerate(bins_b) if x >= rand2) + 1
    return pagerank_nibble(G,v, phi, b, degrees)

def remove_zero_degree_nodes(G_in):
    degrees = nx.degree(G_in)
    z_degree_nodes = [i[0] for i in degrees if i[1] == 0]
    if len(z_degree_nodes) > 0:
        G = G_in.copy()
        G.remove_nodes_from(z_degree_nodes)
        return G
    else:
        return G_in

def pagerank_partition(G_in, phi, prob):
    #note: every node is assumed to have positive degree
    G = remove_zero_degree_nodes(G_in)
    nodes_in = list(G.nodes())
    G = nx.convert_node_labels_to_integers(G)
    
    assert len(G.nodes())>0
    m = nx.number_of_edges(G)
    assert m > 1
    degrees = node_degrees(G)
    assert np.min(degrees)>0
    bins_node = stationary_distribution(G, degrees)
    bins_b = b_distribution(m)
    
    #set up the first W for the iterations
    W = list(G.nodes())
    H = G.subgraph(W)
    H = remove_zero_degree_nodes(H)
    H_nodes = list(H.nodes())
    H = nx.convert_node_labels_to_integers(H)
    H_degrees = node_degrees(H)
    for j in range(56*m*int(np.ceil(np.log10(1./prob)))):
        D = random_pagerank_nibble(H, phi, bins_node, bins_b, H_degrees)
        D = [H_nodes[i] for i in D]
        if len(D)>0:
            #update W
            W = list(set(W) - set(D))
            if nx.algorithms.cuts.volume(G.subgraph(W), W) <= 5./6*2*m:
                D = list(set(list(G.nodes())) - set(W))
                D_out = [nodes_in[i] for i in D]
                return D_out
            H = G.subgraph(W)
            H = remove_zero_degree_nodes(H)
            m = nx.number_of_edges(H)
            if H.number_of_nodes()==0 or m<2:
                break
            H_nodes = list(H.nodes())
            H = nx.convert_node_labels_to_integers(H)
            H_degrees = node_degrees(H)
            bins_node = stationary_distribution(H,H_degrees)
            bins_b = b_distribution(m)
    return []

def k_multi_partition(G, phi, prob, k):
    # Find k-partition; this formulation may never terminate
    phi = float(phi)
    prob = float(prob)
    clusters = deque([list(G.nodes())])
    while len(clusters) < k:
        S = clusters.popleft()
        D = pagerank_partition(G.subgraph(S),phi,prob)
        if len(D) > 0:
            clusters.append(D)
            clusters.append(list(set(S)-set(D)))
        else:
            clusters.append(S)
    return list(clusters)

def multi_partition(G, phi, prob):
    '''Implements MultiwayPartition algorithm in
    "Nearly-linear time algorithms for graph partitioning, graph sparsification,
    and solving linear systems" '''
    
    phi = float(phi)
    prob = float(prob)
    m = nx.number_of_edges(G)
    eps = np.minimum(1.0/16, 1.0/(4.0*np.ceil(np.log10(m))))
    stop = (np.ceil(np.log10(2.0/eps))*np.ceil(np.log10(m))*
            np.ceil(np.log(m)/np.log(17.0/16)))
    C = [list(G.nodes())]
    for t in range(int(stop)):
        CC = []
        for S in C:
            D = pagerank_partition(G.subgraph(S), phi, prob/m)
            if len(D) > 0:
                CC = CC + [D, list(set(S)-set(D))]
            else:
                CC = CC + [S]
        C = CC[:]
    return C

def k_multi_partition2(G, phi, prob, k):
    '''Implements MultiwayPartition algorithm in
    "Nearly-linear time algorithms for graph partitioning, graph sparsification,
    and solving linear systems"
    Stops after k clusters recovered'''
    
    phi = float(phi)
    prob = float(prob)
    m = nx.number_of_edges(G)
    eps = np.minimum(1.0/16, 1.0/(4.0*np.ceil(np.log10(m))))
    stop = (np.ceil(np.log10(2.0/eps))*np.ceil(np.log10(m))*
            np.ceil(np.log(m)/np.log(17.0/16)))
    C = [list(G.nodes())]
    for t in range(int(stop)):
        CC = []
        for S in C:
            D = pagerank_partition(G.subgraph(S), phi, prob/m)
            if len(D) > 0:
                CC = CC + [D, list(set(S)-set(D))]
            else:
                CC = CC + [S]
            i = C.index(S)
            if len(CC) + len(C[i+1:]) == k:
                return C[i+1:] + CC
        C = CC[:]
    return C







