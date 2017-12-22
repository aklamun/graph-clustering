'''
This script evaluates the quality performance of clustering algorithms
by comparing recovered cluster conductances and ground truth cluster
conductances
'''


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# k=number of communities
# n=size of community
k = 3
n = 50
qlist = np.arange(0.2, 0.5, 0.1)
sample_size = 5
mcondvar=np.zeros([sample_size,len(qlist)])
smcondvar=np.zeros([sample_size,len(qlist)])
for l in range(sample_size):
    condvar = []
    scondvar = []
    for q in qlist:
        A = create_planted_partition(q, n, k)
        G = create_nx_graph(range(n*k), A)
        # scluster_code = spec_clust(G, k)
        # scluster = code2clust(scluster_code)
        scluster = k_multi_partition(G, np.max([.5, 2*q]), .9, k)
        # Conductance between TRUE clusters u_i and complement in G
        cond = np.zeros(k)
        for i in range(0, k):
            v = list(G.nodes())
            u = v[(n*i):(n*(i+1))]
            cond[i] = nx.algorithms.cuts.conductance(G, G.subgraph(u))
        condvar.append(np.mean(cond))
        # Conductance between computed clusters u_i and complement in G
        scond = np.zeros(k)
        scond = []
        for elem in scluster:
            scond.append(nx.algorithms.cuts.conductance(G, G.subgraph(elem)))
        scondvar.append(np.mean(scond))
    mcondvar[l, :] = condvar
    smcondvar[l, :] = scondvar
mcv = []
smcv = []
for l in range(len(qlist)):
    mcv.append(np.mean(mcondvar[:, l]))
    smcv.append(np.mean(smcondvar[:, l]))
# Plot conductance stats
fig = plt.figure()
eig_embed, = plt.plot(qlist, mcv, 'b', label="PPRank")
pprank, = plt.plot(qlist, smcv, 'r', label="Ground Truth")
plt.legend(handles=[eig_embed, pprank])
fig.suptitle('Mean Cluster Conductance', fontsize=20)
plt.xlabel('q', fontsize=18)
plt.ylabel('Conductance', fontsize=16)
fig.savefig('Cond_3b_pprank_1.eps')
