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
k = 2
n = 50
qlist = np.arange(0.2, 0.5, 0.05)
sample_size = 5
f1_scores = np.zeros([sample_size, len(qlist)])
f1_scorep = np.zeros([sample_size, len(qlist)])
for l in range(sample_size):
    f1_trials = []
    f1_trialp = []
    for q in qlist:
        # mu = (k-1)*q/(p+(k-1)*q)
        A = create_planted_partition(q, n, k)
        G = create_nx_graph(range(n*k), A)
        scluster_codes = spec_clust(G, k)
        sclusters = code2clust(scluster_codes)
        pclusters = k_multi_partition(G, np.max([.5, 2*q]), .9, k)
        v = list(G.nodes())
        tclusters = [v[(n*i):(n*(i+1))] for i in range(k)]
        f1_trials.append(avg_f1_score(G, sclusters, tclusters))
        f1_trialp.append(avg_f1_score(G, pclusters, tclusters))
    f1_scores[l, :] = f1_trials
    f1_scorep[l, :] = f1_trialp
f1_mean_scores = []
f1_mean_scorep = []
for l in range(len(qlist)):
    f1_mean_scores.append(np.mean(f1_scores[:, l]))
    f1_mean_scorep.append(np.mean(f1_scorep[:, l]))
# Figures: Plots of f1_scores for each q-experiment
fig = plt.figure()
eig_embed, = plt.plot(qlist, f1_mean_scores, 'b', label="EigEmbed")
pprank, = plt.plot(qlist, f1_mean_scorep, 'r', label="PPRank")
plt.legend(handles=[eig_embed, pprank])
fig.suptitle('F1 Scores', fontsize=20)
plt.xlabel('q', fontsize=18)
plt.ylabel('F1 Score', fontsize=16)
fig.savefig('4b_p.8_1.eps')
