# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 13:09:11 2017

@author: aak228
"""

n_set = [50,100,150,200,250,300,350,400,450,500]
trials = 10

tree_pr_comp_time = {}
tree_spec_comp_time = {}
for n in n_set:
    tree_pr_comp_time[n] = 0
    tree_spec_comp_time[n] = 0

phi = 0.5
for n in n_set:
    for t in range(trials):
        H = nx.random_powerlaw_tree(n,3,tries=1000000)
        start_time = time.time()
        partitions_spec = spec_clust(H,2)
        partitions_spec = code2clust(partitions_spec)
        tree_spec_comp_time[n] = tree_spec_comp_time[n] + time.time() - start_time
        
        start_time = time.time()
        partition = k_multi_partition2(H, phi, 0.8, 2)
        tree_pr_comp_time[n] = tree_pr_comp_time[n] + time.time() - start_time

for n in n_set:
    tree_spec_comp_time[n] = tree_spec_comp_time[n]/trials
    tree_pr_comp_time[n] = tree_pr_comp_time[n]/trials

plt.plot(n_set, [tree_spec_comp_time[i] for i in n_set], '-', linewidth=2,label='Spectral')
plt.plot(n_set, [tree_pr_comp_time[i] for i in n_set], '-.', linewidth=2, label='PageRank')
plt.title('Computation Time in Trees', fontsize=15)
plt.xlabel('Number Nodes', fontsize=15)
plt.ylabel('Avg Computation Time', fontsize=15)
plt.legend(loc='upper left')
plt.savefig('trees_compute_time.eps')
plt.show()
plt.close()

##add two trees together:
tree2_pr_comp_time = {}
tree2_spec_comp_time = {}
for n in n_set:
    tree2_pr_comp_time[n] = 0
    tree2_spec_comp_time[n] = 0

phi = 0.5
for n in n_set:
    for t in range(trials):
        H = nx.random_powerlaw_tree(n,3,tries=1000000)
        H2 = nx.random_powerlaw_tree(n,3,tries=1000000)
        H.add_edges_from(H2.edges())
        start_time = time.time()
        partitions_spec = spec_clust(H,2)
        partitions_spec = code2clust(partitions_spec)
        tree2_spec_comp_time[n] = tree2_spec_comp_time[n] + time.time() - start_time
        
        start_time = time.time()
        partition = k_multi_partition2(H, phi, 0.8, 2)
        tree2_pr_comp_time[n] = tree2_pr_comp_time[n] + time.time() - start_time

for n in n_set:
    tree2_spec_comp_time[n] = tree2_spec_comp_time[n]/trials
    tree2_pr_comp_time[n] = tree2_pr_comp_time[n]/trials

plt.plot(n_set, [tree2_spec_comp_time[i] for i in n_set], '-', linewidth=2,label='Spectral')
plt.plot(n_set, [tree2_pr_comp_time[i] for i in n_set], '-.', linewidth=2, label='PageRank')
plt.title('Computation Time in Sparse Graphs', fontsize=15)
plt.xlabel('Number Nodes', fontsize=15)
plt.ylabel('Avg Computation Time', fontsize=15)
plt.legend(loc='upper left')
plt.savefig('trees2_compute_time.eps')
plt.show()
plt.close()