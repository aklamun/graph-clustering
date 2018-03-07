# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 02:02:09 2017

@author: aak228
"""

import time
import matplotlib.pyplot as plt

k_set = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
q = 0.2
n = 50
trials = 20
#avg_pr_comp_time = {}
avg_spec_comp_time = {}
for k in k_set:
    #avg_pr_comp_time[k] = 0
    avg_spec_comp_time[k] = 0

for k in k_set:
    mu = (k-1)*q/(0.5+(k-1)*q)
    phi = np.max([0.5,mu*1.2])
    for t in range(trials):
        A = create_planted_partition(q,n,k)
        H = create_nx_graph(range(k*n),A)
        start_time = time.time()
        partitions_spec = spec_clust(H,2)
        partitions_spec = code2clust(partitions_spec)
        avg_spec_comp_time[k] = avg_spec_comp_time[k] + time.time() - start_time
        
        '''
        start_time = time.time()
        partition = k_multi_partition2(H, phi, 0.8, 2)
        avg_pr_comp_time[k] = avg_pr_comp_time[k] + time.time() - start_time
        '''

for k in k_set:
    avg_spec_comp_time[k] = avg_spec_comp_time[k]/trials
    #avg_pr_comp_time[k] = avg_pr_comp_time[k]/trials

'''
plt.plot(avg_spec_comp_time.keys(), avg_spec_comp_time.values(), '-', linewidth=2)
plt.title('Spectral Clustering Computation Time in Planted Partition', fontsize=15)
plt.xlabel('k', fontsize=15)
plt.ylabel('Avg Computation Time', fontsize=15)
plt.savefig('planted_partition_compute_time_spec.eps')
plt.show()
plt.close()
'''
plt.plot(avg_pr_comp_time.keys(), avg_pr_comp_time.values(), '.-', linewidth=2)
plt.title('PageRank Clustering Computation Time in Planted Partition', fontsize=15)
plt.xlabel('k', fontsize=15)
plt.ylabel('Average Computation Time', fontsize=15)
plt.savefig('planted_partition_compute_time_pr.eps')
plt.show()
plt.close()


        