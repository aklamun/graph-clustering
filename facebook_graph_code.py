# -*- coding: utf-8 -*-
"""
Created on Fri Dec 08 22:18:26 2017

@author: aak228
"""

'''
partition = pagerank_partition(G,0.5,0.9)
partition2 = list(set(range(100))-set(partition))
avg_f1_score(G,[partition,partition2],[range(50),range(50,100)])
partitions = k_multi_partition(G, 0.5, 0.9, 2)
avg_f1_score(G,partitions,[range(50),range(50,100)])
'''

F = nx.read_gexf('AK_facebook_graph.gexf')
F = F.to_undirected()
degrees = nx.degree(F)
z_degree_nodes = [i[0] for i in degrees if i[1] == 0]
F.remove_nodes_from(z_degree_nodes)
#G = nx.convert_node_labels_to_integers(G_in)

F_nodes = list(F.nodes())
F_int = nx.convert_node_labels_to_integers(F)


partitions = k_multi_partition2(F, 0.5, 0.8, 6)
partitions_spec = spec_clust(F_int,6)
partitions_spec = code2clust(partitions_spec)
partitions_spec2 = spec_clust(F_int,10)
partitions_spec2 = code2clust(partitions_spec2)

pr_clusters = {}
spec_clusters = {}
spec_clusters2 = {}
for node in F.nodes():
    for i in range(len(partitions)):
        if node in partitions[i]:
            pr_clusters[node] = i
            break

for ind in F_int.nodes():
    node = F_nodes[ind]
    for i in range(len(partitions_spec)):
        if ind in partitions_spec[i]:
            spec_clusters[node] = i
            break

for ind in F_int.nodes():
    node = F_nodes[ind]
    for i in range(len(partitions_spec2)):
        if ind in partitions_spec2[i]:
            spec_clusters2[node] = i
            break

nx.set_node_attributes(F,pr_clusters,'pr_cluster')
nx.set_node_attributes(F,spec_clusters,'spec_cluster')
nx.set_node_attributes(F,spec_clusters2,'spec_cluster_more')
nx.write_gexf(F,'AK_facebook_graph_clustered.gexf')

#create anonymized version of Facebook graph
FF = nx.Graph()
FF_nodes = list(F_int.nodes())
FF_edges = list(F_int.edges())
FF.add_nodes_from(FF_nodes)
FF.add_edges_from(FF_edges)
nx.write_gexf(FF,'facebook_graph_anonymized.gexf')
