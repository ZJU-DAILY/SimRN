# by dlhu, 07/2024
# compute the distance matrix of the road network

import networkx as nx
import pandas as pd
import numpy as np

edges_df = pd.read_csv('./data/tdrive/road/edge_weight.csv') 

G = nx.DiGraph()  

for _, row in edges_df.iterrows():
    G.add_edge(row['s_node'], row['e_node'], weight=row['length'])

nodes = list(G.nodes())  
distances = np.full((len(nodes), len(nodes)), np.inf) 

for i in range(len(nodes)):
    for j in range(i+1, len(nodes)):
        try:
            distances[i, j] = nx.dijkstra_path_length(G, nodes[i], nodes[j], weight='weight')
            distances[j, i] = distances[i, j]
        except nx.NetworkXNoPath:
            continue

np.save('ground_truth/tdrive/Point_dis_matrix.npy', distances)