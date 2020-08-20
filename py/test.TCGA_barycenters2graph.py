#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys  
sys.path.insert(0, '../py')
from graviti import *

import numpy as np
import os
import os.path
from os import path
import sys
import glob
import h5py
import pandas as pd
import pickle
import timeit
import multiprocessing
from joblib import Parallel, delayed
from datetime import datetime
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

import umap
import networkx as nx

# In[33]:

# samples = glob.glob('/media/garner1/hdd2/TCGA_polygons/*/*/*.freq10.covdNN50.features.pkl')
# num_cores = multiprocessing.cpu_count() # numb of cores

# Load the covd-barycenters for all samples
outfile = 'covd_barycenters.npy'
barycenters = np.load(outfile)
print(barycenters.shape)

#Generate the topological graph adjacency matrix A
print('Generating the graph')
A = umap.umap_.fuzzy_simplicial_set(
    barycenters,
    n_neighbors=10, 
    random_state=np.random.RandomState(seed=42),
    metric='l2',
    metric_kwds={},
    knn_indices=None,
    knn_dists=None,
    angular=False,
    set_op_mix_ratio=1.0,
    local_connectivity=2.0,
    verbose=False
    )
print('Done')
G = nx.from_scipy_sparse_matrix(A)
edges = G.edges()
weights = [G[u][v]['weight'] for u,v in edges]
pos=nx.spring_layout(G)
nx.draw(G, pos, edges=edges, width=weights,node_size=1)

# eset = [(u, v) for (u, v, d) in G.edges(data=True)]
# weights = [d['weight'] for (u, v, d) in G.edges(data=True)]

# nx.draw_networkx_nodes(G,pos,node_color='green',node_size=1)

# unique_weights = list(set(weights))
# for weight in unique_weights:
#         weighted_edges = [(node1,node2) for (node1,node2,edge_attr) in G.edges(data=True) if edge_attr['weight']==weight]
#         width = weight
#         nx.draw_networkx_edges(G,pos,edgelist=weighted_edges,width=width)
        
plt.savefig("graph.png", format="PNG")
