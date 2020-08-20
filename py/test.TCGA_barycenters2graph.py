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

import warnings
warnings.filterwarnings('ignore')

import umap
import networkx as nx


# Load the covd-barycenters for all samples
outfile = 'covd_barycenters.npy'
barycenters = np.load(outfile)
print(barycenters.shape)

# Load sample information
samples = glob.glob('/media/garner1/hdd2/TCGA_polygons/*/*/*.freq10.covdNN50.features.pkl')
cancer_type = []
sample_id = []
for sample in samples:
    cancer_type.append( sample.split('/')[5] )
    sample_id.append( os.path.basename(sample).split('.')[0] )

# Map cancer type strings to integers
mydict={}
i = 0
for item in cancer_type:
    if(i>0 and item in mydict):
        continue
    else:    
       i = i+1
       mydict[item] = i

k=[]
for item in cancer_type:
    k.append(mydict[item])

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
attrs = { i : cancer_type[i] for i in range(0, len(cancer_type) ) }
print('Drawing the graph')
G = nx.from_scipy_sparse_matrix(A)
nx.set_node_attributes(G, attrs,'labels') # setting node labels with cancer type
edges = G.edges()
weights = [G[u][v]['weight'] for u,v in edges]
pos=nx.spring_layout(G)
cmap=plt.cm.tab10
vmin = min(k)
vmax = max(k)
nx.draw_networkx(G, pos, edges=edges, node_color=k,
                 cmap=cmap, vmin=vmin, vmax=vmax,
                 width=weights, node_size=1,
                 labels=nx.get_node_attributes(G, 'labels') )

# Generating a color bar
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
# sm.set_array([])
# cbar = plt.colorbar(sm,orientation='horizontal',shrink=0.75)
# cbar.ax.set_yticklabels(list(set(cancer_type)))
plt.legend(scatterpoints=1)
plt.savefig("graph.png", format="PNG")
print('Done')
