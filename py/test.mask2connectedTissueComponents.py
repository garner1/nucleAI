#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import glob
import pickle
import sys  

sys.path.insert(0, '../py')
from graviti import *

from numpy.linalg import norm
import numpy as np

import pandas as pd

import timeit
import multiprocessing
from joblib import Parallel, delayed
from datetime import datetime
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors

# In[ ]:


dirpath = sys.argv[1] # the full path to the sample directory


# In[ ]:


sample = os.path.basename(dirpath).split(sep='.')[0]; print(sample)

print('Loading the data')
df = pd.DataFrame()
fovs = glob.glob(dirpath+'/*_polygon/*.svs/*.csv.morphometrics.pkl')

print('There are '+str(len(fovs))+' FOVs')
for fov in fovs: # for each fov
    data = pd.read_pickle(fov)
    df = df.append(data, ignore_index = True)

df['area'] = df['area'].astype(float) # convert to float this field
df['circularity'] = 4.0*np.pi*df['area'] / (df['perimeter']*df['perimeter']) # add circularity

numb_nuclei = df.shape[0] 
print(str(numb_nuclei)+' nuclei')

centroids = df.columns[:2];# print(centroids)
pos = df[centroids].to_numpy(dtype='float') # Get the positions of centroids 

X = df[centroids].to_numpy() # the full array of position

print('Generating the UMAP graph')
A = space2graph(pos[:,:],3)
print('Done')
import networkx as nx
G = nx.from_scipy_sparse_matrix(A) # define the networkx obj G

#nodes = nx.draw_networkx_nodes(G, pos[:,:],node_size=1)
#plt.savefig('test.png')

print('Determining the connected components')
S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
glabel = 0
print(pos.shape)
for g in S:
    if len(list(g)) > 1000:
        nodes = nx.draw_networkx_nodes(g, pos, node_size=1)
        plt.savefig('component_'+str(glabel)+'.png')
        glabel += 1
    
print([len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True) if len(c)>1000]) 
print('Done')
