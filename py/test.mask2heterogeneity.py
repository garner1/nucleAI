#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys  
sys.path.insert(0, '../py')
from graviti import *

from numpy.linalg import norm
import numpy as np
import os
import os.path
from os import path
import sys
import glob
import h5py
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
#matplotlib.use('Agg')
import plotly.graph_objects as go
from plotly.graph_objs import *
import plotly.express as px
import hdbscan
import pandas as pd
import umap
import networkx as nx
from scipy import sparse, linalg
import pickle
from sklearn.preprocessing import normalize, scale
from scipy.sparse import find
from numpy.linalg import norm
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


size = int(sys.argv[1]) # number of nuclei to sample, use 0 value for full set
nn = 10 # set the number of nearest neighbor in the umap-graph. Will be used in CovD as well


# In[ ]:


samples = glob.glob('../data/TCGA*.gz')    


# In[ ]:


# Get numb of cores
num_cores = multiprocessing.cpu_count() # numb of cores


# In[ ]:

dirpath = sys.argv[2]
sample = os.path.basename(dirpath).split(sep='.')[0]; print(sample)

print('Loading the data')
df = pd.DataFrame()
fovs = glob.glob(dirpath+'/*_polygon/*.svs/*.pkl')
for fov in fovs: # for each fov
    data = pd.read_pickle(fov)
    df = df.append(data, ignore_index = True)

df['area'] = df['area'].astype(float) # convert to float this field

#df = df.head(n=100000) # consider smaller df
    
print(str(df.shape[0])+' nuclei')
    
centroids = df.columns[:2];# print(centroids)

if size == 0:
    print('Considering all nuclei')
    fdf = df 
else:
    print('Downsampling '+str(size)+' nuclei')
    fdf = df.sample(n=size) 

print('Creating the UMAP graph')
pos = fdf[centroids].to_numpy() # Get the positions of centroids 
A = space2graph(pos,nn)
print('Characterizing the neighborhood')
X = df[centroids].to_numpy() # the full array of position
if size is not 0:
    n_neighbors = df.shape[0]//size + 10
else:
    n_neighbors = 10    
nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree',n_jobs=-1).fit(X) 
distances, indices = nbrs.kneighbors(X) 

#get the morphological data and rescale the data by std
df['circularity'] = 4.0*np.pi*df['area'] / (df['perimeter']*df['perimeter'])
df['area_rescaled'] = df['area'] / df['area'].mean()
df['perimeter_rescaled'] = df['perimeter'] / df['perimeter'].mean()
features = ['area_rescaled', 'eccentricity', 'orientation','perimeter_rescaled', 'solidity','circularity']

data = df[features].to_numpy(); #print(data.shape)
    
# Parallel generation of the local covd
print('Generating the descriptor')
processed_list = Parallel(n_jobs=num_cores)(
    delayed(covd_parallel_sparse)(node,data,indices) for node in tqdm(list(fdf.index))
)
# store the descriptors
filename = dirpath+'/'+sample+'.size'+str(size)+'.graphNN'+str(nn)+'.covdNN'+str(n_neighbors)+'.descriptors.pkl'
pickle.dump( processed_list, open( filename, "wb" ) )
    
nodes_with_covd = [l[0] for l in processed_list if l[2] == 0] # list of nodes with proper covd
nodes_wo_covd = [l[0] for l in processed_list if l[2] == 1] # list of nodes wo covd
fdf['covd'] = [0 for i in range(fdf.shape[0])]
fdf.loc[nodes_wo_covd,'covd'] = 0 # identify nodes wo covd in dataframe
fdf.loc[nodes_with_covd,'covd'] = 1 # identify nodes with covd in dataframe
    
print('There are '+str(len(nodes_with_covd))+' nodes with covd properly defined')

# Construct the descriptor array
descriptor = np.zeros((len(processed_list),processed_list[0][1].shape[0]),dtype=complex)
for r in range(len(processed_list)):
    descriptor[r,:] = processed_list[r][1] # descriptors of the nuclei communities seeded at sampled nodes

# Get positions in fdf.index of nodes_with_covd
fdf2adj = {value: counter for (counter, value) in enumerate(fdf.index)}
adj2fdf = {v: k for k, v in fdf2adj.items()}

# Get info about the graph
A[[fdf2adj[n] for n in nodes_wo_covd]] = 0 # zero-out nodes with no proper covd 
A[:,[fdf2adj[n] for n in nodes_wo_covd]] = 0 # zero-out nodes with no proper covd
row_idx, col_idx, values = find(A) #A.nonzero() # nonzero entries

print('Generating the heterogeneity metric')
node_nn_heterogeneity_weights = Parallel(n_jobs=num_cores)(
    delayed(covd_gradient_parallel)(fdf2adj[node],descriptor,row_idx,col_idx,values) 
    for node in tqdm(nodes_with_covd)
)
    
# define and store dataframe with pairwise diversities
heterogeneity_df = pd.DataFrame(node_nn_heterogeneity_weights, columns =['node', 'nn', 'heterogeneity', 'weight']) 
filename = dirpath+'/'+sample+'.size'+str(size)+'.graphNN'+str(nn)+'.covdNN'+str(n_neighbors)+'.pairwise_heterogeneity.pkl'
heterogeneity_df.to_pickle(filename)

fdf['heterogeneity'] = np.nan # create a new feature in fdf
for idx in list(fdf.index):
    try:
        fdf.at[idx,'heterogeneity'] = np.sum(heterogeneity_df[heterogeneity_df['node']==fdf2adj[idx]]['heterogeneity'].values[0])
    except:
        pass

# store the node diversity dataframe
filename = dirpath+'/'+sample+'.size'+str(size)+'.graphNN'+str(nn)+'.covdNN'+str(n_neighbors)+'.nodeHI.pkl'
fdf.to_pickle(filename)

print('Generating the edge diversity index and its coordinates')
edges_list = Parallel(n_jobs=num_cores)(
     delayed(edge_diversity_parallel)(adj2fdf[node],[adj2fdf[nn] for nn in neightbors],diversity,fdf) 
     for (node, neightbors, diversity, weights) in tqdm(node_nn_heterogeneity_weights) if adj2fdf[node] in nodes_with_covd
 )
edge_list = [item for sublist in edges_list for item in sublist]
edge_df = pd.DataFrame(edge_list, columns=["centroid_x", "centroid_y","heterogeneity"]) 
    
# store the edge diversity dataframe
filename = dirpath+'/'+sample+'.size'+str(size)+'.graphNN'+str(nn)+'.covdNN'+str(n_neighbors)+'.edgeHI.pkl'
edge_df.to_pickle(filename)

#Show contour plot
N = 100
filename = dirpath+'/'+sample+'.size'+str(size)+'.graphNN'+str(nn)+'.covdNN'+str(n_neighbors)+'.contour.node.mean.png'
contourPlot(fdf,N,np.mean,filename)

#Show contour plot
N = 100
filename = dirpath+'/'+sample+'.size'+str(size)+'.graphNN'+str(nn)+'.covdNN'+str(n_neighbors)+'.contour.edge.mean.png'
contourPlot(edge_df,N,np.mean,filename)

#Show contour plot
N = 100
filename = dirpath+'/'+sample+'.size'+str(size)+'.graphNN'+str(nn)+'.covdNN'+str(n_neighbors)+'.contour.node.sum.png'
contourPlot(fdf,N,np.sum,filename)

#Show contour plot
N = 100
filename = dirpath+'/'+sample+'.size'+str(size)+'.graphNN'+str(nn)+'.covdNN'+str(n_neighbors)+'.contour.edge.sum.png'
contourPlot(edge_df,N,np.sum,filename)

