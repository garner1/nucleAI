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


frequency = int(sys.argv[1]) # how often to pick a nuclei as a seed = size of the covd sample nuclei
n_neighbors = int(sys.argv[2]) # the number of nuclei in each descriptor
dirpath = sys.argv[3] # the full path to the sample directory


# In[ ]:


sample = os.path.basename(dirpath).split(sep='.')[0]; print(sample)

print('Loading the data')
df = pd.DataFrame()
fovs = glob.glob(dirpath+'/*_polygon/*.svs/*.csv.morphometrics.pkl')

print('There are '+str(len(fovs))+' FOVs')
for fov in fovs: # for each fov
    data = pd.read_pickle(fov)
    df = df.append(data, ignore_index = True)
    
df = df[df['area']>10] # filter out small nuclei
df = df[df['perimeter']>0] # make sure perimeter is always positive

df['area'] = df['area'].astype(float) # convert to float this field
df['circularity'] = 4.0*np.pi*df['area'] / (df['perimeter']*df['perimeter']) # add circularity

numb_nuclei = df.shape[0] 
print(str(numb_nuclei)+' nuclei')

size = numb_nuclei//frequency
fdf = df.sample(n=size,random_state=1234) #!!!hard-coded random state 
print('We consider '+str(size)+' descriptors')

centroids = df.columns[:2];# print(centroids)
X = df[centroids].to_numpy() # the full array of position

print('Characterizing the neighborhood')
nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree',n_jobs=-1).fit(X) 
distances, indices = nbrs.kneighbors(X) 

# Parallel generation of the local covd
data = df.to_numpy(dtype=np.float64)

# from scipy import stats
# rescaled_data = stats.zscore(data[:,2:]) # rescale morphometric data by mean and std
# data[:,2:] = rescaled_data # update data morphometrics

s1, s2 = data[indices[fdf.index[0],:],:].shape
tensor = np.empty((size,s1,s2))

for node in tqdm(range(size)):
    tensor[node,:,:] = data[indices[node,:],:]

print('Generating the descriptor')
num_cores = multiprocessing.cpu_count() # numb of cores
node_vec_switch_entropy = Parallel(n_jobs=num_cores)(
    delayed(covd_parallel)(node,tensor) for node in tqdm(range(size))
    )

nodes_with_covd = [fdf.index[l[0]] for l in node_vec_switch_entropy if l[2] == 1] # list of nodes with proper covd
nodes_wo_covd = [fdf.index[l[0]] for l in node_vec_switch_entropy if l[2] == 0] # list of nodes wo covd
fdf['covd'] = [0 for i in range(size)]
fdf.loc[nodes_with_covd,'covd'] = 1 # identify nodes with covd in dataframe
fdf.loc[nodes_wo_covd,'covd'] = 0 # identify nodes wo covd in dataframe    
print('There are '+str(len(nodes_with_covd))+' nodes with covd properly defined')

# Add the entropy feature to fdf
fdf["entropy"] = ""; fdf["entropy"].astype(object)
for item in node_vec_switch_entropy:
    entropy = item[3]
    node = fdf.index[item[0]]
    fdf.at[node,'entropy'] = entropy

    # Add the descriptor feature to fdf
fdf["descriptor"] = ""; fdf["descriptor"].astype(object)
for item in node_vec_switch_entropy:
    descriptor = item[1]
    node = fdf.index[item[0]]
    fdf.at[node,'descriptor'] = pd.Series(descriptor).values

# Generate the descriptor array
descriptor = np.zeros((len(nodes_with_covd),node_vec_switch_entropy[0][1].shape[0]))
r_idx = 0
for index, row in fdf.iterrows():
    if row['covd']:
        descriptor[r_idx,:] = row['descriptor']
        r_idx += 1

mean_covd = np.mean(descriptor,axis=0) # evaluate the barycenter descriptor

delta = descriptor-mean_covd # evaluate the distance vec of the barycenter from each descriptor
distance_from_barycenter = norm(delta,axis=1) # take the eucledean norm

# Update the dataframe
fdf.loc[nodes_with_covd,'heterogeneity'] = distance_from_barycenter
fdf.loc[nodes_wo_covd,'heterogeneity'] = np.nan


# In[ ]:


filename = dirpath+'/'+sample+'.nuclei'+str(numb_nuclei)+'.size'+str(size)+'.covdNN'+str(n_neighbors)+'.features.pkl'
fdf.to_pickle(filename)




