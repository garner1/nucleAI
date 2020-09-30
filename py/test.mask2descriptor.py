#!/usr/bin/env python
# coding: utf-8
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

frequency = int(sys.argv[1]) # how often to pick a nuclei as a seed = size of the covd sample nuclei
n_neighbors = int(sys.argv[2]) # the number of nuclei in each descriptor
dirpath = sys.argv[3] # the full path to the sample directory with feature data

sample = dirpath.split('/')
tissue = sample[-2]
samplename = sample[-1].split('.')[0]
outdir = '/home/garner1/Work/pipelines/nucleAI/data/covds/'+tissue+'/'+samplename

try:
    os.stat(outdir)
except:
    os.makedirs(outdir,exist_ok=True)    

print('Loading the data')
df = pd.DataFrame()
fovs = glob.glob(dirpath+'/*.pkl')
print('There are '+str(len(fovs))+' FOVs')
for fov in fovs: # for each fov
    data = pd.read_pickle(fov) # do not consider intensities
    df = df.append(data, ignore_index = True)

numb_nuclei = df.shape[0] 
print(str(numb_nuclei)+' nuclei')
if numb_nuclei > 100:
    df = df[df['area']>10].reset_index(drop=True) # filter out small nuclei
    df = df[df['perimeter']>0].reset_index(drop=True) # make sure perimeter is always positive
    df['area'] = df['area'].astype(float) # convert to float this field
    df['circularity'] = 4.0*np.pi*df['area'] / (df['perimeter']*df['perimeter']) # add circularity
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
    s1, s2 = data[indices[fdf.index[0],:],:].shape
    tensor = np.empty((size,s1,s2))
    for node in range(size):
        tensor[node,:,:] = data[indices[node,:],:]

    print('Generating the descriptor')
    num_cores = multiprocessing.cpu_count() # numb of cores

    node_vecWith_vecWo = Parallel(n_jobs=num_cores)(
            delayed(covd_parallel_with_intensity)(node,tensor) for node in tqdm(range(size))
        )

    # Add the descriptor feature to fdf
    fdf["descriptor_woI"] = ""; fdf["descriptor_woI"].astype(object)
    fdf["descriptor_withI"] = ""; fdf["descriptor_withI"].astype(object)
    for item in node_vecWith_vecWo:
        descriptor_with = item[1]; descriptor_wo = item[2]
        node = fdf.index[item[0]]
        fdf.at[node,'descriptor_withI'] = pd.Series(descriptor_with).values
        fdf.at[node,'descriptor_woI'] = pd.Series(descriptor_wo).values

    # Store file
    filename = outdir+'/nuclei'+str(numb_nuclei)+'.numbCovd'+str(size)+'.freq'+str(frequency)+'.covdNN'+str(n_neighbors)+'.pkl'
    fdf.to_pickle(filename)




