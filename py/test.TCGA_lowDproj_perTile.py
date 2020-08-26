#!/usr/bin/env python
# coding: utf-8

# Samples random tiles from each WSI, for a given tissue, and project covd to lowD with umap

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

from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors

cancer_ID = sys.argv[1]
sample_size = 100 # the number of tiles to sample from each WSI

samples = glob.glob('/media/garner1/hdd2/TCGA_polygons/'+cancer_ID+'/*/*.freq10.covdNN50.features.pkl')
df_tissue = pd.read_pickle(samples[0]).sample(sample_size) # load the first instance of a WSI
for sample in samples[1:100]:
    df = pd.read_pickle(sample)
    df_tissue = df_tissue.append(df.sample(sample_size))

print('Done with loading '+cancer_ID+', now projecting')

import umap
reducer = umap.UMAP(n_components=2)

data = np.array(df_tissue[df_tissue['covd']==1].descriptor.tolist()) # generate the global array of tiles

filename = 'df.'+cancer_ID+'.s'+str(df.shape[0])
df_tissue.to_pickle(filename)
del(df_tissue) # to make space for parallelization over tissue types

embedding = reducer.fit_transform(data) # reduce to lowD with umap

x = embedding[:,0]
y = embedding[:,1]

df = pd.DataFrame(dict(x=x, y=y))
# Plot
fig, ax = plt.subplots(figsize=(10,10))
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
ax.plot(x, y, marker='o', linestyle='', ms=3, alpha=0.75)
plt.title('UMAP projection of '+cancer_ID+' tiles', fontsize=12)
filename = 'umap.'+cancer_ID+'.s'+str(df.shape[0])+'.pdf'
plt.savefig(filename)
