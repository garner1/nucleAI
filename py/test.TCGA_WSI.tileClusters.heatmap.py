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

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # to not display figure while using ssh 

import warnings
warnings.filterwarnings('ignore')

sample = sys.argv[1] # the sample's data file

sample_size = 10000
df = pd.read_pickle(sample)#.head(sample_size) # load the first instance of a WSI

import umap
reducer = umap.UMAP(n_components=2)

data = np.array(df[df['covd']==1].descriptor.tolist()) # generate the global array of tiles

embedding = reducer.fit_transform(data) # reduce to lowD with umap

x_umap = embedding[:,0]
y_umap = embedding[:,1]

df['x_umap'] = x_umap
df['y_umap'] = y_umap

# Clustering
from sklearn.cluster import KMeans
mat_umap = df[['x_umap','y_umap']].values
kmeans = KMeans(n_clusters=2, random_state=42).fit(mat_umap)
df['UMAP_cluster_ID'] = kmeans.labels_

ax = sns.scatterplot(x="centroid_x",
                     y="centroid_y",
                     hue="UMAP_cluster_ID",
                     s=5,alpha=1.0,marker='.',
                     data=df)
plt.title('Heatmap of tile clusters ID', fontsize=12)
ax.invert_yaxis()
filename = 'heatmap.s'+str(df.shape[0])+'.pdf'
plt.savefig(filename)
