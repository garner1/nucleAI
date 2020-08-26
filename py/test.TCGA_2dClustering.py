#!/usr/bin/env python
# coding: utf-8

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
matplotlib.use('Agg') # to not display figure while using ssh 
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
from sklearn.decomposition import PCA
from scipy.sparse import find
from numpy.linalg import norm
import timeit
import multiprocessing
from joblib import Parallel, delayed
from datetime import datetime
from tqdm import tqdm

from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.graph_objs import *
import plotly.express as px
import plotly

import warnings
warnings.filterwarnings('ignore')

from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors

# In[33]:

samples = glob.glob('/media/garner1/hdd2/TCGA_polygons/*/*/*.freq10.covdNN50.features.pkl')

# Load the covd-barycenters for all samples
outfile = 'covd_barycenters.npy'
barycenters = np.load(outfile)

cancer_type = []
sample_id = []
for sample in samples:
    cancer_type.append( sample.split('/')[5] )
    sample_id.append( os.path.basename(sample).split('.')[0] )

print(len(cancer_type),set(cancer_type))

# UMAP representations and clustering

reducer = umap.UMAP(n_components=2)
embedding = reducer.fit_transform(barycenters)

x_umap = embedding[:,0]
y_umap = embedding[:,1]

df = pd.DataFrame(dict(x_umap=x_umap, y_umap=y_umap, label=cancer_type, sample=sample_id))

from sklearn.cluster import KMeans
mat_umap = df[['x_umap','y_umap']].values
kmeans = KMeans(n_clusters=3, random_state=42).fit(mat_umap)
df['UMAP_cluster_ID'] = kmeans.labels_

# PCA representations and clustering

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(barycenters)

x_pca = principalComponents[:,0]
y_pca = principalComponents[:,1]
df['x_pca'] = x_pca
df['y_pca'] = y_pca

mat_pca = df[['x_pca','y_pca']].values
kmeans = KMeans(n_clusters=2, random_state=42).fit(mat_pca)
df['PCA_cluster_ID'] = kmeans.labels_

df.to_pickle("./df_clusters.pkl")
print('Done!')
