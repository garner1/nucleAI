#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


########################
def scattered2d_tcga(df,filename):
    fig = px.scatter(df,
                     x="x", y="y",
                     color="label",
                     hover_name='sample',
                     color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_traces(marker=dict(size=5,opacity=1.0))
    fig.update_layout(template='simple_white')
    fig.update_layout(legend= {'itemsizing': 'constant'})
    fig.write_html(filename+'.tcga.html', auto_open=False)
    return

def scattered3d_tcga(df,filename):
    fig = px.scatter_3d(df,
                        x="x", y="y", z="z",
                        color="label",
                        hover_name='sample',
                        color_discrete_sequence=px.colors.qualitative.Plotly
    )
    fig.update_traces(marker=dict(size=3,opacity=1.0))
    fig.update_layout(template='simple_white')
    fig.update_layout(legend= {'itemsizing': 'constant'})
    fig.write_html(filename+'.tcga.html', auto_open=False)
    return

def load_barycenters(sample):
    df = pd.read_pickle(sample)
    barycenter = df[df['covd']==1]['descriptor'].mean()
    return barycenter

########################


# In[33]:

samples = glob.glob('/media/garner1/hdd2/TCGA_polygons/*/*/*.freq10.*.pkl')

num_cores = multiprocessing.cpu_count() # numb of cores
barycenter_list = Parallel(n_jobs=num_cores)(
    delayed(load_barycenters)(sample) for sample in tqdm(samples)
    )

barycenters = np.zeros((len(samples),pd.read_pickle(samples[0])['descriptor'].iloc[0].shape[0]))
row = 0
for b in barycenter_list:
    barycenters[row,:] = b
    row += 1
barycenters = barycenters[~np.all(barycenters == 0, axis=1)]

cancer_type = []
sample_id = []
for sample in samples:
    cancer_type.append( sample.split('/')[5] )
    sample_id.append( os.path.basename(sample).split('.')[0] )

print(len(cancer_type),set(cancer_type))

# In[42]:

reducer = umap.UMAP(n_components=3)
embedding = reducer.fit_transform(barycenters)

# Generate Data
x = embedding[:,0]
y = embedding[:,1]
z = embedding[:,2]

df = pd.DataFrame(dict(x=x, y=y, z=z, label=cancer_type, sample=sample_id))

filename = 'umap.s'+str(df.shape[0])
scattered3d_tcga(df,filename)

# groups = df.groupby('label')

# # Plot
# fig, ax = plt.subplots(figsize=(10,10))
# ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
# for name, group in groups:
#     ax.plot(group.x, group.y, marker='o', linestyle='', ms=6, label=name, alpha=0.5)
# ax.legend()
# plt.title('UMAP projection of the TCGA dataset', fontsize=12)
# plt.savefig('tcga.umap.s'+str(df.shape[0])+'.png')


# # In[37]:

pca = PCA(n_components=3)
principalComponents = pca.fit_transform(barycenters)

x = principalComponents[:,0]
y = principalComponents[:,1]
z = principalComponents[:,2]

df = pd.DataFrame(dict(x=x, y=y, z=z, label=cancer_type, sample=sample_id))
filename = 'pca.s'+str(df.shape[0])
scattered3d_tcga(df,filename)

# groups = df.groupby('label')

# # Plot
# fig, ax = plt.subplots(figsize=(10, 10))
# ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
# for name, group in groups:
#     ax.plot(group.x, group.y, marker='o', linestyle='', ms=6, label=name, alpha=0.5)
# ax.legend()
# plt.title('PCA projection of the TCGA dataset', fontsize=12)
# plt.savefig('tcga.pca.s'+str(df.shape[0])+'.png')




