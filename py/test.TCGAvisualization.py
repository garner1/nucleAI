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


# In[33]:


samples = glob.glob('/media/garner1/hdd2/TCGA_polygons/*/*/*.freq10.*.pkl')

barycenters = np.zeros((len(samples),pd.read_pickle(samples[0])['descriptor'].iloc[0].shape[0]))
row = 0
ctype = []
for sample in samples:
    df = pd.read_pickle(sample)
    #if df.shape[0] > 1000:
    ct = sample.split('/')[5]
    ctype.append(ct)
    barycenter = df[df['covd']==1]['descriptor'].mean()
    barycenters[row,:] = barycenter
    row += 1
barycenters = barycenters[~np.all(barycenters == 0, axis=1)]
print(len(ctype),set(ctype))


# In[42]:


import umap
reducer = umap.UMAP()
embedding = reducer.fit_transform(barycenters)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Generate Data
x = embedding[:,0]
y = embedding[:,1]
labels = ctype

df = pd.DataFrame(dict(x=x, y=y, label=labels))

groups = df.groupby('label')

# Plot
fig, ax = plt.subplots(figsize=(10,10))
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=10, label=name)
ax.legend()
plt.title('UMAP projection of the TCGA dataset', fontsize=12)
plt.savefig('tcga.umap.s'+str(df.shape[0])+'.png')


# In[37]:


from sklearn.decomposition import PCA
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(barycenters)

x = principalComponents[:,0]
y = principalComponents[:,1]
z = principalComponents[:,2]
labels = ctype

df = pd.DataFrame(dict(x=y, y=z, label=labels))

groups = df.groupby('label')

# Plot
fig, ax = plt.subplots(figsize=(10, 10))
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=10, label=name)
ax.legend()
plt.title('PCA projection of the TCGA dataset', fontsize=12)
plt.savefig('tcga.pca.s'+str(df.shape[0])+'.png')




