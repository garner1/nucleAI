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

data_dir = sys.argv[1]
samples = glob.glob(data_dir+'/*/*.freq10.covdNN50.pkl')

#####################################################################################
# The barycenters array contain the list of covd-barycenters, one per sample
num_cores = multiprocessing.cpu_count() # numb of cores

# Here is where you specify the kind of descriptor 
descriptor = 'descriptor_woI' # "descriptor_woI" or "descriptor_withI"
# descriptor = 'descriptor_withI' # "descriptor_woI" or "descriptor_withI"

barycenter_list = Parallel(n_jobs=num_cores)(
    delayed(load_barycenters)(sample,descriptor) for sample in tqdm(samples) # load_barycenters evaluate the barycenter of the sample
    )
barycenters = np.zeros((len(samples),pd.read_pickle(samples[0])[descriptor].iloc[0].shape[0]))
row = 0
for b in barycenter_list:
    barycenters[row,:] = b
    row += 1
barycenters = barycenters[~np.all(barycenters == 0, axis=1)]

sample_id = []
for sample in samples:
    sample_id.append( os.path.dirname(sample).split('/')[-1] )


# UMAP representations

reducer = umap.UMAP(n_components=2,min_dist=0,random_state=42) # set random state for reproducibility
embedding = reducer.fit_transform(barycenters)

x = embedding[:,0]
y = embedding[:,1]

df = pd.DataFrame(dict(x=x, y=y, sample=sample_id))
# Plot
fig, ax = plt.subplots(figsize=(10,10))
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
ax.plot(x, y, marker='o', linestyle='', ms=3, alpha=0.5)
plt.title('UMAP projection of the BRCA dataset '+descriptor, fontsize=12)
filename = data_dir+'/umap.'+descriptor+'.pdf'
plt.savefig(filename)
df.to_csv(data_dir+'/'+descriptor+'.umap.csv')




