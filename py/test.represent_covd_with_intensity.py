#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
get_ipython().run_line_magic('matplotlib', 'inline')

import sys
sys.path.insert(0, '../py')
from graviti import *

import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix
import scipy.ndimage as ndi

from skimage.draw import polygon
from skimage import io
from skimage.measure import label, regionprops
import skimage.io
import skimage.measure
import skimage.color

import glob
import pickle
import pandas as pd
import os
import timeit
import random

import pyvips
import timeit
import multiprocessing
from joblib import Parallel, delayed
from datetime import datetime
from tqdm import tqdm

import umap
import seaborn as sns; sns.set()

patches = glob.glob('/home/garner1/pipelines/nucleAI/data/TCGA-A2-A0CK-01Z-00-DX1/*.pkl')

reducer = umap.UMAP(n_components=2,min_dist=0,n_neighbors=10)
intensity_features = []
for patch in patches[:]:
    #print(patch)
    infile = open(patch,'rb')
    lista = pickle.load(infile)
    intensity_features.extend(lista)
    infile.close()

list_of_features = random.sample(intensity_features,100000)
pos = np.array([f[0] for f in list_of_features if f is not None])  
data_cov = np.array([np.real(sp.linalg.logm(np.cov(f[1],rowvar=False))).flatten() for f in list_of_features if f is not None])  
data_corrcoef = np.array([np.real(sp.linalg.logm(np.corrcoef(f[1],rowvar=False))).flatten() for f in list_of_features if f is not None])  

embedding = reducer.fit_transform(data_cov)
x = embedding[:,0]
y = embedding[:,1]
df_plot = pd.DataFrame(dict(x=x, y=y))
fig, ax = plt.subplots(figsize=(10,10))
ax = sns.scatterplot(x="x", y="y", data=df_plot)
plt.savefig('cov.png')

embedding = reducer.fit_transform(data_corrcoef)
x = embedding[:,0]
y = embedding[:,1]
df_plot = pd.DataFrame(dict(x=x, y=y))
fig, ax = plt.subplots(figsize=(10,10))
ax = sns.scatterplot(x="x", y="y", data=df_plot)
plt.savefig('corrcoef.png')


