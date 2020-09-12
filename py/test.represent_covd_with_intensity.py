#!/usr/bin/env python
# coding: utf-8

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg') # to not display figure while using ssh 
import matplotlib.patches as mpatches

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

import timeit
import multiprocessing
from joblib import Parallel, delayed
from datetime import datetime
from tqdm import tqdm

import umap
import seaborn as sns; sns.set()

from sklearn.cluster import KMeans

patches = glob.glob('../data/TCGA-4H-AAAK-01Z-00-DX1/*.pkl')

reducer = umap.UMAP(n_components=2,min_dist=0,n_neighbors=10)
features = []
for patch in patches[:]:
    features.append(pd.read_pickle(patch))
df_data = pd.concat(features)
print(df_data.shape)
print(df_data.head())
# list_of_features = random.sample(intensity_features,10000)
# pos = np.array([f[0] for f in intensity_features if f is not None])  
# print(np.max(pos,axis=0))
# data_cov = np.array([np.real(sp.linalg.logm(np.cov(f[1],rowvar=False))).flatten() for f in list_of_features if f is not None])  
# data_corrcoef = np.array([np.real(sp.linalg.logm(np.corrcoef(f[1],rowvar=False))).flatten() for f in list_of_features if f is not None])  

# # UMAP representation
# embedding = reducer.fit_transform(data_cov)
# x = embedding[:,0]
# y = embedding[:,1]
# df_plot = pd.DataFrame(dict(x_umap=x, y_umap=y))
# df_plot['x'] = pos[:,0]; df_plot['y'] = pos[:,1]

# # K-means classification
# kmeans = KMeans(n_clusters=2, random_state=0).fit(embedding)
# df_plot['cluster'] = kmeans.labels_

# # Show the cluster to study
# fig, ax = plt.subplots(figsize=(10,10))
# ax = sns.scatterplot(x="x", y="y",hue='cluster',s=10, data=df_plot)
# plt.savefig('cov.png')
# # ax = seaborn.FacetGrid(data=df_plot[['x_umap','y_umap','cluster']],hue='cluster')
# # ax.map(plt.scatter, 'x_umap', 'y_umap',s=10).add_legend()



