from matplotlib import pyplot as plt
import matplotlib.patches as mpatches

import sys
sys.path.insert(0, '../py')
from graviti import *

import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix
import scipy.ndimage as ndi

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

def get_LogCovMat_parallel(featureMat):
    out = np.real(sp.linalg.logm(np.cov(featureMat,rowvar=False))).flatten()
    return out
def get_LogCorrMat_parallel(featureMat):
    return np.real(sp.linalg.logm(np.corrcoef(featureMat,rowvar=False))).flatten()
def get_position_parallel(featureMat):
    pos = featureMat[0]
    return pos

path_to_data = sys.argv[1]
patches = glob.glob(path_to_data+'/*.pkl')

reducer = umap.UMAP(n_components=2,min_dist=0,n_neighbors=10)
intensity_features = []
for patch in patches[:2]:
    #print(patch)
    infile = open(patch,'rb')
    lista = pickle.load(infile)
    intensity_features.extend(lista)
    infile.close()

list_of_features = random.sample(intensity_features,100)
num_cores = multiprocessing.cpu_count() # numb of cores

pos = Parallel(n_jobs=num_cores)( delayed(get_position_parallel)(f) for f in tqdm(intensity_features) if f is not None)
data_cov = Parallel(n_jobs=num_cores)( delayed(get_LogCovMat_parallel)(f[1]) for f in tqdm(intensity_features) if f is not None)
data_corrcoef = Parallel(n_jobs=num_cores)( delayed(get_LogCorrMat_parallel)(f[1]) for f in tqdm(intensity_features) if f is not None)

barycenter_cov = np.mean(data_cov,axis=0)
dist = sp.linalg.norm(data_cov-barycenter_cov,axis=1)
print(np.mean(dist),np.std(dist),np.cov(dist))

