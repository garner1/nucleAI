from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('Agg') # to not display figure while using ssh 

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

def get_LogCovMat_parallel(featureMat): # evaluate covariance matrix
    cov = np.cov(featureMat,rowvar=False) 
    # add small diag perturbation to address singularity
    m = 10^-6
    cov = cov+np.eye(cov.shape[1])*m

    out = np.real(sp.linalg.logm(cov)).flatten()
    return out

def get_LogCorrMat_parallel(featureMat): # evaluate correlation matrix
    corr = np.corrcoef(featureMat,rowvar=False) 
    # add small diag perturbation to address singularity
    m = 10^-6
    corr = corr+np.eye(corr.shape[1])*m

    out = np.real(sp.linalg.logm(corr)).flatten()
    return out

def get_position_parallel(featureMat): # extract positions
    pos = featureMat[0]
    return pos

path_to_data = sys.argv[1] # directory path to intensity features files
patches = glob.glob(path_to_data+'/*.intensity_features.pkl')

# load feature data
intensity_features = []
for patch in patches:
    infile = open(patch,'rb')
    lista = pickle.load(infile)
    intensity_features.extend(lista)
    infile.close()
    
num_cores = multiprocessing.cpu_count() # numb of cores

# evaluate positions and covd
#pos = Parallel(n_jobs=num_cores)( delayed(get_position_parallel)(f) for f in tqdm(intensity_features) if f is not None)
data_cov = Parallel(n_jobs=num_cores)( delayed(get_LogCovMat_parallel)(f[1]) for f in tqdm(intensity_features) if f is not None)
data_corrcoef = Parallel(n_jobs=num_cores)( delayed(get_LogCorrMat_parallel)(f[1]) for f in tqdm(intensity_features) if f is not None)

#get the barycenter of the entire sample
barycenter_cov_all = np.mean(data_cov,axis=0)
barycenter_corr_all = np.mean(data_corrcoef,axis=0)
sample_dist = []
for sample_size in [int(round(np.percentile(range(len(intensity_features)),p))) for p in range(10,100,10)]:
    list_of_features = random.sample(intensity_features,sample_size)

    data_cov = Parallel(n_jobs=num_cores)( delayed(get_LogCovMat_parallel)(f[1]) for f in list_of_features if f is not None)
    data_corrcoef = Parallel(n_jobs=num_cores)( delayed(get_LogCorrMat_parallel)(f[1]) for f in list_of_features if f is not None)

    barycenter_cov_subsample = np.mean(data_cov,axis=0)
    barycenter_corr_subsample = np.mean(data_corrcoef,axis=0)

    dist_cov = sp.linalg.norm(barycenter_cov_all - barycenter_cov_subsample)
    dist_corr = sp.linalg.norm(barycenter_corr_all - barycenter_corr_subsample)
    sample_dist.append( tuple((sample_size,dist_cov,dist_corr)) )

for item in sample_dist:
    print(item[0],item[1],item[2])
    
#dist = sp.linalg.norm(data_cov-barycenter_cov,axis=1)
#print(np.mean(dist),np.std(dist),np.cov(dist))

