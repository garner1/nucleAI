#!/usr/bin/env python
# coding: utf-8

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

samples = glob.glob('/media/garner1/hdd2/TCGA_polygons/*/*/*.freq10.covdNN50.features.pkl')
num_cores = multiprocessing.cpu_count() # numb of cores

# The barycenters array contain the list of covd-barycenters, one per sample
barycenter_list = Parallel(n_jobs=num_cores)(
    delayed(load_barycenters)(sample) for sample in tqdm(samples) # load_barycenters evaluate the barycenter of the sample
    )

barycenters = np.zeros((len(samples),pd.read_pickle(samples[0])['descriptor'].iloc[0].shape[0]))
row = 0
for b in barycenter_list:
    barycenters[row,:] = b
    row += 1
barycenters = barycenters[~np.all(barycenters == 0, axis=1)]

outfile = 'covd_barycenters'
np.save(outfile,barycenters)
