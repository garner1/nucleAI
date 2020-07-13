#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys  
sys.path.insert(0, '../py')
from graviti import *

import json
import numpy as np
from skimage.draw import polygon
from skimage import io
from matplotlib import pyplot as plt
import glob
import pandas as pd
import os
from scipy.sparse import coo_matrix
from skimage.measure import label, regionprops#, regionprops_table
import math

import timeit
import multiprocessing
from joblib import Parallel, delayed
from datetime import datetime
from tqdm import tqdm


# In[ ]:


datadir = '/home/garner1/Work/dataset/tcga_polygons/LUAD'
samples = glob.glob(datadir+'/*.gz')
print('There are '+str(len(samples))+' samples')


# In[ ]:


features = ['centroid_x','centroid_y','area','eccentricity','orientation','perimeter','solidity']
num_cores = multiprocessing.cpu_count() # numb of cores

#for sample in glob.glob(datadir+'/*.gz/*.gz'): # for each sample compressed file
sample = sys.argv[1]
ID = os.path.basename(sample).split(sep='.')[0] #get sample ID
print(ID)
get_ipython().system('tar -xf $sample')
dirname = os.path.dirname(sample) #get the sample directory
print(dirname)
if True:#not os.path.exists(os.path.join(dirname,'luad_polygon')): # if the polygon folder does not exist proceed
    get_ipython().system("mv './luad_polygon/' $dirname")
    patchlist = glob.glob(dirname+'/*_polygon/*.svs/*.csv')[:] #get the list of patches    

    # Show the patches as png files
    #Parallel(n_jobs=num_cores)(delayed(show_patches_parallel)(filename) for filename in tqdm(patchlist) if ~pd.read_csv(filename).empty)
        
    # Calculate the morphometrics of each patch as a pkl file
    Parallel(n_jobs=num_cores)(
            delayed(measure_patch_of_polygons)(filename,features) for filename in tqdm(patchlist[:]) if ~pd.read_csv(filename).empty
        )

