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
from skimage.measure import label, regionprops
import math

import timeit
import multiprocessing
from joblib import Parallel, delayed
from datetime import datetime
from tqdm import tqdm


# In[ ]:


features = ['centroid_x','centroid_y','area','eccentricity','orientation','perimeter','solidity']
num_cores = multiprocessing.cpu_count() # numb of cores

sample = sys.argv[1] #~/Work/dataset/tcga_polygons/LUAD/*.gz/*.gz
ID = os.path.basename(sample).split(sep='.')[0] #get sample ID
print(ID)
dirname = os.path.dirname(sample) #get the sample directory
print(dirname)

# get_ipython().system('tar -xf $sample')
# get_ipython().system("mv './luad_polygon/' $dirname") # this part might create problems when run in parallel because dirs are not correctly assigned

patchlist = glob.glob(dirname+'/*_polygon/*.svs/*.csv') #get the list of patches    
print('There are '+str(len(patchlist))+' patches')

# print('Showing the patches as png files...')
# Parallel(n_jobs=num_cores)(delayed(show_patches_parallel)(filename) for filename in tqdm(patchlist) if ~pd.read_csv(filename).empty)
        
print('Calculating the morphometry...')
Parallel(n_jobs=num_cores)(
    delayed(measure_patch_of_polygons)(filename,features) for filename in tqdm(patchlist) if ~pd.read_csv(filename).empty
)
print('Done!')
