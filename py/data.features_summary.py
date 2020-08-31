import sys  
sys.path.insert(0, '../py')
from graviti import *

import json
import numpy as np
from matplotlib import pyplot as plt
import glob
import pandas as pd
import os

import timeit
import multiprocessing
from joblib import Parallel, delayed
from datetime import datetime
from tqdm import tqdm

count = 0
for tissue in glob.glob('/media/garner1/hdd2/TCGA_polygons/*')[:]:
    for sample in glob.glob(tissue+'/TCGA-*')[:]:
        for data in glob.glob(sample+'/*.freq10.covdNN50.features.pkl'):
            #summary = pd.read_pickle(data).describe()
            #print(summary)
            path_components = data.split('/')
            tissue_ID = path_components[5]
            sample_ID = path_components[6].split('.')[0]
            filename = './summary_data/'+tissue_ID+'_'+sample_ID+'.csv.gz'
            print(count,tissue_ID,sample_ID)
            pd.read_pickle(data).describe().to_csv(filename)
            count += 1
