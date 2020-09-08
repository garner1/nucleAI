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
        for data in glob.glob(sample+'/*.freq10.covdNN50.features.pkl')[:]:
            df = pd.read_pickle(data)
            df = df.drop(columns = ['covd', 'descriptor'])
            df = df.astype(np.float64)
            summary = df.describe(include='all')
            # print(summary.shape)

            path_components = data.split('/')
            tissue_ID = path_components[5]
            sample_ID = path_components[6].split('.')[0]
            filename = './summary_data/'+tissue_ID+'_'+sample_ID+'.csv.gz'
            print(count,tissue_ID,sample_ID,summary.shape)
            summary.to_csv(filename)
            count += 1

# datas = glob.glob('/media/garner1/hdd2/TCGA_polygons/BLCA/TCGA-2F-A9KW-01Z-00-DX2.*.svs.tar.gz/*.freq10.covdNN50.features.pkl')
# #datas = glob.glob('/media/garner1/hdd2/TCGA_polygons/BLCA/TCGA-2F-A9KW-01Z-00-DX2.*.svs.tar.gz/blca_polygon/*/*.pkl')
# #datas = glob.glob('/media/garner1/hdd2/TCGA_polygons/BLCA/TCGA-2F-A9KO-01Z-00-DX1.*.svs.tar.gz/blca_polygon/*/*.pkl')
# for data in datas[:1]:
#     df = pd.read_pickle(data)
#     df = df.drop(columns = ['covd', 'descriptor'])
#     df = df.astype(np.float64)
#     summary = df.describe(include='all')
#     print(summary.shape)

# # data = glob.glob('/media/garner1/hdd2/TCGA_polygons/BLCA/TCGA-2F-A9KO-01Z-00-DX1.195576CF-B739-4BD9-B15B-4A70AE287D3E.svs.tar.gz/TCGA-2F-A9KO-01Z-00-DX1.nuclei1541213.numbCovd154121.freq10.covdNN50.features.pkl')[0]
# # summary = pd.read_pickle(data).describe()
# # print(summary)
# # print(summary.shape)
