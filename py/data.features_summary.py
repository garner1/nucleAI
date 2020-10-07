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
sample = sys.argv[1]

pklfiles =  glob.glob(sample+'/*features.csv.pkl')
for data in pklfiles:
    if count == 0:
        df = pd.read_pickle(data)
        df = df.astype(np.float64)
    else:
        df = df.append(pd.read_pickle(data), ignore_index=True)
    count += 1
summary = df.describe(include='all')
  
filename = sample+'/features_summary.csv'
summary.to_csv(filename)

