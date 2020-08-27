#!/usr/bin/env python
# coding: utf-8

import sys  
sys.path.insert(0, '../py')
from graviti import *

from numpy.linalg import norm
import numpy as np
import os
import os.path
from os import path
import sys
import glob
import h5py
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # to not display figure while using ssh 
import plotly.graph_objects as go
from plotly.graph_objs import *
import plotly.express as px
import hdbscan
import pandas as pd
import umap
import networkx as nx
from scipy import sparse, linalg
import pickle
from sklearn.preprocessing import normalize, scale
from sklearn.decomposition import PCA
from scipy.sparse import find
from numpy.linalg import norm
import timeit
import multiprocessing
from joblib import Parallel, delayed
from datetime import datetime
from tqdm import tqdm

from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.graph_objs import *
import plotly.express as px
import plotly

import warnings
warnings.filterwarnings('ignore')

samples = glob.glob('/media/garner1/hdd2/TCGA_polygons/*/*/*.freq10.covdNN50.features.pkl')

cancer_type = []
sample_id = []
for sample in samples:
    cancer_type.append( sample.split('/')[5] )
    sample_id.append( os.path.basename(sample).split('.')[0] )

with open('list_of_cancerID.pkl', 'wb') as f:
    pickle.dump(cancer_type,f)





