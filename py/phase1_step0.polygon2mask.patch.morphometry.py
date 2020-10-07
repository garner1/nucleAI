#!/usr/bin/env python
# coding: utf-8

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
from datetime import datetime

features = ['cx','cy','area','eccentricity','orientation','perimeter','solidity']

patch = sys.argv[1] #~/Work/dataset/tcga_polygons/LUAD/*.gz/*.gz

#print('Calculating the morphometry...')
sample = os.path.dirname(patch).split('/')
tissue = sample[5]
samplename = sample[6].split('.')[0]
outdir = '/home/garner1/Work/pipelines/nucleAI/data/features_wo_intensity/'+tissue+'/'+samplename
try:
    os.stat(outdir)
except:
    os.makedirs(outdir,exist_ok=True)    

measure_patch_of_polygons(patch,features,outdir)
