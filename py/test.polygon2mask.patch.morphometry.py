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

features = ['centroid_x','centroid_y','area','eccentricity','orientation','perimeter','solidity']

patch = sys.argv[1] #~/Work/dataset/tcga_polygons/LUAD/*.gz/*.gz

#print('Calculating the morphometry...')
measure_patch_of_polygons(patch,features)
