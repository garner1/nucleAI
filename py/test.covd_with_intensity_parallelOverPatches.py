#!/usr/bin/env python
# coding: utf-8

# In[12]:


import sys
sys.path.insert(0, '../py')
from graviti import *


# In[13]:


import numpy as np
import scipy as sp
from scipy.sparse import coo_matrix
import scipy.ndimage as ndi

from skimage.draw import polygon
from skimage import io
from skimage.measure import label, regionprops
import skimage.io
import skimage.measure
import skimage.color

import glob
import pickle
import pandas as pd
import os
import timeit
import random

import pyvips


# In[14]:


from __future__ import print_function

import histomicstk as htk

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:


import timeit
import multiprocessing
from joblib import Parallel, delayed
from datetime import datetime
from tqdm import tqdm


# In[16]:


def make_mask_from_polygons(filename,x_list,y_list):
    if not (x_list and y_list):
        pass
    else:
        xx = np.array(x_list).reshape((len(x_list),1))
        yy = np.array(y_list).reshape((len(y_list),1))

        arr = np.hstack((xx,yy))

        # subtract the min to translate the mask                                                                                                                                                                   
        mini = np.min(arr,axis=0); arr -= mini

        rr = np.rint(arr[:,1]).astype(int) # xs are cols                                                                                                                                                           
        cc = np.rint(arr[:,0]).astype(int) # ys are rows                                                                                                                                                           
        mtx = coo_matrix((np.ones(rr.shape), (rr, cc)), dtype=bool)
        
        #plt.figure(figsize=(40,40))
        #io.imshow(mtx.todense(),cmap='gray')
        #plt.savefig(filename+'.png')
    return mtx.todense()


# In[17]:


def get_nuclear_view(imInput,maska,z):
    masked = np.multiply(imInput[:,:,z],maska)
    norows = masked[~np.all(masked == 0, axis=1)] #remove 0 rows
    arr = norows[:,~(norows == 0).all(0)] # remove 0 cols
    return arr 


# In[18]:


# get the features 
# consider as features x,y,r,g,b,delta_x([r,g,b])+delta_y([r,g,b]),delta_xx([r,g,b])+delta_yy([r,g,b])
def features_from_3d(arr_3d,color_dim): # color_dim is 0,1,2 for R,G,B
    dx = np.array([[0.0,0,0.0],[-1.0,0,1.0],[0.0,0,0.0],])
    dy = np.transpose(dx)
    dxx = np.array([[0.0,0,0.0],[-1.0,2.0,-1.0],[0.0,0,0.0],])
    dyy = np.transpose(dxx)

    arr_2d = arr_3d[:,:,color_dim]
    coo = coo_matrix(arr_2d)
    
    row = coo.row
    col = coo.col
        
    delta_x = ndi.convolve(arr_2d,dx, output=np.float64, mode='nearest')
    delta_y = ndi.convolve(arr_2d,dy, output=np.float64, mode='nearest')
        
    delta_xx = ndi.convolve(arr_2d,dxx, output=np.float64, mode='nearest')
    delta_yy = ndi.convolve(arr_2d,dyy, output=np.float64, mode='nearest')
    
    return delta_x, delta_y, delta_xx, delta_yy


# In[19]:


def parse_polygons_in_patch(filename):
    x_list = []
    y_list = []
    df = pd.read_csv(filename)
    if ~df.empty:
        cell_list = df['Polygon'].tolist()
        for cell in cell_list: # loop over cells in patch                                                                                                                                                          
            lista = list(np.fromstring(cell[1:-1], dtype=float, sep=':')) #list of vertices in polygon                                                                                                             
            cc = lista[0::2] # list of x coord of each polygon vertex                                                                                                                                              
            rr = lista[1::2] # list of y coord of each polygon verted                                                                                                                                              
            poly = np.asarray(list(zip(rr,cc)))
            mini = np.min(poly,axis=0)
            poly -= mini # subtract the min to translate the mask                                                                                                                                                  

            # create the nuclear mask                                                                                                                                                                              
            mask = np.zeros(tuple(np.ceil(np.max(poly,axis=0) - np.min(poly,axis=0)).astype(int)))
            rr, cc = polygon(poly[:, 0], poly[:, 1], mask.shape) # get the nonzero mask locations                                                                                                                  
            mask[rr, cc] = 1 # nonzero pixel entries                                                                                                                                                               
            # rescale back to original coordinates                                                                                                                                                                 
            rr = rr.astype(float);cc = cc.astype(float)
            rr += mini[0]; cc += mini[1]

            # update the list of nonzero pixel entries                                                                                                                                                             
            x_list.extend( [int(n) for n in list(cc)] )
            y_list.extend( [int(n) for n in list(rr)] )
        mask = make_mask_from_polygons(filename,x_list,y_list)
    return mask


# In[20]:


def tile_from_svs(svs_filename,mask,x,y):
    
    format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
    }
    
    image = pyvips.Image.new_from_file(svs_filename)[0:3] # drop alpha channel
    tile = image.crop(x,y,mask.shape[1],mask.shape[0])
    np_3d = np.ndarray(buffer=tile.write_to_memory(),
                       dtype=format_to_dtype[tile.format],
                       shape=[tile.height, tile.width, tile.bands])
    
    #print(tile.height, tile.width, tile.bands, tile.format, tile.interpretation)
    #tile.write_to_file(svs_filename+'.'+str(x)+'.'+str(y)+'.jpg[Q=100]') # save as jpg file
    return np_3d


# In[82]:


def covd_rgb(l,labels,imInput,regions,x,y):
    maska = labels == l # get the mask
    if maska.nonzero()[0].shape[0] > 100: # condition on mask size to remove small nuclei
        # Repeat over the third axis of the image
        arr0 = get_nuclear_view(imInput,maska,0)
        arr1 = get_nuclear_view(imInput,maska,1)
        arr2 = get_nuclear_view(imInput,maska,2)

        arr_3d = np.dstack((arr0,arr1,arr2))
                
                #plt.figure()
                #plt.imshow(arr_3d)
                #plt.savefig('./nucleus_'+str(l)+'.png')
                
        # get the features
        delta_x_R, delta_y_R, delta_xx_R, delta_yy_R = features_from_3d(arr_3d,0)
        delta_x_G, delta_y_G, delta_xx_G, delta_yy_G = features_from_3d(arr_3d,1)
        delta_x_B, delta_y_B, delta_xx_B, delta_yy_B = features_from_3d(arr_3d,2)

        delta_x = np.zeros((arr_3d.shape[0],arr_3d.shape[1]))
        delta_xx = np.zeros((arr_3d.shape[0],arr_3d.shape[1]))
        delta_y = np.zeros((arr_3d.shape[0],arr_3d.shape[1]))
        delta_yy = np.zeros((arr_3d.shape[0],arr_3d.shape[1]))
        for r in range(arr_3d.shape[0]):
            for c in range(arr_3d.shape[1]):
                delta_x[r,c] = np.sqrt(delta_x_R[r,c]**2+delta_x_G[r,c]**2+delta_x_B[r,c]**2)
                delta_xx[r,c] = np.sqrt(delta_xx_R[r,c]**2+delta_xx_G[r,c]**2+delta_xx_B[r,c]**2)
                delta_y[r,c] = np.sqrt(delta_y_R[r,c]**2+delta_y_G[r,c]**2+delta_y_B[r,c]**2)
                delta_yy[r,c] = np.sqrt(delta_yy_R[r,c]**2+delta_yy_G[r,c]**2+delta_yy_B[r,c]**2)

        feature_data = np.zeros((arr_3d.shape[0]*arr_3d.shape[1],9))
        idx = 0
        for r in range(arr_3d.shape[0]):
            for c in range(arr_3d.shape[1]):
                    feature_data[idx,:] = np.hstack((r,c,
                                                    arr_3d[r,c,0],arr_3d[r,c,1],arr_3d[r,c,2],
                                                    delta_x[r,c],delta_y[r,c],
                                                    delta_xx[r,c],delta_yy[r,c]))
                    idx += 1
                    
        cx = regions[l-1].centroid[0] + np.float(x) # -1 because the list of regions is 0-based
        cy = regions[l-1].centroid[1] + np.float(y) # -1 because the list of regions is 0-based
        return tuple((cx,cy)),feature_data 
    else:
        return None, None


# In[89]:


def process_patch(patch):
    patch_name = patch.split('/')[9:]
    if not pd.read_csv(patch).empty: 
        print('The patch is not empty',patch_name[0])
        x = patch_name[0].split('_')[0]
        y = patch_name[0].split('_')[1]
        #print(x,y)
        #plt.imshow(imInput)
        mask = parse_polygons_in_patch(patch)
        
        labels, num = label(mask, return_num=True, connectivity=1) # connectivity has to be 1 otherwise different mask are placed together
        regions = regionprops(labels)
        
        imInput = tile_from_svs(svs_filename,mask,x,y)
        
        label_id = [r.label for r in regions if r.label is not None]
        generated_covds = []
        for l in label_id[:]:
            print(l)
            nuc_pos, nuc_featureData = covd_rgb(l,labels,imInput,regions,x,y)
            if nuc_pos is not None:
                generated_covds.append(tuple((nuc_pos,nuc_featureData)))
            
        filename = patch+'.intensity_features.pkl' # name of the intensity features output file
        outfile = open(filename,'wb')
        pickle.dump(generated_covds,outfile)
        outfile.close()
        return


# In[90]:


patches = glob.glob('/home/garner1/pipelines/nucleAI/data/TCGA-05-4244-*.*.svs.tar.gz/luad_polygon/*/*.csv')


# In[ ]:


# Load the mask
svs_filename = "/home/garner1/pipelines/nucleAI/data/TCGA-05-4244-01Z-00-DX1.d4ff32cd-38cf-40ea-8213-45c2b100ac01.svs"

num_cores = multiprocessing.cpu_count() # numb of cores

generated_covds = Parallel(n_jobs=num_cores)( delayed(process_patch)(p) for p in tqdm(patches[:4]) )


# In[58]:


filename = patches[1]+'.intensity_features.pkl'


# In[59]:


infile = open(filename,'rb')
intensity_features = pickle.load(infile)
infile.close()


# In[62]:


[f[0] for f in intensity_features if f is not None]


# In[ ]:


data = np.array([np.real(sp.linalg.logm(np.cov(f[1],rowvar=False))).flatten() for f in intensity_features if f is not None])
print(data.shape)

centroids =  np.array([f[0] for f in intensity_features if f is not None])
print(centroids)


# In[ ]:


import umap
reducer = umap.UMAP(n_components=2,min_dist=0,n_neighbors=10)
embedding = reducer.fit_transform(data)
x = embedding[:,0]
y = embedding[:,1]
df_plot = pd.DataFrame(dict(x=x, y=y))
import seaborn as sns; sns.set()
fig, ax = plt.subplots(figsize=(10,10))
ax = sns.scatterplot(x="x", y="y", data=df_plot)


# In[ ]:





# In[ ]:





# In[ ]:


feature_data = np.zeros((arr_3d.shape[0]*arr_3d.shape[1],9))
idx = 0
for r in range(arr_3d.shape[0]):
    for c in range(arr_3d.shape[1]):
        feature_data[idx,:] = np.hstack((r,c,
                                    arr_3d[r,c,0],arr_3d[r,c,1],arr_3d[r,c,2],
                                    delta_x[r,c],delta_y[r,c],
                                    delta_xx[r,c],delta_yy[r,c]))
        idx += 1

print(np.corrcoef(feature_data,rowvar=False)) # get the normalized covariace matrix
print(np.cov(feature_data,rowvar=False)) # get the covariace matrix


# In[ ]:


# create stain to color map
stain_color_map = htk.preprocessing.color_deconvolution.stain_color_map
print('stain_color_map:', stain_color_map, sep='\n')

# specify stains of input image
stains = ['hematoxylin',  # nuclei stain
          'eosin',        # cytoplasm stain
          'null']         # set to null if input contains only two stains

# create stain matrix
W = np.array([stain_color_map[st] for st in stains]).T

# perform standard color deconvolution
imDeconvolved = htk.preprocessing.color_deconvolution.color_deconvolution(imInput, W)

# Display results
for i in [0]:#, 1:
    plt.figure()
    plt.imshow(imDeconvolved.Stains[:, :, i])
    _ = plt.title(stains[i], fontsize=titlesize)


# In[ ]:


plt.figure()
masked = np.multiply(imDeconvolved.Stains[:, :, 0],~mask)
plt.imshow(masked>1)

