#!/usr/bin/env python
# coding: utf-8

import numpy as np
import sys
import umap
import warnings
from scipy import sparse, linalg
from scipy.sparse import coo_matrix
import networkx as nx
from sklearn import preprocessing
from sklearn.decomposition import PCA
warnings.filterwarnings('ignore')
from sklearn.preprocessing import normalize, scale
import numba
import igraph
import pandas as pd
import plotly.graph_objects as go
from plotly.graph_objs import *
from numpy.linalg import norm
from scipy.sparse import find
import matplotlib.pyplot as plt
import os
from skimage.draw import polygon
from skimage.measure import label, regionprops#, regionprops_table

from skimage import io

# Given the nonzero pixel values show the mask
def show_patch_from_polygon(filename,x_list,y_list):
    if not (x_list and y_list):
        print('list is empty')
    else:
        xx = np.array(x_list).reshape((len(x_list),1))
        yy = np.array(y_list).reshape((len(y_list),1))
        arr = np.hstack((xx,yy))
        arr -= np.mean(arr,axis=0).astype(int)
        mini = np.min(arr,axis=0)
        arr -= mini.astype(int) # subtract the min to translate the mask 

        row = np.rint(arr[:,0]).astype(int)
        col = np.rint(arr[:,1]).astype(int)
        mtx = coo_matrix((np.ones(row.shape), (row, col)), dtype=bool)

        plt.figure(figsize=(40,40))
        io.imshow(mtx.todense(),cmap='gray')
        plt.savefig(filename+'.png')
    return

# Given a list of patches of segmented nuclei in polygon format, show the masks
def show_patches_parallel(filename):
    x_list = []
    y_list = []
    df = pd.read_csv(filename)
    if ~df.empty:
        cc0 = float(os.path.basename(filename).split(sep='_')[0])
        rr0 = float(os.path.basename(filename).split(sep='_')[1] )

        for cell in df['Polygon'].tolist()[:]: # loop over cells in patch
            lista = list(np.fromstring(cell[1:-1], dtype=float, sep=':')) #list of vertices in polygon
            cc = lista[0::2] # list of x coord of each polygon vertex
            rr = lista[1::2] # list of y coord of each polygon verted
            poly = np.asarray(list(zip(cc,rr)))
            mean = poly.mean(axis=0) 
            poly -= mean 
            # create the nuclear mask
            mask = np.zeros(tuple(np.ceil(np.max(poly,axis=0) - np.min(poly,axis=0)).astype(int))).astype(int) 
            mini = np.min(poly,axis=0)
            poly -= mini # subtract the min to translate the mask 
            cc, rr = polygon(poly[:, 0], poly[:, 1], mask.shape) # get the nonzero mask locations
            mask[cc, rr] = 1 # nonzero pixel entries
            # rescale back to original coordinates
            rr = rr.astype(float);cc = cc.astype(float)
            rr += mini[0]; cc += mini[1]
            rr += mean[0]; cc += mean[1]
            rr += rr0; cc += cc0
            
            # update the list of nonzero pixel entries
            x_list.extend( [int(n) for n in list(rr)] ) 
            y_list.extend( [int(n) for n in list(cc)] )
            
        show_patch_from_polygon(filename,x_list,y_list)
    return

# given the patch filename containing the polygon coordinates, generate morphometrics
def measure_patch_of_polygons(filename,features): 
    data = pd.DataFrame(columns = features) # create empty df to store morphometrics
    df = pd.read_csv(filename)
    if ~df.empty:
        cc0 = float(os.path.basename(filename).split(sep='_')[0]) # the x position is the col position
        rr0 = float(os.path.basename(filename).split(sep='_')[1] ) # the y position is the row position

        for cell in df['Polygon'].tolist()[:]: # loop over cells in patch
            lista = list(np.fromstring(cell[1:-1], dtype=float, sep=':')) #list of vertices in polygon
            cc = lista[0::2] # list of x coord of each polygon vertex
            rr = lista[1::2] # list of y coord of each polygon verted
            poly = np.asarray(list(zip(cc,rr)))
            mean = poly.mean(axis=0) 
            poly -= mean 
            # create the nuclear mask
            mask = np.zeros(tuple(np.ceil(np.max(poly,axis=0) - np.min(poly,axis=0)).astype(int))).astype(int) # build an empty mask spanning the support of the polygon
            mini = np.min(poly,axis=0)
            poly -= mini # subtract the min to translate the mask 
            cc, rr = polygon(poly[:, 0], poly[:, 1], mask.shape) # get the nonzero mask locations
            mask[cc, rr] = 1 # nonzero pixel entries
            label_mask = label(mask)
            try:
                regions = regionprops(label_mask, coordinates='rc')        
            except ValueError:  #raised if array is empty.
                pass
            
            dicts = {}
            keys = features
            for i in keys:
                if i == 'centroid_x':
                    dicts[i] = regions[0]['centroid'][0]+cc0
                elif i == 'centroid_y':
                    dicts[i] = regions[0]['centroid'][1]+rr0
                else:
                    dicts[i] = regions[0][i]
            # update morphometrics data 
            new_df = pd.DataFrame(dicts, index=[0])
            data = data.append(new_df, ignore_index=True)
    data.to_pickle(filename+'.morphometrics.pkl')
    return 

# Show the log-log plot of the edge heterogeneity
def plot_loglog(df,title):
    values, bins = np.histogram(df['diversity'],bins=1000)
    y = values
    x = [0.5*(bins[i]+bins[i+1]) for i in range(len(bins)-1)]

    plt.loglog(x, y,'r.')
    plt.xlabel("edge heterogeneity", fontsize=14)
    plt.ylabel("counts", fontsize=14)
    plt.title(title)
    plt.savefig(title+'.edgeH.loglog.png')
    plt.close()
    #plt.show()
    return

# Show the lognormal distribution of the node heterogeneity
def plot_lognormal(df,title):
    values, bins = np.histogram(np.log2(df['diversity']),bins=100) # take the hist of the log values
    y = values
    x = [0.5*(bins[i]+bins[i+1]) for i in range(len(bins)-1)]

    plt.plot(x, y,'r.')
    plt.xscale('linear')
    plt.yscale('linear')
    plt.xlabel("Log_2 node heterogeneity", fontsize=14)
    plt.ylabel("counts", fontsize=14)
    plt.title(title)
    plt.savefig(title+'.nodeH.lognorm.png')
    plt.close()
    #plt.show()
    return

# Plotly contour visualization
def plotlyContourPlot(fdf,filename):
    # define the pivot tabel for the contour plot
    table = pd.pivot_table(fdf, 
                           values='field', 
                           index=['x_bin'],
                           columns=['y_bin'],
                           aggfunc=np.sum, # take the mean of the entries in the bin
                           fill_value=None)
    
    fig = go.Figure(data=[go.Surface(z=table.values,
                                     x=table.columns.values, 
                                     y=table.index.values,
                                     colorscale='Jet')])
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_z=True))
    fig.update_layout(title=filename, autosize=True,
                      scene_camera_eye=dict(x=1.87, y=0.88, z=-0.64),
                      width=1000, height=1000,
                      margin=dict(l=65, r=50, b=65, t=90)
                    )
    fig.show()
    return

def contourPlot(fdf,N,aggfunc,filename): # Contour visualization
    ratio = fdf.max()[0]/fdf.max()[1] # ratio of max x and y centroids coordinates
    Nx = int(round(ratio*N))
    fdf['x_bin'] = pd.cut(fdf['centroid_x'], Nx, labels=False) # define the x bin label
    fdf['y_bin'] = pd.cut(fdf['centroid_y'], N, labels=False) # define the y bin label

    # define the pivot tabel for the contour plot
    table = pd.pivot_table(fdf, 
                           values='diversity', 
                           index=['x_bin'],
                           columns=['y_bin'],
                           aggfunc=aggfunc, # take the mean or another function of the entries in the bin
                           fill_value=None)

    X=table.columns.values
    Y=table.index.values
    Z=table.values
    Xi,Yi = np.meshgrid(X, Y)

    fig, ax = plt.subplots(figsize=(ratio*10,10))
    cs = ax.contourf(Yi, Xi, Z, 
                     alpha=1.0, 
                     levels=10,
                     cmap=plt.cm.viridis);
    ax.invert_yaxis()
    cbar = fig.colorbar(cs)
    plt.savefig('./'+filename+'.contour.png')
    plt.close()
    
def get_fov(df,row,col):
    fdf = df[(df['fov_row']==row) & (df['fov_col']==col)]
    pos = fdf[fdf.columns[2:4]].to_numpy() # Get the positions of centroids 

    # Building the UMAP graph
    print('Creating the graph')
    nn = fdf.shape[0]//25 #set the number of nn
    print('The connectivity is '+str(nn))
    A = space2graph(pos,nn)
    
    print('Creating the network')
    G = nx.from_scipy_sparse_matrix(A, edge_attribute='weight')
    
    #get the morphological data and rescale the data by std 
    data = scale(fdf[features].to_numpy(), with_mean=False) 

    print('Generating the descriptor')
    num_cores = multiprocessing.cpu_count() # numb of cores
    row_idx, col_idx, values = find(A) #A.nonzero() # nonzero entries
    processed_list = Parallel(n_jobs=num_cores)(delayed(covd_local)(r,data,row_idx,col_idx) 
                                                                for r in range(A.shape[0])
                                                       )

    # Construct the descriptor array
    descriptor = np.zeros((len(processed_list),processed_list[0][1].shape[0]))
    for r in range(len(processed_list)):
        descriptor[r,:] = processed_list[r][1] # covd descriptors of the connected nodes
    
    print('Generating the field')
    #fdf['field'] = covd_gradient(descriptor,row_idx,col_idx,values)
    fdf['field'] = Parallel(n_jobs=num_cores)(delayed(covd_gradient_parallel)(node,descriptor,row_idx,col_idx,values) 
                                                                for node in range(A.shape[0])
                                                       )
    print('Done')
    return fdf

def edge_diversity_parallel(node,neightbors,diversity,fdf):
    edge = []
    node_arr = fdf.iloc[[node]][['centroid_x','centroid_y']].to_numpy()
    nn_arr = fdf.iloc[neightbors][['centroid_x','centroid_y']].to_numpy()
    centroid = 0.5*(node_arr+nn_arr)
    array = np.hstack((centroid,diversity.reshape((diversity.shape[0],1))))
    edge.extend(array.tolist())
    return edge

def covd_gradient_parallel(node,descriptor,row_idx,col_idx,values):
    mask = row_idx == node         # find nearest neigthbors
    delta = norm(descriptor[node,:]-descriptor[col_idx[mask],:],axis=1) # broadcasting to get change at edges
    delta = np.reshape(delta,(1,delta.shape[0]))
    # if you consider graph weights in computing the diversity
    weights = values[mask]
    
    # if you do not consider graph weights in computing the diversity
    #gradient = sum(delta) 
    return (node, col_idx[mask], np.multiply(delta,weights)[0])

def covd_gradient(descriptor,row_idx,col_idx,values):
    global_gradient = []
    for node in range(descriptor.shape[0]):
        print(node)
        mask = row_idx == node         # find nearest neigthbors
        reference = np.repeat([descriptor[node,:]], sum(mask), axis=0)
        delta = norm(reference-descriptor[col_idx[mask],:],axis=1)
        delta = np.reshape(delta,(1,delta.shape[0]))
        weights = values[mask]
        gradient = np.dot(delta,weights)
        global_gradient.append(gradient)
    return global_gradient


def covd_parallel(node,data,row_idx,col_idx): # returns the vec of the logarithm of the cov matrix
    mask = row_idx == node         # find nearest neigthbors
    cluster = np.append(node,col_idx[mask]) # define the local cluster, its size depends on the local connectivity
    C = np.cov(data[cluster,:],rowvar=False)
    L = linalg.logm(C) 
    iu1 = np.triu_indices(L.shape[1])
    vec = L[iu1]
    return (node,vec)

def covd_parallel_sparse(node,data,nn_idx):
    C = np.cov(data[nn_idx[node,:],:],rowvar=False)
    iu1 = np.triu_indices(C.shape[1])
    vec = C[iu1]
    return (node,vec)

def filtering_HE(df):
    #First removing columns
    filt_df = df[df.columns[7:]]
    df_keep = df.drop(df.columns[7:], axis=1)
    #Then, computing percentiles
    low = .01
    high = .99
    quant_df = filt_df.quantile([low, high])
    #Next filtering values based on computed percentiles
    filt_df = filt_df.apply(lambda x: x[(x>quant_df.loc[low,x.name]) & 
                                        (x < quant_df.loc[high,x.name])], axis=0)
    #Bringing the columns back
    filt_df = pd.concat( [df_keep,filt_df], axis=1 )
    #rows with NaN values can be dropped simply like this
    filt_df.dropna(inplace=True)
    return filt_df

def filtering(df):
    #First removing columns
    filt_df = df[["area","perimeter","solidity","eccentricity","circularity","mean_intensity","std_intensity"]]
    df_keep = df.drop(["area","perimeter","solidity","eccentricity","circularity","mean_intensity","std_intensity"], axis=1)
    #Then, computing percentiles
    low = .01
    high = .99
    quant_df = filt_df.quantile([low, high])
    #Next filtering values based on computed percentiles
    filt_df = filt_df.apply(lambda x: x[(x>quant_df.loc[low,x.name]) & 
                                        (x < quant_df.loc[high,x.name])], axis=0)
    #Bringing the columns back
    filt_df = pd.concat( [df_keep,filt_df], axis=1 )
    filt_df['cov_intensity'] = filt_df['std_intensity']/filt_df['mean_intensity']
    #rows with NaN values can be dropped simply like this
    filt_df.dropna(inplace=True)
    return filt_df

def space2graph(positions,nn):
    XY = positions#np.loadtxt(filename, delimiter="\t",skiprows=True,usecols=(5,6))
    mat_XY = umap.umap_.fuzzy_simplicial_set(
        XY,
        n_neighbors=nn, 
        random_state=np.random.RandomState(seed=42),
        metric='l2',
        metric_kwds={},
        knn_indices=None,
        knn_dists=None,
        angular=False,
        set_op_mix_ratio=1.0,
        local_connectivity=2.0,
        verbose=False
    )
    return mat_XY

def getdegree(graph):
    d = np.asarray(graph.degree(weight='weight'))[:,1] # as a (N,) array
    r = d.shape[0]
    return d.reshape((r,1))

def clusteringCoeff(A):
    AA = A.dot(A)
    AAA = A.dot(AA)  
    d1 = AA.mean(axis=0) 
    m = A.mean(axis=0)
    d2 = np.power(m,2)
    num = AAA.diagonal().reshape((1,A.shape[0]))
    denom = np.asarray(d1-d2)
    cc = np.divide(num,denom*A.shape[0]) #clustering coefficient
    r, c = cc.shape
    return cc.reshape((c,r))

def rescale(data):
    newdata = preprocessing.minmax_scale(data,feature_range=(-1, 1),axis=0) # rescale data so that each feature ranges in [0,1]
    return newdata

def principalComp(data):
    pca = PCA(n_components='mle')
    pca.fit(data)
    return pca

def smoothing(W,data,radius):
    S = normalize(W, norm='l1', axis=1) #create the row-stochastic matrix

    smooth = np.zeros((data.shape[0],data.shape[1]))
    summa = data
    for counter in range(radius):
        newdata = S.dot(data)
        summa += newdata
        data = newdata
        if counter == radius-1:
            smooth = summa*1.0/(counter+1)
    return smooth

def covd(mat):
    ims = coo_matrix(mat)                               # make it sparse
    imd = np.pad( mat.astype(float), (1,1), 'constant') # path with zeros

    [x,y,I] = [ims.row,ims.col,ims.data]                # get position and intensity
    pos = np.asarray(list(zip(x,y)))                    # define position vector
    length = np.linalg.norm(pos,axis=1)                 # get the length of the position vectors
    
    Ix = []  # first derivative in x
    Iy = []  # first derivative in y
    Ixx = [] # second der in x
    Iyy = [] # second der in y 
    Id = []  # magnitude of the first der 
    Idd = [] # magnitude of the second der
    
    for ind in range(len(I)):
        Ix.append( 0.5*(imd[x[ind]+1,y[ind]] - imd[x[ind]-1,y[ind]]) )
        Ixx.append( imd[x[ind]+1,y[ind]] - 2*imd[x[ind],y[ind]] + imd[x[ind]-1,y[ind]] )
        Iy.append( 0.5*(imd[x[ind],y[ind]+1] - imd[x[ind],y[ind]-1]) )
        Iyy.append( imd[x[ind],y[ind]+1] - 2*imd[x[ind],y[ind]] + imd[x[ind],y[ind]-1] )
        Id.append(np.linalg.norm([Ix,Iy]))
        Idd.append(np.linalg.norm([Ixx,Iyy]))
    #descriptor = np.array( list(zip(list(x),list(y),list(I),Ix,Iy,Ixx,Iyy,Id,Idd)),dtype='int64' ).T # descriptor
    descriptor = np.array( list(zip(list(length),list(I),Ix,Iy,Ixx,Iyy,Id,Idd)),dtype='int64' ).T     # rotationally invariant descriptor 
    C = np.cov(descriptor)            # covariance of the descriptor
    iu1 = np.triu_indices(C.shape[1]) # the indices of the upper triangular part
    covd2vec = C[iu1]
    return covd2vec


def covd_old(features,G,threshold,quantiles,node_color):
    L = nx.laplacian_matrix(G)
    delta_features = L.dot(features)
    data = np.hstack((features,delta_features)) #it has 16 features

    covdata = [] # will contain a list for each quantile
    graph2covd = []
    for q in range(quantiles):
        covq = [] # will contain a covmat for each connected subgraph
        nodes = [n for n in np.where(node_color == q)[0]]
        subG = G.subgraph(nodes)
        graphs = [g for g in list(nx.connected_component_subgraphs(subG)) if g.number_of_nodes()>=threshold] # threshold graphs based on their size
        print('The number of connected components is',str(nx.number_connected_components(subG)), ' with ',str(len(graphs)),' large enough')
        g_id = 0
        for g in graphs:
            nodeset = list(g.nodes)
            dataset = data[nodeset]
            covmat = np.cov(dataset,rowvar=False)
            covq.append(covmat)

            quant_graph = list([(q,g_id)])
            tuple_nodes = [tuple(g.nodes)]
            new_graph2covd = list(zip(quant_graph,tuple_nodes))
            graph2covd.append(new_graph2covd)
            g_id += 1
        covdata.append(covq)

    return covdata, graph2covd

def get_subgraphs(G,threshold,quantiles,node_quantiles):
    subgraphs = []
    node_set = []
    for f in range(node_quantiles.shape[1]): # for every feature
        for q in range(quantiles):        # for every quantile
            nodes = [n for n in np.where(node_quantiles[:,f] == q)[0]] #get the nodes
            subG = G.subgraph(nodes) # build the subgraph
            graphs = [g for g in list(nx.connected_component_subgraphs(subG)) if g.number_of_nodes()>=threshold] # threshold connected components in subG based on their size
            subgraphs.extend(graphs)

            node_subset = [list(g.nodes) for g in graphs]
            node_set.extend(node_subset)    
    unique_nodes = list(np.unique(np.asarray([node for sublist in node_set for node in sublist])))    
            
    return subgraphs, unique_nodes

def covd_multifeature(features,G,subgraphs):
    L = nx.laplacian_matrix(G)
    delta_features = L.dot(features)
    data = np.hstack((features,delta_features)) #it has 16 features

    covdata = [] # will contain a list for each quantile
    graph2covd = []

    for g in subgraphs:
        nodeset = list(g.nodes)
        dataset = data[nodeset]
        covmat = np.cov(dataset,rowvar=False)
        covdata.append(covmat)

        graph2covd.append(list(g.nodes))
            
    return covdata, graph2covd

def community_covd(features,G,subgraphs):
    L = nx.laplacian_matrix(G)
    delta_features = L.dot(features)
    data = np.hstack((features,delta_features)) #it has 16 features

    covdata = [] # will contain a list for each community
    
    for g in subgraphs:
        nodes = [int(n) for n in g]
        dataset = data[nodes]
        covmat = np.cov(dataset,rowvar=False)
        covdata.append(covmat)
        
    return covdata

def community_covd_woLaplacian(features,G,subgraphs):
    data = features

    covdata = [] # will contain a list for each community
    
    for g in subgraphs:
        nodes = [int(n) for n in g]
        dataset = data[nodes]
        covmat = np.cov(dataset,rowvar=False)
        covdata.append(covmat)
        
    return covdata

def logdet_div(X,Y): #logdet divergence
    (sign_1, logdet_1) = np.linalg.slogdet(0.5*(X+Y)) 
    (sign_2, logdet_2) = np.linalg.slogdet(np.dot(X,Y))
    return np.sqrt( sign_1*logdet_1-0.5*sign_2*logdet_2 )

def airm(X,Y): #affine invariant riemannian metric
    A = np.linalg.inv(linalg.sqrtm(X))
    B = np.dot(A,np.dot(Y,A))
    return np.linalg.norm(linalg.logm(B))

def cluster_morphology(morphology,graph2covd,labels):
    nodes_in_cluster = []
    numb_of_clusters = len(set(labels))
    cluster_mean = np.zeros((numb_of_clusters,morphology.shape[1]))
    if -1 in set(labels):
        for cluster in set(labels):
            nodes_in_cluster.extend([graph2covd[ind] for ind in range(len(graph2covd)) if labels[ind] == cluster ])
            nodes = [item for sublist in nodes_in_cluster for item in sublist]
            ind = int(cluster)+1
            cluster_mean[ind,:] = np.mean(morphology[nodes,:],axis=0)
    else:
        for cluster in set(labels):
            nodes_in_cluster.extend([graph2covd[ind] for ind in range(len(graph2covd)) if labels[ind] == cluster ])
            nodes = [item for sublist in nodes_in_cluster for item in sublist]
            cluster_mean[cluster,:] = np.mean(morphology[nodes,:],axis=0)
    return cluster_mean

def networkx2igraph(graph,nodes,edges):     # given a networkx graph creates an igraph object
    g = igraph.Graph(directed=False)
    g.add_vertices(nodes)
    g.add_edges(edges)
    edgelist = nx.to_pandas_edgelist(graph)
    for attr in edgelist.columns[2:]:
        g.es[attr] = edgelist[attr]
    return g
