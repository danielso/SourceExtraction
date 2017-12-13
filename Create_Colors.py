# -*- coding: utf-8 -*-
"""
Created on Tue May 02 19:35:51 2017

@author: Daniel

#Regress voxels on activity of delected neurons, and then color voxels with this information
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from past.utils import old_div
import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!

import numpy as np
from pylab import load
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from AuxilaryFunctions import GetFileName, GetData  
from scipy.ndimage.filters import gaussian_filter
from Demo import GetDefaultParams

params,params_dict=GetDefaultParams()

data=GetData(params.data_name) #get data 
dims=np.shape(data)
data=data.reshape(dims[0],-1)

L_S=np.prod(dims[1:])
last_rep=params.repeats
  
for rep in range(last_rep): 
    resultsName=GetFileName(params_dict,rep)
    try:
        results=load('NMF_Results/'+resultsName)
    except IOError:
        if rep==0:
            print('results file not found!!')              
        else:
            break     
    shapes=results['shapes']
    activity=results['activity']
    if rep>params.Background_num:
        adaptBias=False
    else:
        adaptBias=True
    L=len(activity)-adaptBias 
    if L==0: #stop if we find files with zero components
        break
    if rep==0:
        dims_shape=shapes[0].shape
    shapes=shapes[:-adaptBias].reshape(L,-1)
    activity=activity[:-adaptBias]
    if rep==0:
        shapes_array=shapes
        activity_array=activity
    else:
        shapes_array=np.append(shapes_array,shapes,axis=0)
        activity_array=np.append(activity_array,activity,axis=0)  


anchor_components=[3,9,11]
K=len(anchor_components) # should be K=3 for RGB
activity_array=activity_array[anchor_components,:]
shapes_array=-np.reshape(shapes_array[anchor_components,:],(K,-1))
activity_array=activity_array-np.reshape(np.mean(activity_array,axis=1),(-1,1))
activity_cov_inv=np.linalg.inv(np.dot(activity_array,activity_array.T))
data=data-np.dot(activity_array.T,shapes_array)
data=data-np.reshape(np.mean(data,axis=1),(-1,1))

#%%
color_pic=np.dot(activity_cov_inv,np.dot(activity_array,data))
color_pic=np.reshape(color_pic,(-1,)+tuple(dims[1:])) 
D=len(dims)-1
color_pic=np.transpose(color_pic, tuple(range(1,D+1)) + (0,) )
#%% Plotting
color_pic=old_div(color_pic,np.percentile(color_pic[color_pic>0],99.5))  #%% normalize denoised data range
color_pic[color_pic>1]=1 

pp = PdfPages('AnchorRegressionColors' + resultsName + '.pdf')
fig=plt.figure(figsize=(18,11))

# 2D data
plt.imshow(color_pic,interpolation='None')
# 3D data
#plt.imshow(color_pic.max(0),interpolation='None')

pp.savefig(fig)
pp.close()
