# -*- coding: utf-8 -*-
"""
Created on Tue Mar 01 12:04:34 2016

@author: Daniel
"""
import numpy as np

data_diff=np.diff(data,1,axis=0)
data_diff_thresh=np.copy(data_diff)
data_diff_thresh[data_diff_thresh<0]=0

pic_data=np.percentile(data, 95, axis=0)
pic_diff=np.percentile(data, 95, axis=0)
pic_diff_thresh=np.percentile(data, 95, axis=0)

a=3
dims=np.shape(data)
min_dim=np.argmin(dims[1:])
z_slices=range(dims[min_dim+1]) #which z slices to look at slice plots/videos
D=len(z_slices)

#for kk in range(D):        
#            ax = plt.subplot(a,D,kk+1)
#            temp=np.squeeze(np.take(pic_data,(z_slices[kk],),axis=min_dim))
#            ax.imshow(temp,interpolation='None')
#            
#            ax = plt.subplot(a,D,kk+D+1)
#            temp=np.squeeze(np.take(pic_diff,(z_slices[kk],),axis=min_dim))
#            ax.imshow(temp,interpolation='None')            
#            
#            ax = plt.subplot(a,D,kk+2*D+1)
#            temp=np.squeeze(np.take(pic_diff_thresh,(z_slices[kk],),axis=min_dim))
#            ax.imshow(temp,interpolation='None')

  
ax = plt.subplot(a,1,1)
ax.imshow(pic_data,interpolation='None')

ax = plt.subplot(a,1,2)
ax.imshow(pic_diff,interpolation='None')            

ax = plt.subplot(a,1,3)
ax.imshow(pic_diff_thresh,interpolation='None')