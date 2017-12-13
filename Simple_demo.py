# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 10:35:34 2017

@author: Daniel
"""

# Import Stuff
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import dict
from future import standard_library
standard_library.install_aliases()

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!

import numpy as np
from CNMF4Dendrites import CNMF4Dendrites
from AuxilaryFunctions import GetCentersData,GetFileName


#get data - can do instead: data=GetData('YairDendrites')   
#from AuxilaryFunctions import GetDataFolder
#import tifffile as tff    
#img= tff.TiffFile( GetDataFolder() +'YairDendrites\stable_7_ds_23.tif')       
#data=img.asarray()
#data=np.asarray(data,dtype='float')  
#data=data-np.min(data, axis=0)# takes care of negative values (ands strong positive values) in each pixel

from pylab import load  
data=load('data_small')

#Get initialization for components center
NumCent=10 # Max number of centers to import from Group Lasso intialization - if 0, we don't run group lasso
cent=GetCentersData(data,NumCent,mbs=50)

#Define CNMF parameters
mbs=[1] # temporal downsampling of data in intial phase of NMF
ds=1 # spatial downsampling of data in intial phase of NMF. Ccan be an integer or a list of the size of spatial dimensions
TargetAreaRatio=[0.001,0.03] # target sparsity range for spatial components
iters0=[10] # number of intial NMF iterations, in which we downsample data and add components
iters=10 # number of main NMF iterations, in which we fine tune the components on the full data
lam1_s=10# l1 regularization parameter initialization (for increased sparsity). If zero, we have no l1 sparsity penalty
bkg_per=20 # intialize of background shape at this percentile (over time) of video
sig=(5,5,5) # estiamte size of neuron - bounding box is 3 times this size. If larger then data, we have no bounding box.
MergeThreshold_activity=0.95#merge components if activity is correlated above the this threshold (and sufficiently close)
MergeThreshold_shapes=0.99 #merge components if activity is correlated above the this threshold (and sufficiently close)

Connected=True # should we constrain all spatial component to be connected?
SigmaMask=3  # if not [], then update masks so that they are non-zero a radius of SigmaMasks around previous non-zero support of shapes

cnmf_obj=CNMF4Dendrites(sig=sig, verbose=True,adaptBias=True,TargetAreaRatio=TargetAreaRatio,
         Connected=Connected, SigmaMask=SigmaMask,bkg_per=bkg_per,iters=iters,iters0=iters0, mbs=mbs, 
         ds=ds,lam1_s=lam1_s,MergeThreshold_activity=MergeThreshold_activity,MergeThreshold_shapes=MergeThreshold_shapes)  

#Define CNMF parameters
MSE_array, shapes, activity=cnmf_obj.fit(data,cent)

#Save data and Plot Results - optional
from Demo import GetDefaultParams
import pickle 

## !!!! use same data_name in GetDefaultParams  !!! ###
params,params_dict=GetDefaultParams() # pack parameters in a convient way (all parameters have the same values, defined similarly inside the function)
saveName=GetFileName(params_dict,0)

from io import open
f = open('NMF_Results/'+saveName, 'wb')

results=dict([['MSE_array',MSE_array], ['shapes',shapes],['activity',activity],['cent',cent],['params',params]])
pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
f.close()    

from PlotResults import PlotAll
PlotAll([saveName],params)    
                 
