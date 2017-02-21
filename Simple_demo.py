# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 10:35:34 2017

@author: Daniel
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

from builtins import dict
from future import standard_library
standard_library.install_aliases()



if __name__ == "__main__":       
    import numpy as np
    from CNMF4Dendrites import CNMF4Dendrites
    from AuxilaryFunctions import GetCentersData,GetFileName
    from Demo import GetDefaultParams
    import pickle 
    
    #get data 
    from AuxilaryFunctions import GetDataFolder
    import h5py
    temp=h5py.File(GetDataFolder() + 'BaylorV1Axons/quietBlock.mat')
    data=temp["quietBlock"]
    data=np.asarray(data,dtype='float')  
    data=data[:,150:350,150:350] #take only a small patch    

#    T = 1000
#    X = 201
#    Y = 101
#    data = np.random.randn(T, X, Y)
#    centers = np.asarray([[40, 30]])
#    data[:, 30:45, 25:33] += 2*np.sin(np.array(range(T))/200).reshape(-1,1,1)*np.ones([T,15,8])
    
    #Number of components
    NumCent=400 # Max number of centers to import from Group Lasso intialization - if 0, we don't run group lasso
    
    #Get initialization for components center
    cent=GetCentersData(data,NumCent)
    
    #Define CNMF parameters
    cnmf_obj=CNMF4Dendrites(sig=(200,200), verbose=True,adaptBias=True,TargetAreaRatio=[0.01,0.06],
             Connected=True, SigmaMask=[],bkg_per=0.1,iters=100,iters0=[30], mbs=[10], 
             ds=1,lam1_s=10,MergeThreshold_activity=0.8,MergeThreshold_shapes=0.85) 

    #Get initialization for components center
    MSE_array, shapes, activity=cnmf_obj.fit(data,cent)
    
    #Save data
    
    params,params_dict=GetDefaultParams() # get default parameters for dataset
    saveName=GetFileName(params_dict,1)
 
    from io import open
    f = open('NMF_Results/'+saveName, 'wb')
    
    results=dict([['MSE_array',MSE_array], ['shapes',shapes],['activity',activity],['cent',cent],['params',params]])
    pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    f.close()    
    
    from PlotResults import PlotAll
    PlotAll([saveName],params)    
                 
