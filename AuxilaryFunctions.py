# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 13:52:06 2015

@author: Daniel
"""
import numpy as np
    
def GetFileName(params_dict,rep):

    params_Name='mb'+str(params_dict['mbs'][0])+'ds'+str(params_dict['ds'])+'iters'
    params_Name+=str(params_dict['iters0'][0])+'_'+str(params_dict['iters'])+'intervals'+str(params_dict['updateLambdaIntervals'])
    if params_dict['TargetAreaRatio']!=[]: params_Name+='_Area'+str(params_dict['TargetAreaRatio'][0])+'_'+str(params_dict['TargetAreaRatio'][1])
    if params_dict['Background_num']>0: params_Name+='_Bkg'+str(params_dict['Background_num'])
    if params_dict['Connected']: params_Name+='_Connected'
    if params_dict['Deconvolve']: params_Name+='_Deconvolve'
    
    resultsName='NMF_results_'+ params_dict['data_name'] + '_Rep' + str(rep+1) +'of'+ str(params_dict['repeats']) +'_'+params_Name
    return resultsName
    
def GetRandColors(n):
    colors=[]
    for i in range(n):
        color=np.random.uniform(low=0,high=1,size=(1,3))
        colors.append(color/np.sum(color))
    
    return np.array(colors)
    
def max_intensity(x,axis): #get the pixel with maximum total intensity in a color volume
    intensity=np.sum(x,axis=-1)
    ind=np.argmax(intensity,axis=axis)
    tup=np.indices(ind.shape)
    tup_ind=()
    for ii in range(len(tup)+1):
        if ii<axis:
            tup_ind+=(tup[ii],)
        elif ii>axis:
            tup_ind+=(tup[ii-1],)
        else:
            tup_ind+=(ind,)
    return x[tup_ind]
    
    
def GetDataFolder():
    import os
    
    if os.getcwd()[0]=='C':
        DataFolder='G:/BackupFolder/'
    else:
        DataFolder='Data/'
    
    return DataFolder
    
    
def GetData(data_name):
    
    # Input - data_name - string of name of data to load
    # Output - data - Tx(XxYxZ) or Tx(XxY) numpy array 

    import h5py
    from pylab import load  
    from scipy.io import loadmat
    import tifffile as tff

    
    DataFolder=GetDataFolder()
    
    # Fetch experimental 3D data     
    if data_name=='HillmanSmall':
        data=load('data_small')
    elif data_name=='Hillman':
        temp = h5py.File(DataFolder + 'Hillman/150724_mouseRH2d1_data_crop_zig_sm_ds.mat')
        data=temp["moviesub_sm"]
        data=np.asarray(data,dtype='float')
        data = data[10:-70,2:-2,2:-2,2:-2]   # bad values appear near the edges, and everything is moving at the few first and last frames
    elif data_name=='Sophie2D':
        temp=loadmat(DataFolder + 'Sophie2D_drosophila_lightfield/processed_data.mat')
        data=np.transpose(np.asarray(temp['data'],dtype='float'), [2, 0, 1])
        data=data-np.min(data) # takes care of negative values due to detrending
        temp=None
    elif data_name=='Sophie3D':# check data dimenstions
        img = tff.TiffFile(DataFolder + 'Sophie3D_drosophila_lightfield/Sophie3Ddata.tif')
        data=img.asarray()     
    elif data_name=='SophieVoltage3D':# check data dimenstions        
        img = tff.TiffFile(DataFolder + 'Sophie3D_drosophila_lightfield/SophieVoltageData.tif')
        data=img.asarray()   
    elif data_name=='Sophie3DSmall':# check data dimenstions
        img = tff.TiffFile(DataFolder + 'Sophie3D_drosophila_lightfield/Sophie3Ddata_Small.tif')
        data=img.asarray() 
    elif data_name=='SaraSmall':
        data=load(DataFolder + 'Sara19DEC2015/SaraSmall')
    elif data_name=='Sara19DEC2015_w1t1':
        temp = loadmat(DataFolder + 'Sara19DEC2015/processed_data.mat')
        data=temp["data"]
        data=np.asarray(data,dtype='float')  
        data=np.transpose(np.asarray(temp['data'],dtype='float'), [3,0,1, 2])
    elif data_name=='PhilConfocal':
        img= tff.TiffFile(DataFolder + 'Phil24FEB2016/confocal_stack.tif')
        data=img.asarray()
        data=np.transpose(data, [0,2,3,1]) 
        data=data-np.percentile(data, 0.1, axis=0)# takes care of negative values (ands strong positive values) in each pixel
    elif data_name=='PhilConfocal2':
        img= tff.TiffFile(DataFolder + 'Phil14MAR2016/confocal_diffeo.tif')
        data=img.asarray()
        data=np.transpose(data, [0,2,3,1]) 
        data=data-np.percentile(data, 0.1, axis=0)# takes care of negative values (ands strong positive values) in each pixel
    elif data_name=='PhilMFM':
        img= tff.TiffFile(DataFolder + 'Phil24FEB2016/Du_che2.tif')
        data=img.asarray()
#        data=np.asarray(data,dtype='float')  
        data=np.transpose(data, [0,2,3,1])  
        data=data-np.percentile(data, 0.1, axis=0)# takes care of negative values (ands strong positive values) in each pixel    
    elif data_name=='BaylorAxonsSmall':           
        temp=loadmat(DataFolder + 'BaylorV1Axons/data_small')
        data=temp["data"]
        data=np.asarray(data,dtype='float')  
        data=np.transpose(data, [2,0,1])         
        data=data-np.min(data, axis=0)# takes care of negative values (ands strong positive values) in each pixel
    elif data_name=='BaylorAxons':           
        temp=h5py.File(DataFolder + 'BaylorV1Axons/animal9962session3scan4slice1frames5000.mat')
        data=temp["X2"]
        data=np.asarray(data,dtype='float')  
        data=data[:,:200,:200] #take only a small patch
        data=data-np.min(data, axis=0)# takes care of negative values (ands strong positive values) in each pixel
    elif data_name=='BaylorAxonsQuiet':           
        temp=h5py.File(DataFolder + 'BaylorV1Axons/quietBlock.mat')
        data=temp["quietBlock"]
        data=np.asarray(data,dtype='float')  
        data=data[:,150:350,150:350] #take only a small patch
#        ds=3  #downscale time by this factor
#        data=data[:int(len(data) / ds) * ds].reshape((-1, ds) + data.shape[1:]).mean(1)
        data=data-np.min(data, axis=0)# takes care of negative values (ands strong positive values) in each pixel
    elif data_name=='BaylorAxonsJiakun2':           
        temp=h5py.File(DataFolder + 'BaylorV1Axons/11273_2_1(1).mat')
        data=temp["X"]
        data=np.asarray(data,dtype='float')  
        data=data[:,200:450,200:450] #take only a small patch
        data=data-np.min(data, axis=0)# takes care of negative values (ands strong positive values) in each pixel
    elif data_name=='BaylorAxonsJiakun1':           
        temp=h5py.File(DataFolder + 'BaylorV1Axons/11273_2_1(3).mat')
        data=temp["X"]
        data=np.asarray(data,dtype='float')  
        data=data[:,200:450,200:450] #take only a small patch
        data=data-np.min(data, axis=0)# takes care of negative values (ands strong positive values) in each pixel
    elif data_name=='BaylorAxonsActive':           
        temp=h5py.File(DataFolder + 'BaylorV1Axons/activeBlock.mat')
        data=temp["activeBlock"]
        data=np.asarray(data,dtype='float')  
        data=data[:,150:350,150:350] #take only a small patch
        data=data-np.min(data, axis=0)# takes care of negative values (ands strong positive values) in each pixel
    elif data_name=='Ja_Ni_ds3':           
        img= tff.TiffFile(DataFolder + 'NaJi_Dendrites\sparse_dendrites\Sparse-dendrite_Aaron.tif')
        data=img.asarray()
        data=np.transpose(data, [0,2,1]) 
        data=data[:,100:400]
        ds=3  
        data=data[:int(len(data) / ds) * ds].reshape((-1, ds) + data.shape[1:]).mean(1)
        data=data-np.min(data, axis=0)# takes care of negative values (ands strong positive values) in each pixel
      
    else:
        print 'unknown dataset name!'
    return data
    
def GetCentersData(data,NumCent,data_name=[],rep=0): 
    """
    Get intialization centers using group lasso
    
    Input
    ----------
    data : array, shape (T, X,Y,(,Z))
        data
    data_name: string
        dataset name so we can save load previous center data
    NumCent: integer
        number centers to extract
    rep: integer
        repetition number
        
    Output
    ----------
    activity: array, shape (L,T)
        extracted temporal components
    
    """
    from numpy import  array,percentile    
    from BlockGroupLasso import gaussian_group_lasso, GetCenters
    from pylab import load
    import os
    import cPickle
        
    
    DataFolder=GetDataFolder()    
    center_file_name=DataFolder + '/centers_'+ str(data_name) + '_rep_' + str(rep)
    if NumCent>0:
        if data_name==[] or os.path.isfile(center_file_name)==False:
            if data.ndim==3:
                sig0=(2,2)
            else:
                sig0=(2,2,2)
                
            TargetRange = [0.1, 0.2]    
            lam = 500
            ds= 50 #downscale time for group lasso        
            NonNegative=True
    
            downscaled_data=data[:int(len(data) / ds) * ds].reshape((-1, ds) + data.shape[1:]).max(1) #for speed ups
            x = gaussian_group_lasso(downscaled_data, sig0, lam,NonNegative=NonNegative, TargetAreaRatio=TargetRange, verbose=True, adaptBias=False)
            pic_x = percentile(x, 95, axis=0)

                ######
            # centers extracted from fista output using RegionalMax
            cent = GetCenters(pic_x)
            print np.shape(cent)[0]
                
            # Plot Results
#            import matplotlib.pyplot as plt
#            pic_data = np.percentile(data, 95, axis=0)
#            plt.figure(figsize=(12, 4. * data.shape[1] / data.shape[2]))
#            ax = plt.subplot(131)
#            ax.scatter(cent[1], cent[0],  marker='o', c='white')
#            plt.hold(True)
#            ax.set_title('Data + centers')
#            ax.imshow(pic_data.max(2))
#            ax2 = plt.subplot(132)
#            ax2.scatter(cent[1], cent[0], marker='o', c='white')
#            ax2.imshow(pic_x.max(2))
#            ax2.set_title('Inferred x')
#            ax3 = plt.subplot(133)
#            ax3.scatter(cent[1], cent[0],   marker='o', c='white')
#            ax3.imshow(pic_x.max(2))
#            ax3.set_title('Denoised data')
#            plt.show()
#        
            # save results
            f = file(center_file_name, 'wb')
        
            cPickle.dump(cent, f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()
        else:
            if data_name!=[]:
                cent=load(center_file_name)
                
        new_cent=(array(cent)[:-1]).T                
        new_cent=new_cent[:NumCent] #just give strongest centers
    else:
        new_cent=np.reshape([],(0,data.ndim-1))
        
    return new_cent
    
    
def ThresholdData(data): #obsolete
    from BlockLocalNMF_AuxilaryFunctions import GetSnPSDArray
    
    sn_target,sn_std= GetSnPSDArray(data)
    data[data<sn_target]=0
    return data

def SuperVoxelize(data): #obsolete

    import scipy.io
    data=np.transpose(data,axes=[1,2,3,0])
    shape_data=data.shape
    data=np.reshape(data,(np.prod(shape_data[0:3]),shape_data[3]))
    #load mask
    DataFolder=GetDataFolder()
    
    temp=scipy.io.loadmat(DataFolder + 'oversegmentationForDaniel_daniel0_4e-4.mat')
    mask=temp["L"]            
    mask=np.transpose(mask,axes=[2,1,0])
    mask=np.reshape(mask,(1,np.size(mask)))
    ma=np.max(mask)
    mi=np.min(mask)
    for ii in range(mi,ma+1):
        ind=(mask==ii)
        trace=np.dot(ind,data)
        ind=np.ravel(ind)
        data[ind]=trace
        print ii
    
    data=np.reshape(data,shape_data)
    data=np.transpose(data,axes=[3,0,1,2])
    return data
    
#%% post processing

from scipy.sparse.csgraph import connected_components
from collections import defaultdict
from scipy.ndimage.filters import gaussian_filter

def MergeComponents(shapes,activity,L,threshold,sig):
    # merge nearby components (with spatial components within sig of each other), and high (>threshold) spatial or temporal correlation
    # Inputs:
    # shapes - numpy array with all shape components - size (L,X,Y(,Z)) 
    # activity - numpy array with all activity components - size (L,T)
    # L - int, number of background components.
    # threshold - float, cutoff for merging
    # sig - float, cutoff of spatial proximity
    # Outputs:  
    # shapes - numpy array with all shape components - size (L,X,Y(,Z)) 
    # activity - numpy array with all activity components - size (L,T)
    # L - int, number of background components.
   
    if len(shapes)==0:
       return shapes,activity,L
    dims_shape=shapes[0].shape
    D=len(dims_shape)
    masks=0*shapes[:L]

    for ll in range(L): 
        temp=0*shapes[ll]
        temp[shapes[ll]>0]=1
        temp2=gaussian_filter(temp,[sig]*D)
        masks[ll]=temp2>0.5/(np.sqrt(2*np.pi)*sig)**D

    masks=masks.reshape(L,-1)
    nearby_shapes=np.dot(masks,masks.T)>0

    activity_cov=np.dot(activity[:L],activity[:L].T)
    activity_vars=np.diag(activity_cov).reshape(-1,1)
    activity_corr=activity_cov/np.sqrt(np.dot(activity_vars,activity_vars.T))
    
    shapes_array=shapes[:L].reshape(L,-1)
    shapes_cov=np.dot(shapes_array,shapes_array.T)
    shape_vars=np.diag(shapes_cov).reshape(-1,1)
    shapes_corr=np.nan_to_num(shapes_cov/np.sqrt(np.dot(shape_vars,shape_vars.T)))

    merge_ind=(activity_corr>threshold)|(shapes_corr>threshold)
    merge_ind[nearby_shapes==0]=0
    num,labels=connected_components(merge_ind)
    
    #im2=plt.imshow(merge_ind, interpolation='none',cmap=cmap)
    #plt.colorbar(im2)
    
    comp2merge = defaultdict(list)
    for ii,item in enumerate(labels):
        comp2merge[item].append(ii)
    comp2merge = {k:v for k,v in comp2merge.items() if len(v)>1}

    deleted_indices=[]
    for item in comp2merge.itervalues():    
        for kk in item[1:]:
            deleted_indices.append(kk)        
            shapes[item[0]]+=shapes[kk]
            
    deleted_indices.sort()
    for ll in deleted_indices[::-1]:
        activity=np.delete(activity,(ll),axis=0)
        shapes=np.delete(shapes,(ll),axis=0)
    
    L=L-len(deleted_indices)
    return shapes,activity,L
    
def PruneComponents(shapes,activity,L,TargetAreaRatio=[],deleted_indices=[]):
    # Prune unecessary components 
    # Inputs:
    # shapes - numpy array with all shape components - size (L,X,Y(,Z)) 
    # activity - numpy array with all activity components - size (L,T)
    # L - int, number of background components.
    # TargetAreaRatio - list with 2 numbers, tagrget sparsity range, outside we delete component
    # deleted_indices - list, delete only specific components in this indice list
    # Outputs:
    # new_shapes- numpy array with new shape components - size (L,X,Y(,Z))
    # new_activity- numpy array with new activity components - size (L,T)
    # L - number of new components
    # all_local_max - location of (unique) smoothed local maxima of all components
    
    if deleted_indices==[]:
        # If sparsity is too high or too low        
        cond1=0
        cond2=0
        for ll in range(L):
            S_normalization=np.sum(shapes[ll])
            A_normalization=np.sum(activity[ll])
            if ((A_normalization<=0) or (S_normalization<=0)):
                cond0=True
            else:
                cond0=False
            if TargetAreaRatio!=[]:
                cond1=np.mean(shapes[ll]>0)<TargetAreaRatio[0]
                cond2=np.mean(shapes[ll]>0)>(TargetAreaRatio[1]*3)
            if cond0 or cond1 or cond2:
                deleted_indices.append(ll) 
            
        # If highly correlated with motion artifact?
        
        # If shape overlaps with edge too much?
        
        # constraint on L2 of shape?
        
        # constraint on Euler number?

    for ll in deleted_indices[::-1]:
        activity=np.delete(activity,(ll),axis=0)
        shapes=np.delete(shapes,(ll),axis=0)
    
    L=L-len(deleted_indices)
    return shapes,activity,L

def SplitComponents(shapes,activity,NumBKG):
    # split components according to watershed around peaks
    # Inputs:
    # shapes - numpy array with all shape components - size (L,X,Y(,Z)) 
    # activity - numpy array with all activity components - size (L,T)
    # NumBKG - int, number of background components.
    # Outputs:
    # new_shapes- numpy array with new shape components - size (L,X,Y(,Z))
    # new_activity- numpy array with new activity components - size (L,T)
    # L - number of new components
    # all_local_max - location of (unique) smoothed local maxima of all components
    
    from scipy.ndimage.measurements import label
    from skimage.morphology import watershed
    from skimage.feature import peak_local_max
    from scipy.ndimage.filters import gaussian_filter

    sig=[1,1,1] # convolve with this gaussian, before finding peaks
    too_many_peaks=5 #how much is too many peaks? remove component with this many peaks
    split_background=False
    
    if split_background==False:
        L=len(shapes)-NumBKG
    else:
        L=len(shapes)
        
    new_shapes=np.zeros((0,)+np.shape(shapes)[1:])
    new_activity=np.zeros((0,)+np.shape(activity)[1:])
    all_local_max=np.zeros((0,3))
    all_markers=0
    
    for ll in range(L): 
        temp=np.copy(shapes[ll])
        temp=gaussian_filter(temp,sig)
        local_maxi = peak_local_max(temp, exclude_border=False, indices=False)
        local_maxi_loc = peak_local_max(temp, exclude_border=False, indices=True)
        markers,num_markers = label(local_maxi)
        
        all_markers=all_markers+local_maxi
    #    print ll,num_markers
        nonzero_mask=temp>0
        if np.sum(nonzero_mask)>9:
            labels = watershed(-temp, markers, mask=nonzero_mask)        #watershe regions
    #        for kk in range(Z):
    #            plt.subplot(L,Z,ll*Z+kk)
    #            plt.imshow(labels[:,:,kk])
            if num_markers<=too_many_peaks or ((ll>=L-NumBKG) and (NumBKG>0)): #throw away any component with too many peaks, except the background
                all_local_max=np.append(all_local_max,local_maxi_loc,axis=0)            
                for pp in range(num_markers):                
                    temp=np.copy(shapes[ll])
                    temp[labels!=(pp+1)]=0
                    new_shapes=np.append(new_shapes,np.reshape(temp,(1,)+np.shape(temp)),axis=0)
                    new_activity=np.append(new_activity,np.reshape(activity[ll],(1,)+np.shape(activity[ll])),axis=0)
    
    for pp in range(NumBKG):
        new_shapes=np.append(new_shapes,np.reshape(shapes[-pp-1],(1,)+np.shape(shapes[-pp-1])),axis=0)
        new_activity=np.append(new_activity,np.reshape(activity[-pp-1],(1,)+np.shape(activity[-pp-1])),axis=0)
    
    L=len(new_shapes)
    
    return new_shapes,new_activity,L,all_local_max
    
def ThresholdShapes(shapes,adaptBias,TargetAreaRatio,MaxRatio):
    #%% Threshold shapes
#    TargetAreaRatio - list with 2 components, 
#                      target area for the sparsity of largest connected component
#    MaxRatio - float in [0,1], 
#                if TargetAreaRatio =[], then we threshold according to value of MaxRatio*max(shapes[ll])
#    adaptBias - should we skip last component

    from scipy.ndimage.measurements import label
    from BlockLocalNMF_AuxilaryFunctions import GetSnPSD
    rho=2 #exponential search parameter
    L=len(shapes)-adaptBias

    if TargetAreaRatio!=[]:
        for ll in range(L): 
            threshold=0.1
            threshold_high=-1
            threshold_low=-1
            while True:    
                temp=np.copy(shapes[ll])
                temp[temp<threshold]=0
                temp[temp>=threshold]=1
                # connected components target
        #        CC,num_CC=label(temp)
        #        sz=0
        #        ind_best=0
        #        for nn in range(num_CC):
        #            current_sz=np.count_nonzero(CC[CC==nn])
        #            if current_sz>sz:
        #                ind_best=nn
        #                sz=current_sz
        #        print threshold,sz/sz_all
        #        if ((sz/sz_all < TargetAreaRatio[0]) and (sz!=0)) or (np.sum(temp)==0):
        #            threshold_high = threshold
        #        elif (sz/sz_all > TargetAreaRatio[1]) or (sz==0):
        #            threshold_low = threshold
        #        else:
        #            temp[CC!=ind_best]=0
        #            shapes[ll]=np.copy(temp)
        #            break
                # sparsity target
                if (np.mean(temp) < TargetAreaRatio[0]):
                    threshold_high = threshold
                elif (np.mean(temp) > TargetAreaRatio[1]):
                    threshold_low = threshold
                else:
                    print np.mean(temp)
                    temp=np.copy(shapes[ll])
                    temp[temp<threshold]=0
                    shapes[ll]=np.copy(temp)
                    break
        
                if threshold_high == -1:
                    threshold = threshold * rho
                elif threshold_low == -1:
                    threshold = threshold / rho
                else:
                    threshold = (threshold_high + threshold_low) / 2
        
        for ll in range(L): 
            temp=np.copy(shapes[ll])
            CC,num_CC=label(temp)
            sz=0
            for nn in range(num_CC):
                current_sz=np.count_nonzero(CC[CC==nn])
                if current_sz>sz:
                    ind_best=nn
                    sz=current_sz
            temp[CC!=ind_best]=0
            shapes[ll]=np.copy(temp)
    else:
        for ll in range(L): 
            temp=np.copy(shapes[ll])
            if MaxRatio!=[]:
                threshold=MaxRatio*np.max(temp)
            else:
                temp=shapes[ll]
                temp2=temp[temp>0].ravel()
                threshold= GetSnPSD(temp2)
            temp[temp<threshold]=0
            shapes[ll]=temp
    return shapes
        
    
## Depricated
    
# plot random projection of data
#    C=3 #RGB colors
#    dims=np.shape(data)
#    W=np.random.randn(C,dims[0])  
#    proj_data=np.dot(W,np.transpose(data,[1,2,0,3]))
#    proj_data=np.transpose(proj_data,[1,2,3,0])
#    pp = PdfPages('RandProj.pdf')
#    fig=plt.figure(figsize=(18 , 11))
#    for dd in range(dims[2]):
#        plt.subplot(8,8,dd+1)
#        plt.imshow(proj_data[:,dd],interpolation='none')
#    
#    pp.savefig(fig)
#    pp.close()
#    
    