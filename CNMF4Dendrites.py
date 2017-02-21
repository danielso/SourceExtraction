# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 08:32:34 2017

@author: Daniel
"""

from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import int
from future import standard_library
standard_library.install_aliases()
from builtins import map
from builtins import range
from past.utils import old_div
from numpy import asarray, percentile, zeros, ones, ix_, arange, exp, prod, repeat
import numpy as np
from BlockLocalNMF_AuxilaryFunctions import  HALS4activity, HALS4shape,RenormalizeDeleteSort,addComponent,GetBox, \
RegionAdd,RegionCut,DownScale,LargestConnectedComponent,LargestWatershedRegion,SmoothBackground,ExponentialSearch,GrowMasks
from AuxilaryFunctions import PruneComponents,MergeComponents

    
class CNMF4Dendrites(object):
    """
    Source extraction using constrained non-negative matrix factorization, specialized for denderites
    """
    
    def __init__(self,sig, verbose=False,adaptBias=True,TargetAreaRatio=[],
             Connected=False, SigmaMask=[],bkg_per=20,iters=10,iters0=[30], mbs=[1], 
             ds=1,lam1_s=0,MergeThreshold_activity=1,MergeThreshold_shapes=1,
             shapes=[],activity=[]):         
        """
        Parameters
        ----------   
        sig : array, shape (D,)
            size of the gaussian kernel in different spatial directions
        verbose : boolean
            print progress and record MSE if true (about 2x slower)
        adaptBias : boolean
            subtract rank 1 estimate of bias (background)
        TargetAreaRatio : list of length 2
            Lower and upper bounds on sparsity of non-background components
        Connected: boolean
            impose connectedness of spatial component by keeping only the largest non-zero connected component in each iteration of HALS
        Deconvolve : boolean
            Deconvolve activity to get smoothed (denoised) calcium trace. This is done only on the main itreations, and if FineTune=True
        SigmaMask : scalar or empty
             if not [], then update masks so that they are non-zero a radius of SigmaMasks around previous non-zero support of shapes
        bkg_per : float in the range [0,100]
            the background is intialized at this height (percentrilce image)
        iters : int
            number of final iterations on whole data
        iters0 : list
            numbers of initial iterations on subset
        mbs : list
            minibatchsizes for temporal downsampling 
        ds : int or list
            factor for spatial downsampling, can be an integer or a list of the size of spatial dimensions
        lam1_s : float
            L_1 regularization constant for sparsity of shapes
        MergeThreshold_activity: float between 0 and 1
            merge components if activity is correlated above the this threshold (and sufficiently close)
        MergeThreshold_shapes: float between 0 and 1
            merge components if shapes are correlated above the this threshold (and sufficiently close)
        shapes: array of size Lx(XxY(xZ)
            spatial components initialization
        activity: array of size LxT
            temporal components initialization
    
        """
        
        self.sig=sig
        self.verbose=verbose
        self.adaptBias=adaptBias
        self.TargetAreaRatio=TargetAreaRatio
        self.Connected=Connected
        self.SigmaMask=SigmaMask
        self.bkg_per=bkg_per
        self.iters=iters
        self.iters0=iters0 
        self.mbs=mbs
        self.ds=ds
        self.lam1_s=lam1_s
        self.MergeThreshold_activity=MergeThreshold_activity
        self.MergeThreshold_shapes=MergeThreshold_shapes      
        
        
        self.shapes=shapes
        self.activity=activity
        
    def fit(self,data,centers):
            
        """    
        Parameters
        ----------
        self: CNMF4Dendrites object    
        data : array, shape (T, X, Y[, Z])
            block of the data
        centers : array, shape (L, D)
            L centers of suspected neurons where D is spatial dimension (2 or 3)
            
        Returns
        -------
        MSE_array : list (empty if verbose is False)
            Mean square error during algorithm operation
        shapes : array, shape (L+adaptBias, X, Y (,Z))
            the neuronal shape vectors: empty if no components found, and if adaptBias=1, then first component is background component
        activity : array, shape (L+adaptBias, T)
            the neuronal activity for each shape (empty if no components found), and if adaptBias=1, then first component is background component
        """
        

        # unpack CNMF parameters
        sig=self.sig
        verbose=self.verbose
        adaptBias=self.adaptBias
        TargetAreaRatio=self.TargetAreaRatio
        Connected=self.Connected
        SigmaMask=self.SigmaMask
        bkg_per=self.bkg_per
        iters=self.iters
        iters0=self.iters0
        mbs=self.mbs
        ds=self.ds
        lam1_s=self.lam1_s
        MergeThreshold_activity=self.MergeThreshold_activity
        MergeThreshold_shapes=self.MergeThreshold_shapes  

        # Set addtional internal parameters
        NonNegative=True  # should we constrain activity and shapes to be non-negative?
        FinalNonNegative=True # should we constrain activity to be non-negative at final iteration?
        WaterShed=False  # should we constrain all spatial component to have only one watershed component?
        MedianFilt=False # should we perfrom median filtering of shape?
        SmoothBkg=False # Should we cut out out peaks from background component?
        lam1_t=0 #L_1 regularization constant for sparsity of activity
        lam2_s=0 # L_2 regularization constant for sparsity of shapes
        lam2_t=0 # L_2 regularization constant for sparsity of time
        FineTune=False # Should use the full data at the main iterations (to fine tune shapes)? If not then just extract the activity and not the shapes from the full data
        
        updateLambdaIntervals=2 # update sparsity parameter every updateLambdaIntervals iterations
        addComponentsIntervals=1 # in initial NMF phase, add new component every updateLambdaIntervals*addComponentsIntervals iterations
        updateRhoIntervals=1 # in main NMF phase, update sparsity learning speed (Rho) every updateLambdaIntervals*updateRhoIntervals iterations
             
        
        # Initialize Parameters
        dims = data.shape # data dimensions
        D = len(dims) #number of data dimensions
        R = 3 * asarray(sig)  # size of bounding box is 3 times size of neuron
        L = len(centers) # number of components (not including background)
        inner_iterations=10 # number of iterations in inners loops
        shapes = [] #array of spatial components
        mask = [] # binary array, support of spatial components
        boxes = zeros((L, D - 1, 2), dtype=int) #initial support of spatial components
        MSE_array = [] #CNMF residual error
        mb = mbs[0] if iters0[0] > 0 else 1 
        activity = zeros((L, old_div(dims[0], mb))) #array of temporal components
        lam1_s0=np.copy(lam1_s) #intial spatial sparsity (l1) parameters
        if TargetAreaRatio!=[]:
            if TargetAreaRatio[0]>TargetAreaRatio[1]:            
                print('WARNING -  TargetAreaRatio[0]>TargetAreaRatio[1] !!!')
        if iters0[0] == 0:
            ds = 1
    
            
    ### Initialize shapes, activity, and residual ###        
        
        data0,dims0=DownScale(data,mb,ds) #downscaled data and dimensions
        if isinstance(ds,int):
            ds=ds*np.ones(D-1)
    
        if D == 4: #downscale activity
            activity = data0[:, list(map(int, old_div(centers[:, 0], ds[0]))), list(map(int, old_div(centers[:, 1], ds[1]))),
                  list(map(int, old_div(centers[:, 2], ds[2])))].T
        else:
            activity = data0[:, list(map(int, old_div(centers[:, 0], ds[0]))), list(map(int, old_div(centers[:, 1], ds[1])))].T
            
        data0 = data0.reshape(dims0[0], -1) #reshape data0 to more convient timexspace form
        Energy0=np.sum(data0**2,axis=0) #data0 energy per pixel
        data0sum=np.sum(data0,axis=0) # for sign check later
    
        data = data.astype('float').reshape(dims[0], -1) #reshape data to more convient timexspace form
        datasum=np.sum(data,axis=0)# for sign check later
        
        # float is faster than float32, presumable float32 gets converted later on
        # to float again and again
        Energy=np.sum((data**2),axis=0) #data energy per pixel
        
        # extract shapes and activity from given centers
        for ll in range(L):
            boxes[ll] = GetBox(old_div(centers[ll], ds), old_div(R, ds), dims0[1:])
            temp = zeros(dims0[1:])
            temp[[slice(*a) for a in boxes[ll]]]=1
            mask += np.where(temp.ravel())
            temp = [old_div((arange(int(old_div(dims[i + 1], ds[i]))) -int( old_div(centers[ll][i], ds[i]))) ** 2, (2 * (old_div(sig[i], ds[i])) ** 2))
                    for i in range(D - 1)]
            temp = exp(-sum(ix_(*temp)))
            temp.shape = (1,) + dims0[1:]
            temp = RegionCut(temp, boxes[ll])
            shapes.append(temp[0])
        S = zeros((L + adaptBias, prod(dims0[1:]))) #shape component
        for ll in range(L):
            S[ll] = RegionAdd(
                zeros((1,) + dims0[1:]), shapes[ll].reshape(1, -1), boxes[ll]).ravel()
        if adaptBias:
            # Initialize background as bkg_per percentile
            S[-1] = percentile(data0, bkg_per, 0)
            activity = np.r_[activity, ones((1, dims0[0]))]
        
        lam1_s=lam1_s0*np.ones_like(S)*mbs[0] #intialize sparsity parameters
    
    
    ### Get shape estimates on subset of data ###
        if iters0[0] > 0:
            for it in range(len(iters0)):
                sn_target=np.zeros(prod(dims0[1:]))
                sn_std=sn_target
                MSE_target = np.mean(sn_target**2)
                ES=ExponentialSearch(lam1_s) #object to update sparsity parameters
                lam1_s=ES.lam
                for kk in range(iters0[it]):
                    # update sparisty parameters     
                    if kk%updateLambdaIntervals==0:                 
                        sn=old_div(np.sqrt(Energy0-2*np.sum(np.dot(activity,data0)*S,axis=0)+np.sum(np.dot(np.dot(activity,activity.T),S)*S,axis=0)),dims0[0]) # efficient way to calcuate MSE per pixel
            
                        delta_sn=sn-sn_target # noise margin
                        signcheck=(data0sum-np.dot(np.sum(activity.T,axis=0),S))<0
                        
                        if len(S)==0:
                            spars=0
                        else:
                            spars=np.mean(S>0,axis=1)
                            
                            temp=repeat(delta_sn.reshape(1,-1),L+adaptBias,axis=0) 
        
                            if TargetAreaRatio==[]:  
                                cond_decrease=temp>sn_std
                                cond_increase=temp<-sn_std
                            else:
                                if adaptBias:
                                    spars[-1]=old_div((TargetAreaRatio[1]+TargetAreaRatio[0]),2) # ignore sparsity target for background (bias) component  
                                temp2=repeat(spars.reshape(-1,1),len(S[0]),axis=1)
                                cond_increase=np.logical_or(temp2>TargetAreaRatio[1],temp<-sn_std)
                                cond_decrease=np.logical_and(temp2<TargetAreaRatio[0],temp>sn_std)
            
                            ES.update(cond_decrease,cond_increase)    
                            lam1_s=ES.lam
                            
                        #Print residual error and additional information
                        MSE = np.mean(sn**2)
                        
                        if verbose and L>0:                       
                            print(' MSE = {0:.6f}, Target MSE={1:.6f},Sparsity={2:.4f},lam1_s={3:.6f}'.format(MSE,MSE_target,np.mean(spars[:L]),np.mean(lam1_s)))
                        
                        #add a new component
                        if (kk%addComponentsIntervals==0) and (kk!=iters0[it]-1):
                            
                            delta_sn[signcheck]=-float("inf") # residual should not have negative pixels
                            new_cent=np.argmax(delta_sn) #should I smooth the data a bit first?
                            MSE_std=np.mean(sn_std**2)
                            checkNoZero= not((0 in np.sum(activity,axis=1)) and (0 in np.sum(S,axis=1)))
                            if ((MSE-MSE_target>2*MSE_std) and checkNoZero and (delta_sn[new_cent]>sn_std[new_cent])):                            
                                S, activity, mask,centers,boxes,L=addComponent(new_cent,data0,dims0,old_div(R,ds),S, activity, mask,centers,boxes,adaptBias)
                                new_lam=lam1_s0*np.ones_like(data0[0,:]).reshape(1,-1)
                                lam1_s=np.insert(lam1_s,0,values=new_lam,axis=0)
                                ES=ExponentialSearch(lam1_s) #we need to restart exponential search each time we add a component
                                
                    #apply additional constraints/processing
                    S = HALS4shape(data0, S, activity,mask,lam1_s,lam2_s,adaptBias,inner_iterations)
                    
                    if Connected==True:
                        S=LargestConnectedComponent(S,dims0,adaptBias)
                    if WaterShed==True:
                        S=LargestWatershedRegion(S,dims0,adaptBias)
                    activity = HALS4activity(data0, S, activity,NonNegative,lam1_t,lam2_t,dims0,[],inner_iterations)                                
                    if SigmaMask!=[]:
                        mask=GrowMasks(S,mask,boxes,dims0,adaptBias,SigmaMask)
                    S, activity, mask,centers,boxes,ES,L=RenormalizeDeleteSort(S, activity, mask,centers,boxes,ES,adaptBias,MedianFilt)
                    lam1_s=ES.lam
                    if SmoothBkg==True:
                        S=SmoothBackground(S,dims0,adaptBias,tuple(old_div(np.array(sig),np.array(ds))))
                    
                    print('Subsampled iteration',kk,'it=',it,'L=',L)
                
                # use next (smaller) value for temporal downscaling
                if it < len(iters0) - 1:
                    mb = mbs[it + 1]
                    data0 = data[:len(data) / mb * mb].reshape(-1, mb, prod(dims[1:])).mean(1)
                    if D==4:
                        data0 = data0.reshape(len(data0), int(old_div(dims[1], ds[0])), ds[0], int(old_div(dims[2], ds[1])), ds[1],
                                              int(old_div(dims[3], ds[2])), ds[2]).mean(-1).mean(-2).mean(-3)                    
                    else:
                        data0 = data0.reshape(len(data0), int(old_div(dims[1], ds[0])), ds[0], int(old_div(dims[2], ds[1])),
                                              ds[1]).mean(-1).mean(-2)
                    data0.shape = (len(data0), -1)
                    
                    activity = ones((L + adaptBias, len(data0))) * activity.mean(1).reshape(-1, 1)
                    lam1_s=lam1_s*mbs[it+1]/mbs[it]
                    activity = HALS4activity(data0, S, activity,NonNegative,lam1_t,lam2_t,dims0,[],30)
                    S, activity, mask,centers,boxes,ES,L=RenormalizeDeleteSort(S, activity, mask,centers,boxes,ES,adaptBias,MedianFilt)
                    lam1_s=ES.lam
    
        ### Stop adding components ###
            if L==0: #if no non-background components found, return empty arrays
                print('No non-background components found, aborting...')
                return [], [], []
            
            if FineTune: ### Upscale Back to full data ##
                activity = ones((L + adaptBias, dims[0])) * activity.mean(1).reshape(-1, 1)
                data0=data
                dims0=dims
                if D==4:
                    S = repeat(repeat(repeat(S.reshape((-1,) + dims0[1:]), ds[0], 1), ds[1], 2), ds[2], 3)
                    lam1_s= repeat(repeat(repeat(lam1_s.reshape((-1,) + dims0[1:]), ds[0], 1), ds[1], 2), ds[2], 3)
                else:
                    S = repeat(repeat(S.reshape((-1,) + dims0[1:]), ds[0], 1), ds[1], 2)
                    lam1_s= repeat(repeat(lam1_s.reshape((-1,) + dims0[1:]), ds[0], 1), ds[1], 2)
                for dd in range(1,D):
                    while S.shape[dd]<dims[dd]:
                        shape_append=np.array(S.shape)
                        shape_append[dd]=1
                        S=np.append(S,values=np.take(S,-1,axis=dd).reshape(shape_append),axis=dd)
                        lam1_s=np.append(lam1_s,values=np.take(lam1_s,-1,axis=dd).reshape(shape_append),axis=dd)
                S=S.reshape(L + adaptBias, -1)
                lam1_s=lam1_s.reshape(L+ adaptBias,-1)
                for ll in range(L):
                    boxes[ll] = GetBox(centers[ll], R, dims[1:])
                    temp = zeros(dims[1:])
                    temp[[slice(*a) for a in boxes[ll]]] = 1
                    mask[ll] = np.where(temp.ravel())[0]                
                
                ES=ExponentialSearch(lam1_s)
                activity = HALS4activity(data0, S, activity,NonNegative,lam1_t,lam2_t,dims0,[], 30)
                S, activity, mask,centers,boxes,ES,L=RenormalizeDeleteSort(S, activity, mask,centers,boxes,ES,adaptBias,MedianFilt)
                lam1_s=ES.lam
                
                sn_target=np.zeros(prod(dims0[1:]))
                sn_std=sn_target
                MSE_target = np.mean(sn_target**2)
                MSE_std=np.mean(sn_std**2)
        #        MSE = np.mean((data0-np.dot(activity.T,S))**2)
            
    #### Main Loop ####
      
        print('starting main NMF loop')
        for kk in range(iters):
            lam1_s=ES.lam #update sparsity parameters
            S = HALS4shape(data0, S, activity,mask,lam1_s,lam2_s,adaptBias,inner_iterations)
            #apply additional constraints/processing 
            if Connected==True:            
                S=LargestConnectedComponent(S,dims0,adaptBias)
            if WaterShed==True:
                S=LargestWatershedRegion(S,dims0,adaptBias)
            if kk==iters-1:
                if FinalNonNegative==False:
                    NonNegative=False
            activity = HALS4activity(data0, S, activity,NonNegative,lam1_t,lam2_t,dims0,[],inner_iterations)
     
            if SigmaMask!=[]:
                mask=GrowMasks(S,mask,boxes,dims0,adaptBias,SigmaMask)
            S, activity, mask,centers,boxes,ES,L=RenormalizeDeleteSort(S, activity, mask,centers,boxes,ES,adaptBias,MedianFilt)
            
            # Measure MSE and update sparsity parameters
            print('main iteration kk=',kk,'L=',L)
            if (kk+1)%updateLambdaIntervals==0:            
                sn=np.sqrt(old_div((Energy-2*np.sum(np.dot(activity,data0)*S,axis=0)+np.sum(np.dot(np.dot(activity,activity.T),S)*S,axis=0)),dims0[0]))
                delta_sn=sn-sn_target
                MSE = np.mean(sn**2)
                
                signcheck=(datasum-np.dot(np.sum(activity.T,axis=0),S))<0
                
                if S==[]:
                    spars=0
                else:
                    spars=np.mean(S>0,axis=1)
                    
                temp=repeat(delta_sn.reshape(1,-1),L+adaptBias,axis=0) 
    
                if TargetAreaRatio==[]:  
                    cond_decrease=temp>sn_std
                    cond_increase=temp<-sn_std
                else:
                    if adaptBias:
                        spars[-1]=old_div((TargetAreaRatio[1]+TargetAreaRatio[0]),2) # ignore sparsity target for background (bias) component  
                    temp2=repeat(spars.reshape(-1,1),len(S[0]),axis=1)
                    cond_increase=np.logical_or(temp2>TargetAreaRatio[1],temp<-sn_std)
                    cond_decrease=np.logical_and(temp2<TargetAreaRatio[0],temp>sn_std)
                
                
                ES.update(cond_decrease,cond_increase)
                lam1_s=ES.lam
                if kk<old_div(iters,3): #restart exponential search unless enough iterations have passed
                    ES=ExponentialSearch(lam1_s)                
                else:
                    if not(np.any(cond_increase) or np.any(cond_decrease)):
                        print('sparsity target reached')
                        break
                    if L+adaptBias>1: # if we have more then one component just keep exponitiated grad descent instead
                        if (kk+1)%updateRhoIntervals==0: #update rho every updateRhoIntervals if we are still not converged
                            if np.any(spars[:L]<TargetAreaRatio[0]) or np.any(spars[:L]>TargetAreaRatio[1]):
                                ES.rho=2-old_div(1,(ES.rho))
                                print('rho=',ES.rho)
                        ES=ExponentialSearch(lam1_s,rho=ES.rho)
                
                # prinst MSE and other information
                if verbose:             
                    print(' MSE = {0:.6f}, Target MSE={1:.6f},Sparsity={2:.4f},lam1_s={3:.6f}'.format(MSE,MSE_target,np.mean(spars[:L]),np.mean(lam1_s)))
                    if kk == (iters - 1):
                        print('Maximum iteration limit reached')
                    MSE_array.append(MSE)
        
        # Some post-processing 
        S=S.reshape((-1,) + dims[1:])
        S,activity,L=PruneComponents(S,activity,L) #prune "bad" components
        if len(S)>1:
            S,activity,L=MergeComponents(S,activity,L,threshold_activity=MergeThreshold_activity,threshold_shape=MergeThreshold_shapes,sig=10)    #merge very similar components
            if not FineTune:
                activity = ones((L + adaptBias, dims[0])) * activity.mean(1).reshape(-1, 1) #extract activity from full data
            activity=HALS4activity(data, S.reshape((len(S),-1)), activity,NonNegative,lam1_t,lam2_t,dims0,[],iters=30)
        
        return asarray(MSE_array), S, activity
