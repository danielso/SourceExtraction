from numpy import asarray, percentile, zeros, ones, ix_, arange, exp, prod, repeat
import numpy as np
from BlockLocalNMF_AuxilaryFunctions import  HALS4activity, HALS4shape,RenormalizeDeleteSort,addComponent,GetBox, \
RegionAdd,RegionCut,DownScale,LargestConnectedComponent,LargestWatershedRegion,SmoothBackground,GetSnPSDArray,ExponentialSearch,GrowMasks,GetSnPSD,FISTA4shape
from AuxilaryFunctions import PruneComponents,MergeComponents      
import sys
sys.path.append('OASIS/')
from functions import deconvolve  

def LocalNMF(data, centers, sig, NonNegative=True,FinalNonNegative=True,verbose=False,adaptBias=True,TargetAreaRatio=[],estimateNoise=False,
             PositiveError=False,MedianFilt=False,Connected=False,FixSupport=False, WaterShed=False,SmoothBkg=False,FineTune=True,Deconvolve=False,
             SigmaMask=[],updateLambdaIntervals=2,updateRhoIntervals=2,addComponentsIntervals=1,bkg_per=20,SigmaBlur=[],
             iters=10,iters0=[30], mbs=[1], ds=1,lam1_s=0,lam1_t=0,lam2_s=0,lam2_t=0):
    """
    Parameters
    ----------
    data : array, shape (T, X, Y[, Z])
        block of the data
    centers : array, shape (L, D)
        L centers of suspected neurons where D is spatial dimension (2 or 3)
    sig : array, shape (D,)
        size of the gaussian kernel in different spatial directions
    NonNegative : boolean
        if True, neurons activity should be considered as non-negative
    FinalNonNegative : boolean
        if False, last activity iteration is done without non-negativity constraint, even if NonNegative==True       
    verbose : boolean
        print progress and record MSE if true (about 2x slower)
    adaptBias : boolean
        subtract rank 1 estimate of bias (background)
    TargetAreaRatio : list of length 2
        Lower and upper bounds on sparsity of non-background components
    estimateNoise : boolean
        estimate noise variance and use it determine if to add components, and to modify sparsity by affecting lam1_s (does not work very well)
    PositiveError : boolean
        do not allow pixels in which the residual (summed over time) becomes negative, by increasing lam1_s in these pixels
    MedianFilt : boolean
        do median filter of spatial components 
    Connected: boolean
        impose connectedness of spatial component by keeping only the largest non-zero connected component in each iteration of HALS
    WaterShed: boolean
        impose that each spatial component has a single watershed region
    SmoothBkg: boolean
        Remove local peaks from background component
    FixSupport : boolean
        do not allow spatial components to be non-zero where sub-sampled spatial components are zero
    FineTune :  boolean
        fine tune main iterations on full data, if not, use (last) downsampled data
    Deconvolve : boolean
        Deconvolve activity to get smoothed (denoised) calcium trace. This is done only on the main itreations, and if FineTune=True
    SigmaMask : scalar or empty
        if not [], then update masks so that they are SigmaMasks around non-zero support of shapes
    SigmaBlur : scalar
        if not [], then de-blur spatial components using Gaussian Kernel of this width
    updateLambdaIntervals : int
        update lam1_s every this number of HALS iterations, to match contraints
    updateRhoIntervals : int
        decrease rho, update rate of lam1_s, every this number of updateLambdaIntervals HALS iterations (only active during main iterations)
    addComponentsIntervals : int
        add new component, if possible, every this number of updateLambdaIntervals HALS iterations (only active during sub-sampled iterations)
    bkg_per : float
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
    lam2_s : float
        L_2 regularization constant for sparsity of shapes
    lam_t : float
        L_1 regularization constant for sparsity of activity
    lam2_t : float
        L_2 regularization constant for sparsity of activity

    Returns
    -------
    MSE_array : list (empty if verbose is False)
        Mean square error during algorithm operation
    shapes : array, shape (L+adaptBias, X, Y (,Z))
        the neuronal shape vectors (empty if no components found)
    activity : array, shape (L+adaptBias, T)
        the neuronal activity for each shape (empty if no components found)
    boxes : array, shape (L, D, 2)
        edges of the boxes in which each neuronal shapes lie (empty if no components found)
    """
    
    # Catch Errors
    if ds!=1 and SigmaBlur!=[]:
        raise NameError('case ds!=1 and SigmaBlur!=[] no yet written in NMF code')
        
    
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
    activity = zeros((L, dims[0] / mb)) #array of temporal components
    lam1_s0=np.copy(lam1_s) #intial spatial sparsity (l1) parameters
    if TargetAreaRatio!=[]:
        if TargetAreaRatio[0]>TargetAreaRatio[1]:            
            print 'WARNING -  TargetAreaRatio[0]>TargetAreaRatio[1] !!!'
    if iters0[0] == 0:
        ds = 1

        
### Initialize shapes, activity, and residual ###        
    
    data0,dims0=DownScale(data,mb,ds) #downscaled data and dimensions
    if type(ds)==int:
        ds=ds*np.ones(D-1)

    if D == 4: #downscale activity
        activity = data0[:, map(int, centers[:, 0] / ds[0]), map(int, centers[:, 1] / ds[1]),
                         map(int, centers[:, 2] / ds[2])].T
    else:
        activity = data0[:, map(int, centers[:, 0] / ds[0]), map(int, centers[:, 1] / ds[1])].T
        
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
        boxes[ll] = GetBox(centers[ll] / ds, R / ds, dims0[1:])
        temp = zeros(dims0[1:])
        temp[map(lambda a: slice(*a), boxes[ll])]=1
        mask += np.where(temp.ravel())
        temp = [(arange(int(dims[i + 1] / ds[i])) -int( centers[ll][i] / ds[i])) ** 2 / (2 * (sig[i] / ds[i]) ** 2)
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
            if estimateNoise:
                sn_target,sn_std= GetSnPSDArray(data0)#target noise level
            else:
                sn_target=np.zeros(prod(dims0[1:]))
                sn_std=sn_target
            MSE_target = np.mean(sn_target**2)
            ES=ExponentialSearch(lam1_s) #object to update sparsity parameters
            lam1_s=ES.lam
            for kk in range(iters0[it]):
                # update sparisty parameters     
                if kk%updateLambdaIntervals==0:                 
                    sn=np.sqrt(Energy0-2*np.sum(np.dot(activity,data0)*S,axis=0)+np.sum(np.dot(np.dot(activity,activity.T),S)*S,axis=0))/dims0[0] # efficient way to calcuate MSE per pixel
        
                    delta_sn=sn-sn_target # noise margin
                    signcheck=(data0sum-np.dot(np.sum(activity.T,axis=0),S))<0
                    if PositiveError: #obsolete
                        delta_sn[signcheck]=-float("inf") # residual should not have negative pixels, so we increase lambda for these pixels
                    
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
                                spars[-1]=(TargetAreaRatio[1]+TargetAreaRatio[0])/2 # ignore sparsity target for background (bias) component  
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
                            S, activity, mask,centers,boxes,L=addComponent(new_cent,data0,dims0,R/ds,S, activity, mask,centers,boxes,adaptBias)
                            new_lam=lam1_s0*np.ones_like(data0[0,:]).reshape(1,-1)
                            lam1_s=np.insert(lam1_s,0,values=new_lam,axis=0)
                            ES=ExponentialSearch(lam1_s) #we need to restart exponential search each time we add a component
                            
                #apply additional constraints/processing                            
                if SigmaBlur==[]:
                    S = HALS4shape(data0, S, activity,mask,lam1_s,lam2_s,adaptBias,inner_iterations)
                else: #obsolete
                    S=FISTA4shape(data0, S, activity,mask,lam1_s,adaptBias,SigmaBlur,dims0)
                
                if Connected==True:
                    S=LargestConnectedComponent(S,dims0,adaptBias)
                if WaterShed==True:
                    S=LargestWatershedRegion(S,dims0,adaptBias)
                activity = HALS4activity(data0, S, activity,NonNegative,lam1_t,lam2_t,dims0,SigmaBlur,inner_iterations)                                
                if SigmaMask!=[]:
                    mask=GrowMasks(S,mask,boxes,dims0,adaptBias,SigmaMask)
                S, activity, mask,centers,boxes,ES,L=RenormalizeDeleteSort(S, activity, mask,centers,boxes,ES,adaptBias,MedianFilt)
                lam1_s=ES.lam
                if SmoothBkg==True:
                    S=SmoothBackground(S,dims0,adaptBias,tuple(np.array(sig)/np.array(ds)))
                
                print 'Subsampled iteration',kk,'it=',it,'L=',L
            
            # use next (smaller) value for temporal downscaling
            if it < len(iters0) - 1:
                mb = mbs[it + 1]
                data0 = data[:len(data) / mb * mb].reshape(-1, mb, prod(dims[1:])).mean(1)
                if D==4:
                    data0 = data0.reshape(len(data0), int(dims[1] / ds[0]), ds[0], int(dims[2] / ds[1]), ds[1],
                                          int(dims[3] / ds[2]), ds[2]).mean(-1).mean(-2).mean(-3)                    
                else:
                    data0 = data0.reshape(len(data0), int(dims[1] / ds[0]), ds[0], int(dims[2] / ds[1]),
                                          ds[1]).mean(-1).mean(-2)
                data0.shape = (len(data0), -1)
                
                activity = ones((L + adaptBias, len(data0))) * activity.mean(1).reshape(-1, 1)
                lam1_s=lam1_s*mbs[it+1]/mbs[it]
                activity = HALS4activity(data0, S, activity,NonNegative,lam1_t,lam2_t,dims0,SigmaBlur,30)
                S, activity, mask,centers,boxes,ES,L=RenormalizeDeleteSort(S, activity, mask,centers,boxes,ES,adaptBias,MedianFilt)
                lam1_s=ES.lam

    ### Stop adding components ###
        if L==0: #if no non-background components found, return empty arrays
            return [], [], [], []
        
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
                temp[map(lambda a: slice(*a), boxes[ll])] = 1
                mask[ll] = np.where(temp.ravel())[0]
            
            if FixSupport: #obsolete
                for ll in range(L):
                    lam1_s[ll,S[ll]==0]=float("inf")
                
            
            ES=ExponentialSearch(lam1_s)
            activity = HALS4activity(data0, S, activity,NonNegative,lam1_t,lam2_t,dims0,SigmaBlur, 30)
            S, activity, mask,centers,boxes,ES,L=RenormalizeDeleteSort(S, activity, mask,centers,boxes,ES,adaptBias,MedianFilt)
            lam1_s=ES.lam
            
            if estimateNoise:
                sn_target,sn_std= GetSnPSDArray(data0)#target noise level
            else:
                sn_target=np.zeros(prod(dims0[1:]))
                sn_std=sn_target
            MSE_target = np.mean(sn_target**2)
            MSE_std=np.mean(sn_std**2)
    #        MSE = np.mean((data0-np.dot(activity.T,S))**2)
        
#### Main Loop ####
  
    print 'starting main NMF loop'
    for kk in range(iters):
        lam1_s=ES.lam #update sparsity parameters
        if SigmaBlur==[]:
            S = HALS4shape(data0, S, activity,mask,lam1_s,lam2_s,adaptBias,inner_iterations)
        else: #obsolete
            S = FISTA4shape(data0, S, activity,mask,lam1_s,adaptBias,SigmaBlur,dims0)
        #apply additional constraints/processing 
        if Connected==True:            
            S=LargestConnectedComponent(S,dims0,adaptBias)
        if WaterShed==True:
            S=LargestWatershedRegion(S,dims0,adaptBias)
        if kk==iters-1:
            if FinalNonNegative==False:
                NonNegative=False
        activity = HALS4activity(data0, S, activity,NonNegative,lam1_t,lam2_t,dims0,SigmaBlur,inner_iterations)
        if FineTune and Deconvolve:
            for ll in range(L):
                if np.sum(np.abs(activity[ll])>0)>30: #make sure there is enough signal before we try to deconvolve
                    activity[ll], _, _, _, _ = deconvolve(activity[ll], penalty=0)
 
        if SigmaMask!=[]:
            mask=GrowMasks(S,mask,boxes,dims0,adaptBias,SigmaMask)
        S, activity, mask,centers,boxes,ES,L=RenormalizeDeleteSort(S, activity, mask,centers,boxes,ES,adaptBias,MedianFilt)
        
        # Measure MSE and update sparsity parameters
        print 'main iteration kk=',kk,'L=',L
        if (kk+1)%updateLambdaIntervals==0:            
            sn=np.sqrt((Energy-2*np.sum(np.dot(activity,data0)*S,axis=0)+np.sum(np.dot(np.dot(activity,activity.T),S)*S,axis=0))/dims0[0])
            delta_sn=sn-sn_target
            MSE = np.mean(sn**2)
            
            signcheck=(datasum-np.dot(np.sum(activity.T,axis=0),S))<0
            if PositiveError: #obsolete
                delta_sn[signcheck]=-float("inf") # residual should not have negative pixels, so we increase lambda for these pixels
            
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
                    spars[-1]=(TargetAreaRatio[1]+TargetAreaRatio[0])/2 # ignore sparsity target for background (bias) component  
                temp2=repeat(spars.reshape(-1,1),len(S[0]),axis=1)
                cond_increase=np.logical_or(temp2>TargetAreaRatio[1],temp<-sn_std)
                cond_decrease=np.logical_and(temp2<TargetAreaRatio[0],temp>sn_std)
            
            
            ES.update(cond_decrease,cond_increase)
            lam1_s=ES.lam
            if kk<iters/3: #restart exponential search unless enough iterations have passed
                ES=ExponentialSearch(lam1_s)                
            else:
                if not(np.any(cond_increase) or np.any(cond_decrease)):
                    print('sparsity target reached')
                    break
                if L+adaptBias>1: # if we have more then one component just keep exponitiated grad descent instead
                    if (kk+1)%updateRhoIntervals==0: #update rho every updateRhoIntervals if we are still not converged
                        if np.any(spars[:L]<TargetAreaRatio[0]) or np.any(spars[:L]>TargetAreaRatio[1]):
                            ES.rho=2-1/(ES.rho)
                            print 'rho=',ES.rho
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
        S,activity,L=MergeComponents(S,activity,L,threshold=0.9,sig=10)    #merge very similar components
        if not FineTune:
            activity = ones((L + adaptBias, dims[0])) * activity.mean(1).reshape(-1, 1) #extract activity from full data
        activity=HALS4activity(data, S.reshape((len(S),-1)), activity,NonNegative,lam1_t,lam2_t,dims0,SigmaBlur,iters=30)
    
    return asarray(MSE_array), S, activity


# example to check code works


#T = 1000
#X = 201
#Y = 101
#data = np.random.randn(T, X, Y)
#centers = asarray([[40, 30]])
#data[:, 30:45, 25:33] += 2*np.sin(np.array(range(T))/200).reshape(-1,1,1)*np.ones([T,15,8])
#sig = [300, 300]
#
#MSE_array, shapes, activity, boxes = LocalNMF( 
#    data, centers, sig, NonNegative=True, verbose=True,lam1_s=0.1,adaptBias=True)
#
#
#import matplotlib.pyplot as plt
#plt.imshow(shapes[0])
#
#for ll in range(len(shapes)):
#    print np.mean(shapes[ll]>0)