from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

from builtins import dict
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from builtins import object

import os #change path to where the python scripty is located
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)


class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def GetDefaultParams():
    # Get parameters for: Data type, NMF algorithm and intialization
    
    # choose dataset name (function GetData will use this to fetch the correct dataset)
    data_name_set=['Hillman','HillmanSmall','Sophie2D','Sophie3D','SophieVoltage3D','Sophie3DSmall',
    'SaraSmall','Sara19DEC2015_w1t1','PhilConfocal','PhilMFM','PhilConfocal2','BaylorAxonsSmall',
    'BaylorAxons','BaylorAxonsQuiet','BaylorAxonsActive','BaylorAxonsJiakun1','BaylorAxonsJiakun2','Ja_Ni_ds3','YairDendrites']
    data_name=data_name_set[-6]
    
    # "default" parameters - for additional information see "LocalNMF" function in BlockLocalNMF
    
    NumCent=0 # Max number of centers to import from Group Lasso intialization - if 0, we don't run group lasso
    mbs=[1] # temporal downsampling of data in intial phase of NMF
    ds=1 # spatial downsampling of data in intial phase of NMF. Ccan be an integer or a list of the size of spatial dimensions
    TargetAreaRatio=[0.005,0.05] # target sparsity range for spatial components - low
    repeats=1 # how many repeations to run NMF algorithm
    iters0=[10] # number of intial NMF iterations, in which we downsample data and add components
    iters=50 # number of main NMF iterations, in which we fine tune the components on the full data
    lam1_s=0.1 # l1 regularization parameter initialization (for increased sparsity). If zero, we have no l1 sparsity penalty
    updateLambdaIntervals=2 # update sparsity parameter every updateLambdaIntervals iterations
    addComponentsIntervals=1 # in initial NMF phase, add new component every updateLambdaIntervals*addComponentsIntervals iterations
    updateRhoIntervals=1 # in main NMF phase, update sparsity learning speed (Rho) every updateLambdaIntervals*updateRhoIntervals iterations
    Background_num=1 #number of background components - one of which at every repetion
    bkg_per=0.2 # intialize of background shape at this percentile (over time) of video, in the range [0,100]
    sig=(500,500,500) # estiamte size of neuron - bounding box is 3 times this size. If larger then data, we have no bounding box.
    SigmaMask=[]    # if not [], then update masks so that they are non-zero a radius of SigmaMasks around previous non-zero support of shapes
    MergeThreshold_activity=1#merge components if activity is correlated above the this threshold (and sufficiently close)
    MergeThreshold_shapes=1 #merge components if activity is correlated above the this threshold (and sufficiently close)
    
    NonNegative=True # should we constrain activity and shapes to be non-negative?
    FinalNonNegative=True # should we constrain activity to be non-negative at final iteration?
    Connected=False # should we constrain all spatial component to be connected?
    WaterShed=False # should we constrain all spatial component to have only one watershed component?
    MedianFilt=False 
    FineTune=True # Should use the full data at the main iterations (to fine tune shapes)? If not then just extract the activity and not the shapes from the full data
    Deconvolve=False #Deconvolve activity to get smoothed (denoised) calcium trace
    
    # obsolete experimental stuff - don't use these for now
    ThresholdData= False  # threshold data with PSD level
    estimateNoise=False # should we tune sparsity and number of neurons to reach estimated noise level?
    PositiveError=False # should we tune sparsity and number of neurons to have only positive residual error?
    FixSupport=False # should we fix non-zero support at main NMF iterations?
    SmoothBackground=False # Should we cut out out peaks from background component?
    SigmaBlur=[] # Spatial de-bluring with Gaussian Kernel of this width. 
    
    SuperVoxelize=False # should we supervoxelize data (does not work now)

    # change parameters for other datasets    
    if data_name=='Hillman' or data_name=='HillmanSmall':
        NumCent=0 # Max number of centers to import from Group Lasso intialization - if 0, we don't run group lasso
        mbs=[1] # temporal downsampling of data in intial phase of NMF
        ds=1 # spatial downsampling of data in intial phase of NMF. Ccan be an integer or a list of the size of spatial dimensions
        TargetAreaRatio=[0.005,0.02] # target sparsity range for spatial components
        repeats=2 # how many repeations to run NMF algorithm
        iters0=[10] # number of intial NMF iterations, in which we downsample data and add components
        iters=10 # number of main NMF iterations, in which we fine tune the components on the full data
        lam1_s=0.1 # l1 regularization parameter initialization (for increased sparsity). If zero, we have no l1 sparsity penalty
        updateLambdaIntervals=2 # update sparsity parameter every updateLambdaIntervals iterations
        addComponentsIntervals=1 # in initial NMF phase, add new component every updateLambdaIntervals*addComponentsIntervals iterations
        updateRhoIntervals=1 # in main NMF phase, add new component every updateLambdaIntervals*updateRhoIntervals iterations
        Background_num=2 #number of background components - one of which at every repetion
        bkg_per=0.2 # intialize of background shape at this percentile (over time) of video
        sig=(500,500,500) # estiamte size of neuron - bounding box is 3 times this size. If larger then data, we have no bounding box.
        
        NonNegative=True # should we constrain activity and shapes to be non-negative?
        FinalNonNegative=True # should we constrain activity to be non-negative at final iteration?
        Connected=False # should we constrain all spatial component to be connected?
        WaterShed=False # should we constrain all spatial component to have only one watershed component?    
    elif data_name=='Sophie2D':
        mbs=[2]
        ds=2
        TargetAreaRatio=[0.03,0.15]
        iters=30
        iters0=[30]
        repeats=10
        updateLambdaIntervals=2
        updateRhoIntervals=1
        lam1_s=0.01
        Background_num=0 #number of background components

        Connected=True
        sig=(500,500)
    elif data_name=='Sophie3D':
        mbs=[1]
        ds=1
        TargetAreaRatio=[0.002,0.02]
        iters=50
        iters0=[10]
        repeats=100
        Background_num=5 #number of background components
    elif data_name=='SophieVoltage3D':
        mbs=[1]
        ds=1
        TargetAreaRatio=[0.002,0.02]
        iters=50
        iters0=[10]
        repeats=100
        Background_num=0 #number of background components
        FinalNonNegative=False
    elif data_name=='Sara19DEC2015_w1t1' or data_name=='SaraSmall':
        mbs=[1]
        ds=1
        TargetAreaRatio=[0.005,0.02]
        repeats=1
        iters0=[50]
        iters=200
        updateLambdaIntervals=2
        updateRhoIntervals=1
        lam1_s=0.1
        Background_num=1 #number of background components
        bkg_per=0.01
        Connected=True
        WaterShed=True
        FinalNonNegative=False
        sig=(10,10,1)
    elif data_name=='PhilConfocal' or data_name=='PhilConfocal2':
        NumCent=50
        mbs=[1]
        ds=[2,2,1]
        TargetAreaRatio=[0.005,0.05]
        repeats=1
        iters0=[50]
        iters=20
        updateLambdaIntervals=2
        updateRhoIntervals=1
        lam1_s=1
        Background_num=1 #number of background components
        bkg_per=0.2

        Connected=True
        WaterShed=True
        FinalNonNegative=False
        sig=(20,20,2)
    elif data_name=='PhilMFM':
        NumCent=0
        mbs=[2]
        ds=[2,2,1]
        TargetAreaRatio=[0.005,0.03]
        repeats=1
        iters0=[60]
        iters=20        
        updateLambdaIntervals=2
        addComponentsIntervals=1
        updateRhoIntervals=1
        lam1_s=1
        Background_num=1 #number of background components
        bkg_per=0.02

        Connected=True
        WaterShed=True

        FinalNonNegative=False
        sig=(500,500,3)
    elif data_name=='BaylorAxonsSmall':
        NumCent=300 # Max number of centers to import from Group Lasso intialization - if 0, we don't run group lasso
        mbs=[1] # temporal downsampling of data in intial phase of NMF
        ds=1 # spatial downsampling of data in intial phase of NMF. Ccan be an integer or a list of the size of spatial dimensions
        TargetAreaRatio=[0.01,0.03] # target sparsity range for spatial components
        repeats=1 # how many repeations to run NMF algorithm
        iters0=[20] # number of intial NMF iterations, in which we downsample data and add components
        iters=100 # number of main NMF iterations, in which we fine tune the components on the full data
        lam1_s=0.001# l1 regularization parameter initialization (for increased sparsity). If zero, we have no l1 sparsity penalty
        updateLambdaIntervals=2 # update sparsity parameter every updateLambdaIntervals iterations
        addComponentsIntervals=1 # in initial NMF phase, add new component every updateLambdaIntervals*addComponentsIntervals iterations
        updateRhoIntervals=1 # in main NMF phase, update sparsity learning speed (Rho) every updateLambdaIntervals*updateRhoIntervals iterations
        Background_num=1 #number of background components - one of which at every repetion
        bkg_per=0.1 # intialize of background shape at this percentile (over time) of video
        sig=(200,200) # estiamte size of neuron - bounding box is 3 times this size. If larger then data, we have no bounding box.
        SigmaBlur=[]
        
        NonNegative=True # should we constrain activity and shapes to be non-negative?
        FinalNonNegative=True # should we constrain activity to be non-negative at final iteration?
        Connected=True # should we constrain all spatial component to be connected?
        WaterShed=False # should we constrain all spatial component to have only one watershed component?
        SigmaMask=3       
        
    elif data_name=='BaylorAxons':
        NumCent=500 # Max number of centers to import from Group Lasso intialization - if 0, we don't run group lasso
        mbs=[10] # temporal downsampling of data in intial phase of NMF
        ds=1 # spatial downsampling of data in intial phase of NMF. Ccan be an integer or a list of the size of spatial dimensions
        TargetAreaRatio=[0.01,0.05] # target sparsity range for spatial components
        repeats=1 # how many repeations to run NMF algorithm
        iters0=[50] # number of intial NMF iterations, in which we downsample data and add components
        iters=100 # number of main NMF iterations, in which we fine tune the components on the full data
        lam1_s=1e-4# l1 regularization parameter initialization (for increased sparsity). If zero, we have no l1 sparsity penalty
        updateLambdaIntervals=2 # update sparsity parameter every updateLambdaIntervals iterations
        addComponentsIntervals=1 # in initial NMF phase, add new component every updateLambdaIntervals*addComponentsIntervals iterations
        updateRhoIntervals=1 # in main NMF phase, update sparsity learning speed (Rho) every updateLambdaIntervals*updateRhoIntervals iterations
        Background_num=1 #number of background components - one of which at every repetion
        bkg_per=0.1 # intialize of background shape at this percentile (over time) of video
        sig=(200,200) # estiamte size of neuron - bounding box is 3 times this size. If larger then data, we have no bounding box.
        
        FineTune=False
        NonNegative=True # should we constrain activity and shapes to be non-negative?
        FinalNonNegative=True # should we constrain activity to be non-negative at final iteration?
        Connected=True # should we constrain all spatial component to be connected?
        WaterShed=False # should we constrain all spatial component to have only one watershed component?        
        SigmaMask=3  
    elif data_name=='BaylorAxonsQuiet':
        NumCent=400 # Max number of centers to import from Group Lasso intialization - if 0, we don't run group lasso
        mbs=[10] # temporal downsampling of data in intial phase of NMF
        ds=1 # spatial downsampling of data in intial phase of NMF. Ccan be an integer or a list of the size of spatial dimensions
        TargetAreaRatio=[0.01,0.06] # target sparsity range for spatial components
        repeats=1 # how many repeations to run NMF algorithm
        iters0=[30] # number of intial NMF iterations, in which we downsample data and add components
        iters=100 # number of main NMF iterations, in which we fine tune the components on the full data
        lam1_s=10# l1 regularization parameter initialization (for increased sparsity). If zero, we have no l1 sparsity penalty
        updateLambdaIntervals=2 # update sparsity parameter every updateLambdaIntervals iterations
        addComponentsIntervals=1 # in initial NMF phase, add new component every updateLambdaIntervals*addComponentsIntervals iterations
        updateRhoIntervals=1 # in main NMF phase, update sparsity learning speed (Rho) every updateLambdaIntervals*updateRhoIntervals iterations
        Background_num=1 #number of background components - one of which at every repetion
        bkg_per=0.1 # intialize of background shape at this percentile (over time) of video
        sig=(200,200) # estiamte size of neuron - bounding box is 3 times this size. If larger then data, we have no bounding box.
        MergeThreshold_activity=0.7#merge components if activity is correlated above the this threshold (and sufficiently close)
        MergeThreshold_shapes=0.7 #merge components if activity is correlated above the this threshold (and sufficiently close)        
        
        ThresholdData= False  # threshold data with PSD level
        FineTune=False
        Deconvolve=False #Deconvolve activity to get smoothed (denoised) calcium trace
        NonNegative=True # should we constrain activity and shapes to be non-negative?
        FinalNonNegative=True # should we constrain activity to be non-negative at final iteration?
        Connected=True # should we constrain all spatial component to be connected?
        WaterShed=False # should we constrain all spatial component to have only one watershed component?        
        SigmaMask=3   
        
    elif data_name=='BaylorAxonsActive':
        NumCent=400 # Max number of centers to import from Group Lasso intialization - if 0, we don't run group lasso
        mbs=[10] # temporal downsampling of data in intial phase of NMF
        ds=1 # spatial downsampling of data in intial phase of NMF. Ccan be an integer or a list of the size of spatial dimensions
        TargetAreaRatio=[0.01,0.06] # target sparsity range for spatial components
        repeats=1 # how many repeations to run NMF algorithm
        iters0=[30] # number of intial NMF iterations, in which we downsample data and add components
        iters=100 # number of main NMF iterations, in which we fine tune the components on the full data
        lam1_s=10# l1 regularization parameter initialization (for increased sparsity). If zero, we have no l1 sparsity penalty
        updateLambdaIntervals=2 # update sparsity parameter every updateLambdaIntervals iterations
        addComponentsIntervals=1 # in initial NMF phase, add new component every updateLambdaIntervals*addComponentsIntervals iterations
        updateRhoIntervals=1 # in main NMF phase, update sparsity learning speed (Rho) every updateLambdaIntervals*updateRhoIntervals iterations
        Background_num=1 #number of background components - one of which at every repetion
        bkg_per=0.1 # intialize of background shape at this percentile (over time) of video
        sig=(200,200) # estiamte size of neuron - bounding box is 3 times this size. If larger then data, we have no bounding box.
        
        ThresholdData= True  # threshold data with PSD level
        FineTune=False
        NonNegative=True # should we constrain activity and shapes to be non-negative?
        FinalNonNegative=True # should we constrain activity to be non-negative at final iteration?
        Connected=True # should we constrain all spatial component to be connected?
        WaterShed=False # should we constrain all spatial component to have only one watershed component?        
        SigmaMask=3  
        
    elif data_name=='BaylorAxonsJiakun1' :
        NumCent=400 # Max number of centers to import from Group Lasso intialization - if 0, we don't run group lasso
        mbs=[10] # temporal downsampling of data in intial phase of NMF
        ds=1 # spatial downsampling of data in intial phase of NMF. Ccan be an integer or a list of the size of spatial dimensions
        TargetAreaRatio=[0.01,0.06] # target sparsity range for spatial components
        repeats=2 # how many repeations to run NMF algorithm
        iters0=[30] # number of intial NMF iterations, in which we downsample data and add components
        iters=100 # number of main NMF iterations, in which we fine tune the components on the full data
        lam1_s=10# l1 regularization parameter initialization (for increased sparsity). If zero, we have no l1 sparsity penalty
        updateLambdaIntervals=2 # update sparsity parameter every updateLambdaIntervals iterations
        addComponentsIntervals=1 # in initial NMF phase, add new component every updateLambdaIntervals*addComponentsIntervals iterations
        updateRhoIntervals=1 # in main NMF phase, update sparsity learning speed (Rho) every updateLambdaIntervals*updateRhoIntervals iterations
        Background_num=1 #number of background components - one of which at every repetion
        bkg_per=0.1 # intialize of background shape at this percentile (over time) of video
        sig=(200,200) # estiamte size of neuron - bounding box is 3 times this size. If larger then data, we have no bounding box.
        
        ThresholdData= True  # threshold data with PSD level
        FineTune=False
        NonNegative=True # should we constrain activity and shapes to be non-negative?
        FinalNonNegative=True # should we constrain activity to be non-negative at final iteration?
        Connected=True # should we constrain all spatial component to be connected?
        WaterShed=False # should we constrain all spatial component to have only one watershed component?        
        SigmaMask=3  
        
    elif data_name=='Ja_Ni_ds3' :
        NumCent=30 # Max number of centers to import from Group Lasso intialization - if 0, we don't run group lasso
        mbs=[10] # temporal downsampling of data in intial phase of NMF
        ds=1 # spatial downsampling of data in intial phase of NMF. Ccan be an integer or a list of the size of spatial dimensions
        TargetAreaRatio=[0.01,0.35] # target sparsity range for spatial components
        repeats=1 # how many repeations to run NMF algorithm
        iters0=[1] # number of intial NMF iterations, in which we downsample data and add components
        iters=100 # number of main NMF iterations, in which we fine tune the components on the full data
        lam1_s=10# l1 regularization parameter initialization (for increased sparsity). If zero, we have no l1 sparsity penalty
        updateLambdaIntervals=2 # update sparsity parameter every updateLambdaIntervals iterations
        addComponentsIntervals=1 # in initial NMF phase, add new component every updateLambdaIntervals*addComponentsIntervals iterations
        updateRhoIntervals=1 # in main NMF phase, update sparsity learning speed (Rho) every updateLambdaIntervals*updateRhoIntervals iterations
        Background_num=1 #number of background components - one of which at every repetion
        bkg_per=0.05 # intialize of background shape at this percentile (over time) of video
        sig=(5,5) # estiamte size of neuron - bounding box is 3 times this size. If larger then data, we have no bounding box.
        
        FineTune=False
        NonNegative=True # should we constrain activity and shapes to be non-negative?
        FinalNonNegative=True # should we constrain activity to be non-negative at final iteration?
        Connected=True # should we constrain all spatial component to be connected?
        WaterShed=False # should we constrain all spatial component to have only one watershed component?        
        SigmaMask=3 
    elif data_name=='YairDendrites' :
        NumCent=100 # Max number of centers to import from Group Lasso intialization - if 0, we don't run group lasso
        mbs=[1] # temporal downsampling of data in intial phase of NMF
        ds=1 # spatial downsampling of data in intial phase of NMF. Ccan be an integer or a list of the size of spatial dimensions
        TargetAreaRatio=[0.001,0.03] # target sparsity range for spatial components
        repeats=1 # how many repeations to run NMF algorithm
        iters0=[1] # number of intial NMF iterations, in which we downsample data and add components
        iters=100 # number of main NMF iterations, in which we fine tune the components on the full data
        lam1_s=10# l1 regularization parameter initialization (for increased sparsity). If zero, we have no l1 sparsity penalty
        updateLambdaIntervals=2 # update sparsity parameter every updateLambdaIntervals iterations
        addComponentsIntervals=1 # in initial NMF phase, add new component every updateLambdaIntervals*addComponentsIntervals iterations
        updateRhoIntervals=1 # in main NMF phase, update sparsity learning speed (Rho) every updateLambdaIntervals*updateRhoIntervals iterations
        Background_num=1 #number of background components - one of which at every repetion
        bkg_per=0.05 # intialize of background shape at this percentile (over time) of video
        sig=(5,5) # estiamte size of neuron - bounding box is 3 times this size. If larger then data, we have no bounding box.
        
        FineTune=False
        NonNegative=True # should we constrain activity and shapes to be non-negative?
        FinalNonNegative=True # should we constrain activity to be non-negative at final iteration?
        Connected=True # should we constrain all spatial component to be connected?
        WaterShed=False # should we constrain all spatial component to have only one watershed component?        
        SigmaMask=3 
        
        
        

    params_dict=dict([['data_name',data_name],['SuperVoxelize',SuperVoxelize],['NonNegative',NonNegative],
                      ['FinalNonNegative',FinalNonNegative],['mbs',mbs],['TargetAreaRatio',TargetAreaRatio],['Deconvolve',Deconvolve],
                     ['iters',iters],['iters0',iters0],['lam1_s',lam1_s],['MedianFilt',MedianFilt],['SigmaMask',SigmaMask],['FineTune',FineTune],
                     ['updateLambdaIntervals',updateLambdaIntervals],['updateRhoIntervals',updateRhoIntervals],['addComponentsIntervals',addComponentsIntervals],
                     ['estimateNoise',estimateNoise],['PositiveError',PositiveError],['sig',sig],['NumCent',NumCent],['SigmaBlur',SigmaBlur],
                    ['bkg_per',bkg_per],['ds',ds],['sig',sig],['Background_num',Background_num],['Connected',Connected],['WaterShed',WaterShed],
                    ['SmoothBackground',SmoothBackground],['FixSupport',FixSupport],['repeats',repeats],['ThresholdData',ThresholdData],
                    ['MergeThreshold_shapes',MergeThreshold_shapes],['MergeThreshold_activity',MergeThreshold_activity]    ])

    params=Bunch(params_dict)
    
    return params,params_dict

#%% Main script for running NMF

if __name__ == "__main__":
        
    import matplotlib
    matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
        
    import numpy as np
    from BlockLocalNMF import LocalNMF
    import matplotlib.pyplot as plt
    from AuxilaryFunctions import GetFileName,SuperVoxelize,GetData,GetCentersData,ThresholdData,make_sure_path_exists
    import pickle 
    
    plt.close('all')
    
    #Make sure relevant folders exist 
    NMF_Results_Folder='NMF_Results'
    make_sure_path_exists(NMF_Results_Folder)
    
    plot_all=True #Plot final results (see end of this file)
    do_NMF=True # Do CNMF on data
        
    params,params_dict=GetDefaultParams() # get default parameters for dataset
    
    data=GetData(params.data_name) #get data 
    
    if params.ThresholdData:  #obsolete
        data=ThresholdData(data)
        
    if params.SuperVoxelize==True: #obsolete 
        data=SuperVoxelize(data)  
            
    if do_NMF==True:         
        for rep in range(params.repeats):  #perform several iterations of NMF over data, each time extracting more components
            
            if params.NumCent>0: #extract intialization centers using group lasso, if needed
                cent=GetCentersData(data,params.NumCent)
#                cent=GetCentersData(data,params.NumCent,params.data_name,rep) #obsolete version
            else: #no intialization case - neuron are added one by one during initial downsampled iterations
                cent=np.reshape([],(0,data.ndim-1)) 
                
            if rep>=params.Background_num: # no background component case
                adaptBias=False
            else: # have a background component in this repetition
                adaptBias=True
            #main CNMF function
            MSE_array, shapes, activity = LocalNMF(
                data, cent, params.sig,TargetAreaRatio=params.TargetAreaRatio,updateLambdaIntervals=params.updateLambdaIntervals,addComponentsIntervals=params.addComponentsIntervals,
                WaterShed=params.WaterShed,SigmaMask=params.SigmaMask,PositiveError=params.PositiveError,NonNegative=params.NonNegative,Deconvolve=params.Deconvolve,
                FinalNonNegative=params.FinalNonNegative,MedianFilt=params.MedianFilt,verbose=True,lam1_s=params.lam1_s,MergeThreshold_activity=params.MergeThreshold_activity,
                adaptBias=adaptBias,estimateNoise=params.estimateNoise,FineTune=params.FineTune,SigmaBlur=params.SigmaBlur,MergeThreshold_shapes=params.MergeThreshold_shapes,
                Connected=params.Connected,SmoothBkg=params.SmoothBackground,FixSupport=params.FixSupport,bkg_per=params.bkg_per,iters0=params.iters0,iters=params.iters,mbs=params.mbs, ds=params.ds)
            
            # save results to file
            L=len(shapes)
            print(str(L)+' components extracted')
            if L<=adaptBias:
                break
            saveName=GetFileName(params_dict,rep)  
            from io import open
            f = open('NMF_Results/'+saveName, 'wb')
            
            results=dict([['MSE_array',MSE_array], ['shapes',shapes],['activity',activity],['cent',cent],['params',params_dict]])
            pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.close()

            #subtract this iteration components from data       
            data=data- activity.T.dot(shapes.reshape(len(shapes), -1)).reshape(np.shape(data))  
        
    #%% PLotting
    if plot_all==True:
        from PlotResults import PlotAll
        SaveNames=[] 
        for rep in range(params.repeats):
            SaveNames+=[GetFileName(params_dict,rep)]
        PlotAll(SaveNames,params)
        
    #    L=len(activity)   
    #    for ll in range(L):
    #        print 'Sparsity=',np.mean(shapes[ll]>0)
    
