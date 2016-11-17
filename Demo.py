from __future__ import division

def GetDefaultParams():
    # Get parameters for: Data type, NMF algorithm and intialization
    
    # choose dataset name (function GetData will use this to fetch the correct dataset)
    data_name_set=['Hillman','HillmanSmall','Sophie2D','Sophie3D','SophieVoltage3D','Sophie3DSmall',
    'SaraSmall','Sara19DEC2015_w1t1','PhilConfocal','PhilMFM','PhilConfocal2','BaylorAxonsSmall',
    'BaylorAxons','BaylorAxonsQuiet','BaylorAxonsActive','BaylorAxonsJiakun1']
    data_name=data_name_set[-2]
    
    # "default" parameters - for additional information see "LocalNMF" in BlockLocalNMF
    
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
    bkg_per=0.2 # intialize of background shape at this percentile (over time) of video
    sig=(500,500,500) # estiamte size of neuron - bounding box is 3 times this size. If larger then data, we have no bounding box.
    SigmaBlur=[] # Spatial de-bluring with Gaussian Kernel of this width. 
    
    NonNegative=True # should we constrain activity and shapes to be non-negative?
    FinalNonNegative=True # should we constrain activity to be non-negative at final iteration?
    Connected=False # should we constrain all spatial component to be connected?
    WaterShed=False # should we constrain all spatial component to have only one watershed component?
    MedianFilt=False # should we perfrom median filtering of shape?
    FineTune=False # Should use the full data at the main iterations (to fine tune shapes)? If not then just extract the activity and not the shapes from the full data
    ThresholdData= True  # threshold data with PSD level
    
    # experimental stuff - don't use for now
    estimateNoise=False # should we tune sparsity and number of neurons to reach estimated noise level?
    PositiveError=False # should we tune sparsity and number of neurons to have only positive residual error?
    FixSupport=False # should we fix non-zero support at main NMF iterations?
    SmoothBackground=False # Should we cut out out peaks from background component?
    SigmaMask=[]    # if not [], then update masks so that they are SigmaMasks around non-zero support of shapes
    
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
        
        ThresholdData= True  # threshold data with PSD level
        FineTune=False
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
        
       
        
        
        

    params_dict=dict([['data_name',data_name],['SuperVoxelize',SuperVoxelize],['NonNegative',NonNegative],
                      ['FinalNonNegative',FinalNonNegative],['mbs',mbs],['TargetAreaRatio',TargetAreaRatio],
                     ['iters',iters],['iters0',iters0],['lam1_s',lam1_s],['MedianFilt',MedianFilt],['SigmaMask',SigmaMask],['FineTune',FineTune],
                     ['updateLambdaIntervals',updateLambdaIntervals],['updateRhoIntervals',updateRhoIntervals],['addComponentsIntervals',addComponentsIntervals],
                     ['estimateNoise',estimateNoise],['PositiveError',PositiveError],['sig',sig],['NumCent',NumCent],['SigmaBlur',SigmaBlur],
                    ['bkg_per',bkg_per],['ds',ds],['sig',sig],['Background_num',Background_num],['Connected',Connected],['WaterShed',WaterShed],
                    ['SmoothBackground',SmoothBackground],['FixSupport',FixSupport],['repeats',repeats],['ThresholdData',ThresholdData]])
    class Bunch(object):
        def __init__(self, adict):
            self.__dict__.update(adict)

    params=Bunch(params_dict)
    
    return params,params_dict

#%% Main script for running NMF

if __name__ == "__main__":
        
    import matplotlib
    matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
        
    import numpy as np
    from BlockLocalNMF import LocalNMF
    import matplotlib.pyplot as plt
    from AuxilaryFunctions import GetFileName,SuperVoxelize,GetData,GetCentersData,ThresholdData
    import cPickle
    
    plt.close('all')
    
    plot_all=True
    do_NMF=True
        
    params,params_dict=GetDefaultParams()
    
    data=GetData(params.data_name)
    
    if params.ThresholdData:
        data=ThresholdData(data)
        
    if params.SuperVoxelize==True:
        data=SuperVoxelize(data)        
            
    if do_NMF==True:         
        for rep in range(params.repeats):  
            
            if params.NumCent>0:
                cent=GetCentersData(data,params.data_name,params.NumCent,rep)
            else:
                cent=np.reshape([],(0,data.ndim-1)) 
                
            if rep>=params.Background_num:
                adaptBias=False
            else:
                adaptBias=True
            MSE_array, shapes, activity = LocalNMF(
                data, cent, params.sig,TargetAreaRatio=params.TargetAreaRatio,updateLambdaIntervals=params.updateLambdaIntervals,addComponentsIntervals=params.addComponentsIntervals,
                WaterShed=params.WaterShed,SigmaMask=params.SigmaMask,PositiveError=params.PositiveError,NonNegative=params.NonNegative,
                FinalNonNegative=params.FinalNonNegative,MedianFilt=params.MedianFilt,verbose=True,lam1_s=params.lam1_s,
                adaptBias=adaptBias,estimateNoise=params.estimateNoise,FineTune=params.FineTune,SigmaBlur=params.SigmaBlur,
                Connected=params.Connected,SmoothBkg=params.SmoothBackground,FixSupport=params.FixSupport,bkg_per=params.bkg_per,iters0=params.iters0,iters=params.iters,mbs=params.mbs, ds=params.ds)
            
            L=len(shapes)
            print str(L)+' components extracted'
            if L<=adaptBias:
                break
            saveName=GetFileName(params_dict,rep)        
            f = file('NMF_Results/'+saveName, 'wb')
            
            results=dict([['MSE_array',MSE_array], ['shapes',shapes],['activity',activity],['cent',cent],['params',params_dict]])
            cPickle.dump(results, f, protocol=cPickle.HIGHEST_PROTOCOL)
            f.close()
#            print 'rep #',str(rep+1), ' finished'  

            data=data- activity.T.dot(shapes.reshape(len(shapes), -1)).reshape(np.shape(data)) #subtract this iteration components from data        
        
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
    
