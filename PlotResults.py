from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import int
from builtins import round
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
from past.utils import old_div

def PlotAll(SaveNames,params):
    from numpy import  min, max, percentile,asarray,ceil,sqrt
    import numpy as np
    import sys
    from scipy.signal import welch
    from pylab import load
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.backends.backend_pdf import PdfPages
#    from scipy.ndimage.measurements import label    
    from AuxilaryFunctions import GetRandColors, max_intensity,SuperVoxelize,GetData,PruneComponents,SplitComponents,ThresholdShapes,MergeComponents,ThresholdData,make_sure_path_exists
#    from BlockLocalNMF_AuxilaryFunctions import HALS4activity
    from mpl_toolkits.axes_grid1 import make_axes_locatable    
    
    # makse sure relevant folders exist, and add to path    
    Results_folder='Results/'
    make_sure_path_exists(Results_folder)
        
    OASIS_path='OASIS/'   
    make_sure_path_exists(OASIS_path)
    sys.path.append(OASIS_path)
    from functions import deconvolve

    ## plotting params 
    # what to plot 
    plot_activities=True
    plot_activities_PSD=False
    plot_shapes_projections=True
    plot_shapes_slices=False
    plot_activityCorrs=False
    plot_clustered_shape=False
    plot_residual_slices=False
    plot_residual_projections=False
    # videos to generate
    video_shapes=False
    video_residual=False
    video_slices=True
    # what to save
    save_video=False
    save_plot=False
    close_figs=True#close all figs right after saving (to avoid memory overload)
    # PostProcessing   
    Split=False   
    Threshold=False   #threshold shapes in the end and keep only connected components
    Prune=False  # Remove "Bad" components (where bad is defined within SplitComponent fucntion)
    Merge=True # Merge highly correlated nearby components
    FineTune=False # SHould we fine tune activity after post-processing? (mainly after merging)
    IncludeBackground=False #should we include the background as an extracted component?
    
    # how to plot
    detrend=True #should we detrend the data (remove background component)?
    scale=2 #scale colormap to enhance colors
    satuartion_percentile=96 #saturate colormap ont this percentile, when ma=percentile is used
    dpi=200 #for videos
    restrict_support=True #in shape video, zero out data outside support of shapes
    C=4 #number of components to show in shape videos (if larger then number of shapes L, then we automatically set C=L)
    color_map='gray' #'gnuplot'
    frame_rate= 'color' #10.0 #Hz
    
    # Fetch experimental 3D data 
    data=GetData(params.data_name)
    if params.SuperVoxelize==True:
        data=SuperVoxelize(data)
    
    if params.ThresholdData==True:
        data=ThresholdData(data)        
        
    dims=np.shape(data)
    
    if len(dims)<4:
        plot_Shapes2D=False
        video_residual_2D=False
        if plot_shapes_projections or plot_shapes_slices:
            plot_Shapes2D=True
        
        if video_residual==True:
            video_residual_2D=True
    
        plot_shapes_projections=False
        plot_shapes_slices=False
        plot_activityCorrs=False
        plot_clustered_shape=False
        plot_residual_slices=False
        plot_residual_projections=False
        video_shapes=False
        video_residual=False
        video_slices=False
        print('2D data, ignoring 3D plots/video options')
    else:
        plot_Shapes2D=False
        video_residual_2D=False

    
    min_dim=np.argmin(dims[1:])
    denoised_data=0    
    detrended_data=data
    
    for rep in range(len(SaveNames)): 
        resultsName=SaveNames[rep]
        try:
            results=load('NMF_Results/'+SaveNames[rep])
        except IOError:
            if rep==0:
                print('results file not found!!')              
            else:
                break            
        SS=results['shapes']
        AA=results['activity']

        if rep>=params.Background_num:
            adaptBias=False
        else:
            adaptBias=True
            
        if IncludeBackground==True:
            adaptBias=False        
               
        L=len(AA)-adaptBias
        if L==0: #Stop if we encounter a file with zero components
            break
        S=SS[:-adaptBias]
        b=SS[L:(L+adaptBias)]
        A=AA[:-adaptBias]
        f=AA[L:(L+adaptBias)]
        if rep==0:
            shapes=S
            activity=A
            background_shapes=b
            background_activity=f
        else:
            shapes=np.append(shapes,S,axis=0)
            activity=np.append(activity,A,axis=0) 
            background_shapes=np.append(background_shapes,b,axis=0)
            background_activity=np.append(background_activity,f,axis=0) 
        
    L=len(shapes)
    adaptBias=0
    
    if Split==True:
        shapes,activity,L,all_local_max=SplitComponents(shapes,activity,adaptBias)   
    
    if Merge==True:
        shapes,activity,L=MergeComponents(shapes,activity,L,threshold_activity=0.95,threshold_shape=0.99,sig=10)
        
    if Prune==True:
#           deleted_indices=[5,9,11,14,15,17,24]+range(25,36)
        shapes,activity,L=PruneComponents(shapes,activity,L,params.TargetAreaRatio)
    
    activity_NonNegative=np.copy(activity)
    activity_NonNegative[activity_NonNegative<0]=0
    activity_noisy=np.copy(activity_NonNegative)
    if FineTune==True:
        for ll in range(L):
            activity[ll], spikes, baseline, g, lam = deconvolve(activity_NonNegative[ll],optimize_g=10,penalty=0)
#            activity,background_activity,S,bl,c1,sn,g,junk = update_temporal_components(data.reshape((len(data),-1)).transpose(), shapes.reshape((len(shapes),-1)).transpose(), background_shapes.reshape((len(background_shapes),-1)).transpose(), activity,background_activity,**options['temporal_params'])
        activity_noisy=np.copy(activity_NonNegative)
        activity_NonNegative=activity
    
    print(str(L)+' shapes detected')
    
    detrended_data= detrended_data - background_activity.T.dot(background_shapes.reshape((len(background_shapes), -1))).reshape(dims)        

    if Threshold==True:            
        shapes=ThresholdShapes(shapes,adaptBias,[],MaxRatio=[])
                
    if plot_residual_projections or video_shapes or video_residual or video_slices or video_residual_2D:
        colors=GetRandColors(L)
        color_shapes=np.transpose(shapes.reshape(L, -1,1)*colors,[1,0,2]) #weird transpose for tensor dot product next line
        denoised_data = denoised_data + (activity_NonNegative.T.dot(color_shapes)).reshape(tuple(dims)+(3,))   
        residual = detrended_data - activity_NonNegative.T.dot(shapes.reshape(L, -1)).reshape(dims)
    
    if detrend==True:
        data=detrended_data
 
#%% After loading loop - Normalize (colored) denoised data
    if plot_residual_projections or video_shapes or video_residual or video_slices or video_residual_2D:
#       denoised_data=denoised_data/np.max(denoised_data)
       denoised_data=old_div(denoised_data,np.percentile(denoised_data[denoised_data>0],99.5))  #%% normalize denoised data range
       denoised_data[denoised_data>1]=1           
    
    #    plt.close('all')
        
    #%% plotting params
    ComponentsInFig=20

    left  = 0.05 # the left side of the subplots of the figure
    right = 0.95   # the right side of the subplots of the figure
    bottom = 0.05   # the bottom of the subplots of the figure
    top = 0.95      # the top of the subplots of the figure
    wspace = 0.1   # the amount of width reserved for blank space between subplots
    hspace = 0.12  # the amount of height reserved for white space between subplots        
    
              
    #%% ###### Plot Individual neurons' activities
    index=0 #component display index
    sz=np.min([ComponentsInFig,L+adaptBias])
    
#    a=ceil(sqrt(sz))  
#    b=ceil(sz/a)  
    
    a=sz
    b=1
    
    if plot_activities:
        pp = PdfPages(Results_folder + 'Activities'+resultsName+'.pdf')        
        for ii in range(L+adaptBias):
            if index==0:
#                fig0=plt.figure(figsize=(dims[1] , dims[2]))
                 fig0=plt.figure(figsize=(11,18))
            ax = plt.subplot(a,b,index+1)
#            dt=1/30 # 30 Hz sample rate
            time=list(range(len(activity[ii])))
            plt.plot(time,activity_noisy[ii],linewidth=0.5,c='r')
            plt.plot(time,activity[ii],linewidth=3,c='b')
            ma=np.max([np.max(activity[ii]),np.max(activity_noisy[ii])])            
            plt.setp(ax, xticks=[],yticks=[0,ma])
            # component number
            ax.text(0.02, 0.8, str(ii),
                verticalalignment='bottom', horizontalalignment='left',
                transform=ax.transAxes,
                color='black',weight='bold', fontsize=13)
            index+=1   
            if ((ii%ComponentsInFig)==(ComponentsInFig-1)) or ii==(L+adaptBias-1):                 
                index=0
                if save_plot==True:
                    plt.subplots_adjust(left*2, bottom, right, top, wspace, hspace*2)
                    pp.savefig(fig0)    
        pp.close()
        if close_figs:
            plt.close('all')
            
    #%% Plot activities` PSDs
    index=0 #component display index
    sz=np.min([ComponentsInFig,L+adaptBias])
    a=ceil(sqrt(sz))  
    b=ceil(old_div(sz,a))  
    
    if plot_activities_PSD:
        pp = PdfPages(Results_folder + 'ActivityPSDs'+resultsName+'.pdf')        
        for ii in range(L+adaptBias):
            if index==0:
#                fig0=plt.figure(figsize=(dims[1] , dims[2]))
                 fig0=plt.figure(figsize=(11,18))
            ax = plt.subplot(a,b,index+1)
            ff, psd_activity = welch(activity[ii], nperseg=round(old_div(len(activity[ii]), 64)))
            plt.plot(ff,psd_activity,linewidth=3)
            plt.setp(ax, xticks=[],yticks=[0])
            # component number
            ax.text(0.02, 0.8, str(ii),
                verticalalignment='bottom', horizontalalignment='left',
                transform=ax.transAxes,
                color='black',weight='bold', fontsize=13)
            index+=1   
            if ((ii%ComponentsInFig)==(ComponentsInFig-1)) or ii==(L+adaptBias-1):                 
                index=0
                if save_plot==True:
                    plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
                    pp.savefig(fig0)    
        pp.close()
        if close_figs:
            plt.close('all')
            

            

    #%%  2D shapes
    index=0 #component display index
    sz=np.min([ComponentsInFig,L+adaptBias])
    a=ceil(0.5*sqrt(sz))  
    b=ceil(old_div(sz,a))  
    
    if plot_Shapes2D:            
        if save_plot==True:
            pp = PdfPages(Results_folder + 'Shapes2D_'+resultsName+'.pdf')
        for ll in range(L+adaptBias):
            if index==0:
                fig=plt.figure(figsize=(18 , 11))
            ax = plt.subplot(a,b,index+1)  
            temp=shapes[ll]
            mi=0
            try:
                ma=np.percentile(temp[temp>0],satuartion_percentile)
            except IndexError:
                ma=0
            im=plt.imshow(temp,vmin=mi,vmax=ma,cmap=color_map)
            plt.setp(ax,xticks=[],yticks=[])
            mn=int(np.floor(mi))        # colorbar min value
            mx=int(np.ceil(ma))         # colorbar max value
            md=old_div((mx-mn),2)
#                divider = make_axes_locatable(ax)
#                cax = divider.append_axes("right", size="5%", pad=0.05)
#                cb=plt.colorbar(im,cax=cax)
#                    cb.set_ticks([mn,md,mx])
#                    cb.set_ticklabels([mn,md,mx])
            
            # component number
            ax.text(0.02, 0.8, str(ll),
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax.transAxes,
            color='white',weight='bold', fontsize=13)
            #sparsity
            spar_str=str(np.round(np.mean(shapes[ll]>0)*100,2))+'%'
            ax.text(0.02, 0.02, spar_str,
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax.transAxes,
            color='white',weight='bold', fontsize=13)
            #L^p
            for p in range(2,6,2):
                Lp=old_div((np.sum(shapes[ll]**p))**(old_div(1,float(p))),np.sum(shapes[ll]))
                Lp_str=str(np.round(Lp*100,2))+'%' #'L'+str(p)+'='+
                ax.text(0.02+p*0.2, 0.02, Lp_str,
                verticalalignment='bottom', horizontalalignment='left',
                transform=ax.transAxes,
                color='yellow',weight='bold', fontsize=13)         
            index+=1
            if (ll%ComponentsInFig==(ComponentsInFig-1)) or ll==L+adaptBias-1: 
                plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
                index=0
                if save_plot==True:
                    pp.savefig(fig)            
        pp.close()
        if close_figs:
            plt.close('all')
            
    #%% Re-write plot code from here, so that each figure has only ComponentsInFig components         
    #%% ###### Plot Individual neurons' area which is correlated with their activities
    a=ceil(sqrt(L+adaptBias))
    b=ceil(old_div((L+adaptBias),a))
    
    if plot_activityCorrs:
        if save_plot==True:
            pp = PdfPages(Results_folder + 'CorrelationWithActivity'+resultsName+'.pdf')
        for dd in range(len(shapes[0].shape)):
            fig0=plt.figure(figsize=(11,18))
    
            for ii in range(L+adaptBias):
                ax = plt.subplot(a,b,ii+1)
                corr_imag=old_div(np.dot(activity[ii],np.transpose(data,[1,2,0,3])),np.sqrt(np.sum(data**2,axis=0)*np.sum(activity[ii]**2)))
                plt.imshow(np.abs(corr_imag).max(dd),cmap=color_map)
                plt.setp(ax,xticks=[],yticks=[])
            plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
        
            if save_plot==True:
                pp.savefig(fig0)
        pp.close()
        if close_figs:
            plt.close('all')

    #%%  All Shapes projections
    a=ceil(sqrt(L+adaptBias))
    b=ceil(old_div((L+adaptBias),a))

    if plot_shapes_projections:
        if save_plot==True:
            pp = PdfPages(Results_folder + 'Shapes_projections'+resultsName+'.pdf')

        for dd in range(len(shapes[0].shape)):
            fig=plt.figure(figsize=(18 , 11))
            for ll in range(L+adaptBias):
                ax = plt.subplot(a,b,ll+1)  
                temp=shapes[ll].max(dd)
                if dd==2:
                    temp=temp.T
                mi=np.min(shapes[ll])
                ma=np.max(shapes[ll])
                im=plt.imshow(temp,vmin=mi,vmax=ma,cmap=color_map)
                plt.setp(ax,xticks=[],yticks=[])
                mn=int(np.floor(mi))        # colorbar min value
                mx=int(np.ceil(ma))         # colorbar max value
                md=old_div((mx-mn),2)
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cb=plt.colorbar(im,cax=cax)
#                    cb.set_ticks([mn,md,mx])
#                    cb.set_ticklabels([mn,md,mx])
                
                # component number
                ax.text(0.02, 0.8, str(ll),
                verticalalignment='bottom', horizontalalignment='left',
                transform=ax.transAxes,
                color='white',weight='bold', fontsize=13)
#                    #sparsity
                spar_str=str(np.round(np.mean(shapes[ll]>0)*100,2))+'%'
                ax.text(0.02, 0.02, spar_str,
                verticalalignment='bottom', horizontalalignment='left',
                transform=ax.transAxes,
                color='white',weight='bold', fontsize=13)
#                    #L^p
#                    for p in range(2,2,2):
#                        Lp=(np.sum(shapes[ll]**p))**(1/float(p))/np.sum(shapes[ll])
#                        Lp_str=str(np.round(Lp*100,2))+'%' #'L'+str(p)+'='+
#                        ax.text(0.02+p*0.2, 0.02, Lp_str,
#                        verticalalignment='bottom', horizontalalignment='left',
#                        transform=ax.transAxes,
#                        color='yellow',weight='bold', fontsize=13)
            plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
            if save_plot==True:
                pp.savefig(fig)            
        pp.close()
        if close_figs:
            plt.close('all')
    #for ll in range(L+adaptBias):
    #    print 'Sparsity=',np.mean(shapes[ll]>0)
            
   
   
   #%%  All Shapes slices        
    transpose_shape= True # should we transpose shape
    ComponentsInFig=3 # number of components in Figure
    index=0 #component display index
#        z_slices=[0,1,2,3,4,5,6,7,8] #which z slices to look at slice plots/videos
    z_slices=list(range(dims[min_dim+1])) #which z slices to look at slice plots/videos
    
    if plot_shapes_slices:            
        if save_plot==True:
            pp = PdfPages(Results_folder + 'Shapes_slices'+resultsName+'.pdf')
        for ll in range(L+adaptBias):
            if index==0:
                fig=plt.figure(figsize=(18, 11))
            for dd in range(len(z_slices)):                
                ax = plt.subplot(ComponentsInFig,len(z_slices),index*len(z_slices)+dd+1) 
                temp=shapes[ll].take(dd,axis=min_dim)
                if transpose_shape:
                    temp=np.transpose(temp)                                           
                    
                mi=np.min(shapes[ll])
                ma=np.max(shapes[ll])
                im=plt.imshow(temp,vmin=mi,vmax=ma,cmap=color_map)
                plt.setp(ax,xticks=[],yticks=[])
                
                if dd==0:
                    # component number
                    ax.text(0.02, 0.8, str(ll),
                    verticalalignment='bottom', horizontalalignment='left',
                    transform=ax.transAxes,
                    color='white',weight='bold', fontsize=13)
                    #sparsity
                    spar_str=str(np.round(np.mean(shapes[ll]>0)*100,2))+'%'
                    ax.text(0.02, 0.02, spar_str,
                    verticalalignment='bottom', horizontalalignment='left',
                    transform=ax.transAxes,
                    color='white',weight='bold', fontsize=13)
                    mn=int(np.floor(mi))        # colorbar min value
                    mx=int(np.ceil(ma))         # colorbar max value
                    md=old_div((mx-mn),2)
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("bottom", size="5%", pad=0.05)
                    cb=plt.colorbar(im,cax=cax,orientation="horizontal")
                    cb.set_ticks([mn,md,mx])
                    cb.set_ticklabels([mn,md,mx])
                    #L^p
                    for p in range(2,2,2):
                        Lp=old_div((np.sum(shapes[ll]**p))**(old_div(1,float(p))),np.sum(shapes[ll]))
                        Lp_str=str(np.round(Lp*100,2))+'%' #'L'+str(p)+'='+
                        ax.text(0.02+p*0.15, 0.02, Lp_str,
                        verticalalignment='bottom', horizontalalignment='left',
                        transform=ax.transAxes,
                        color='yellow',weight='bold', fontsize=13)
                        
                    
            plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
            index+=1
            if (ll%ComponentsInFig==(ComponentsInFig-1)) or ll==L+adaptBias-1:                    
                if save_plot==True:
                    pp.savefig(fig)    
                index=0
        pp.close()
        if close_figs:
            plt.close('all')
    #for ll in range(L+adaptBias):
    #    print 'Sparsity=',np.mean(shapes[ll]>0)
            
    #%% ###### Plot Individual neurons' shape projection with clustering
    a=ceil(sqrt(L+adaptBias))
    b=ceil(old_div((L+adaptBias),a))
    
    if plot_clustered_shape:
        from sklearn.cluster import spectral_clustering
        pp = PdfPages(Results_folder + 'ClusteredShapes'+resultsName+'.pdf')
        figs=[]
        for dd in range(len(shapes[0].shape)):
            figs.append(plt.figure(figsize=(18 , 11)))
        for ll in range(L):              
            ind=np.reshape(shapes[ll],(1,)+tuple(dims[1:]))>0
            temp=data[np.repeat(ind,dims[0],axis=0)].reshape(dims[0],-1)
            delta=1 #affinity trasnformation parameter
            clust=3 #number of cluster
            similarity=np.exp(old_div(-np.corrcoef(temp.T),delta))                    
            labels = spectral_clustering(similarity, n_clusters=clust, eigen_solver='arpack')
            ind2=np.array(np.nonzero(ind.reshape(-1))).reshape(-1)
            temp_shape=np.repeat(np.zeros_like(shapes[ll]).reshape(-1,1),clust,axis=1)
            for cc in range(clust):
                temp_shape[ind2[labels==cc],cc]=1
            temp_shape=temp_shape.reshape(tuple(dims[1:])+(clust,))

            for dd in range(len(shapes[0].shape)):
                current_fig=figs[dd]
                ax = current_fig.add_subplot(a,b,ll+1)
                if dd==2:
                    temp_shape=np.transpose(temp_shape,axes=[1,0,2,3])
                ax.imshow(temp_shape.max(dd))

                plt.setp(ax,xticks=[],yticks=[])
                plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
        
        if save_plot==True:
            for dd in range(len(shapes[0].shape)):
                current_fig=figs[dd]
                pp.savefig(current_fig)
        pp.close()
        if close_figs:
            plt.close('all')
            
            

    #%% #####  Video Shapes
    if video_shapes:
        components=list(range(min(asarray([C,L]))))
        C=len(components)
        if restrict_support==True:
            shape_support=shapes[components[0]]>0            
            for cc in range(C):
                shape_support=np.logical_or(shape_support,shapes[components[cc]]>0)
            detrended_data=shape_support.reshape((1,)+tuple(dims[1:]))*detrended_data
        
        fig = plt.figure(figsize=(16,7))
        mi = 0
        ma = max(data)*scale
        #mi2 = 0
        #ma2 = max(shapes[ll])*max(activity[ll])
        
        ii=0
        #import colormaps as cmaps
        #cmap=cmaps.viridis
        cmap=color_map
        a=3
        b=1+C
        
        ax1 = plt.subplot(a,b,1)
        im1 = ax1.imshow(data[ii].max(0), vmin=mi, vmax=ma,cmap=cmap)
        title=ax1.set_title('Data')
        #plt.colorbar(im1)
        ax2=[] 
        ax4=[] 
        ax6=[]
        im2=[]
        im4=[]
        im6=[]
        
        for cc in range(C):
            ax2.append(plt.subplot(a,b,2+cc))
            comp=shapes[components[cc]].max(0)*activity_NonNegative[components[cc],ii]
            ma2=max(shapes[components[cc]].max(0))*max(activity_NonNegative[components[cc]])*scale
            im2.append(ax2[cc].imshow(comp,vmin=0,vmax=ma2,cmap=cmap))
        #ax2[0].set_title('Shape')
        #    plt.colorbar(im2)
        
        ax3 = plt.subplot(a,b,1+b)
        im3 = ax3.imshow(data[ii].max(1), vmin=mi, vmax=ma,cmap=cmap)
        
        #plt.colorbar(im3)
        
        for cc in range(C):
            ax4.append(plt.subplot(a,b,2+b+cc))
            comp=shapes[components[cc]].max(1)*activity_NonNegative[components[cc],ii]
            ma2=max(shapes[components[cc]].max(1))*max(activity_NonNegative[components[cc]])*scale
            im4.append(ax4[cc].imshow(comp,vmin=0,vmax=ma2,cmap=cmap))
        
        #plt.colorbar(im4)
        
        ax5 = plt.subplot(a,b,1+2*b)
        im5 = ax5.imshow(np.transpose(detrended_data[ii].max(2)), vmin=mi, vmax=ma,cmap=cmap)
        
        #plt.colorbar(im5)
        for cc in range(C):
            ax6.append(plt.subplot(a,b,2+2*b+cc))
            comp=np.transpose(shapes[components[cc]].max(2))*activity_NonNegative[components[cc],ii]
            ma2=max(shapes[components[cc]].max(2))*max(activity_NonNegative[components[cc]])*scale
            im6.append(ax6[cc].imshow(comp,vmin=0,vmax=ma2,cmap=cmap))
        
        #plt.colorbar(im6)
        
        fig.tight_layout()
        ComponentsActive=np.array([])
        for cc in range(C):
            ComponentsActive=np.append(ComponentsActive,np.nonzero(activity_NonNegative[components[cc]]))
        ComponentsActive=np.unique(ComponentsActive)
        
        def update(tt):
            ii=ComponentsActive[tt]
            im1.set_data(data[ii].max(0))        
            im3.set_data(data[ii].max(1))        
            im5.set_data(np.transpose(data[ii].max(2)))
        
            for cc in range(C): 
                im2[cc].set_data(shapes[components[cc]].max(0)*activity_NonNegative[components[cc],ii])
                im4[cc].set_data(shapes[components[cc]].max(1)*activity_NonNegative[components[cc],ii])
                im6[cc].set_data(np.transpose(shapes[components[cc]].max(2))*activity_NonNegative[components[cc],ii])
            title.set_text('Data, time = %.1f' % ii)
        
        if save_video==True:
            writer = animation.writers['ffmpeg'](fps=10)
            ani = animation.FuncAnimation(fig, update, frames=len(ComponentsActive), blit=True, repeat=False)
            if restrict_support==True:
                ani.save(Results_folder + 'Shapes_Restricted'+resultsName+'.mp4',dpi=dpi,writer=writer)
            else:                        
                ani.save(Results_folder + 'Shapes_'+resultsName+'.mp4',dpi=dpi,writer=writer)
        else:
            ani = animation.FuncAnimation(fig, update, frames=len(ComponentsActive), blit=True, repeat=False)
            plt.show()

    
    #%% ##### Plot denoised projection - Results

    if plot_residual_projections==True:
        
        dims=data.shape
        cmap=color_map         
        
        pic_residual=percentile(residual, 95, axis=0)
        pic_denoised = max_intensity(denoised_data, axis=0)
        pic_data=percentile(data, 95, axis=0)
        
        left  = 0.05 # the left side of the subplots of the figure
        right = 0.95   # the right side of the subplots of the figure
        bottom = 0.05   # the bottom of the subplots of the figure
        top = 0.95      # the top of the subplots of the figure
        wspace = 0.05   # the amount of width reserved for blank space between subplots
        hspace = 0.05  # the amount of height reserved for white space between subplots
        
        
        fig1=plt.figure(figsize=(11,18))
        mi=min(pic_data)
        ma=max(pic_data)
        ax = plt.subplot(311)
        im=ax.imshow(pic_data.max(0),vmin=mi,vmax=ma,cmap=cmap)
        ax.set_title('Data')
        plt.colorbar(im)
        plt.setp(ax,xticks=[],yticks=[])
        ax2 = plt.subplot(312)
        im2=ax2.imshow(max_intensity(pic_denoised,0),interpolation='None')
        ax2.set_title('Denoised')
        plt.setp(ax,xticks=[],yticks=[])
        plt.colorbar(im2)
        ax3 = plt.subplot(313)
        im3=ax3.imshow(pic_residual.max(0),cmap=cmap)
        ax3.set_title('Residual')
        plt.setp(ax,xticks=[],yticks=[])
        plt.colorbar(im3)
        plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
        
        fig2=plt.figure(figsize=(11,18))
        mi=min(pic_data)
        ma=max(pic_data)
        ax = plt.subplot(311)
        im=ax.imshow(pic_data.max(1),vmin=mi,vmax=ma,cmap=cmap)
        ax.set_title('Data')
        plt.colorbar(im)
        plt.setp(ax,xticks=[],yticks=[])
        ax2 = plt.subplot(312)
        im2=ax2.imshow(max_intensity(pic_denoised,1),interpolation='None')
        ax2.set_title('Denoised')
        plt.colorbar(im2)
        plt.setp(ax,xticks=[],yticks=[])
        ax3 = plt.subplot(313)
        im3=ax3.imshow(pic_residual.max(1),cmap=cmap)
        ax3.set_title('Residual')
        plt.colorbar(im3)
        plt.setp(ax,xticks=[],yticks=[])
        plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
        
        fig3=plt.figure(figsize=(11,18))
        mi=min(pic_data)
        ma=max(pic_data)
        ax = plt.subplot(311)
        im=ax.imshow(pic_data.max(2).T,vmin=mi,vmax=ma,cmap=cmap)
        ax.set_title('Data')
        plt.colorbar(im)
        plt.setp(ax,xticks=[],yticks=[])
        ax2 = plt.subplot(312)
        im2=ax2.imshow(np.transpose(max_intensity(pic_denoised,2),[1,0,2]),interpolation='None')
        ax2.set_title('denoised')
        plt.setp(ax,xticks=[],yticks=[])
        plt.colorbar(im2)
        ax3 = plt.subplot(313)
        im3=ax3.imshow(np.transpose(pic_residual.max(2)),cmap=cmap)
        ax3.set_title('Residual')
        plt.colorbar(im3)
        plt.setp(ax,xticks=[],yticks=[])
        plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
    
        if save_plot==True:
            pp = PdfPages(Results_folder + 'Data_Denoised_Residual_Projections'+resultsName+'.pdf')
            pp.savefig(fig1)
            pp.savefig(fig2)
            pp.savefig(fig3)
            pp.close()
    
    
    #fig = plt.figure()
    #plt.plot(MSE_array)
    #plt.xlabel('Iteration')
    #plt.ylabel('MSE')
    #plt.show()
    
     #%% ##### Plot denoised slices - Results
    z_slices=[0,2,4,6,8] #which z slices to look at slice plots/videos
#    z_slices=list(range(dims[min_dim+1])) #which z slices to look at slice plots/videos
    D=len(z_slices)
    if plot_residual_slices==True:
        
        dims=data.shape
        cmap=color_map         
        
        pic_residual=percentile(residual, 95, axis=0)
        pic_denoised = max_intensity(denoised_data, axis=0)
        pic_data=percentile(data, 95, axis=0)
        
        
        a=3 #number of rows
        fig1=plt.figure(figsize=(18,11))
        mi=min(pic_data)
        ma=max(pic_data)
        for kk in range(D):        
            ax2 = plt.subplot(a,D,kk+1)
            temp=np.squeeze(np.take(pic_denoised,(z_slices[kk],),axis=min_dim))
            im2=ax2.imshow(temp,interpolation='None')
            ax2.set_title('Denoised')
            plt.setp(ax2,xticks=[],yticks=[])
            plt.colorbar(im2)
            ax = plt.subplot(a,D,kk+D+1)
            temp=np.squeeze(np.take(pic_data,(z_slices[kk],),axis=min_dim))
            im=ax.imshow(temp,vmin=mi,vmax=ma,cmap=cmap)
            ax.set_title('Data')
            plt.colorbar(im)
            plt.setp(ax,xticks=[],yticks=[])
            ax3 = plt.subplot(a,D,kk+2*D+1)
            temp=np.squeeze(np.take(pic_residual,(z_slices[kk],),axis=min_dim))
            im3=ax3.imshow(temp,cmap=cmap)
            ax3.set_title('Residual')
            plt.setp(ax3,xticks=[],yticks=[])
            plt.colorbar(im3)
            plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
            
        
            if save_plot==True:
                pp = PdfPages(Results_folder + 'Data_Denoised_Residual_Slice_'+resultsName+'.pdf')
                pp.savefig(fig1)
                pp.close()
        
        
    #fig = plt.figure()
    #plt.plot(MSE_array)
    #plt.xlabel('Iteration')
    #plt.ylabel('MSE')
    #plt.show()


 
    #%% #####  2D Video Residual    
    if video_residual_2D:
        fig = plt.figure(figsize=(16,7))
        mi = 0
        ma = np.percentile(data,satuartion_percentile)
        mi3 = 0
        ma3 = old_div(ma,np.max([np.floor(old_div(ma,np.percentile(residual[residual>0],satuartion_percentile))),1]))

        ii=0
        #import colormaps as cmaps
        #cmap=cmaps.viridis
        cmap=color_map        
        
        a=1
        b=3
        
        im_array=[]
        temp=np.shape(data[ii])                   

        ax1 = plt.subplot(a,b,1)            

        pic=denoised_data[ii]
        im_array += [ax1.imshow(pic,interpolation='None')]
        ax1.set_title('Denoised')
        plt.setp(ax1,xticks=[],yticks=[])
        
        ax2 = plt.subplot(a,b,2)
        pic=data[ii]
            
        im_array += [ax2.imshow(pic, vmin=mi, vmax=ma,cmap=cmap)]
        title=ax2.set_title('Data')  
        plt.setp(ax2,xticks=[],yticks=[])
        divider = make_axes_locatable(ax2)
        cax2 = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im_array[-1], cax=cax2)          

        
        
        ax3 = plt.subplot(a,b,3)            
        pic=residual[ii]   
        im_array += [ax3.imshow(pic, vmin=mi3, vmax=ma3,cmap=cmap)]
        ax3.set_title('Residual x' + '%.1f' % (old_div(ma,ma3)))
        plt.setp(ax3,xticks=[],yticks=[])
        divider = make_axes_locatable(ax3)
        cax3 = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im_array[-1], cax=cax3)
        

#        fig.tight_layout()
        plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
            
        def update(ii):
            im_array[0].set_data(denoised_data[ii])
            im_array[1].set_data(data[ii])        
            im_array[2].set_data(residual[ii])                     
            
            if frame_rate==[]:
                title.set_text('Data, time = %.1f' % ii)
            elif frame_rate=='color':#%for Fisseq
                title.set_text('Data, frame = {0:d}, color={1:d}'.format(ii,np.mod(ii,4)))
            else:
                title.set_text('Data, time = %.2f sec' % (old_div(ii,frame_rate)))
        
        if save_video==True:
            writer = animation.writers['ffmpeg'](fps=10)
            ani = animation.FuncAnimation(fig, update, frames=len(data), blit=False, repeat=False)
            ani.save(Results_folder + 'Data_Denoised_Residual_2D_' +resultsName+'.avi',dpi=dpi,writer=writer)
        else:
            ani = animation.FuncAnimation(fig, update, frames=len(data), blit=False, repeat=False)
            plt.show()  
   
    #%% #####  Video Projections Residual    
    if video_residual:
        fig = plt.figure(figsize=(16,7))
        mi = 0
        ma = max(data)*scale
        mi3 = 0
        ma3 = max(residual)*scale

        ii=0
        #import colormaps as cmaps
        #cmap=cmaps.viridis
        cmap=color_map
        
        spatial_dims_ind=list(range(len(dims)-1))
        D=len(spatial_dims_ind)
        a=D
        b=3
        
        im_array=[]
        transpose_flags=[]
        for kk in range(D):
            transpose_flags+= [False]
            temp=np.shape(data[ii].max(spatial_dims_ind[kk]))
            if temp[0]>temp[1]:
                transpose_flags[kk]=True    
                
        for kk in range(D):
            ax1 = plt.subplot(a,b,D*kk+1)            
            if transpose_flags[kk]==False:
                pic=max_intensity(denoised_data[ii],spatial_dims_ind[kk])
            else:
                pic=np.transpose(max_intensity(denoised_data[ii],spatial_dims_ind[kk]),[1,0,2])  
            im_array += [ax1.imshow(pic,interpolation='None')]
            ax1.set_title('Denoised')
            plt.colorbar(im_array[-1])
            plt.setp(ax1,xticks=[],yticks=[])
            
            ax2 = plt.subplot(a,b,D*kk+2)
            if transpose_flags[kk]==False:
                pic=data[ii].max(spatial_dims_ind[kk])
            else:
                pic=np.transpose(data[ii].max(spatial_dims_ind[kk]))
                
            im_array += [ax2.imshow(pic, vmin=mi, vmax=ma,cmap=cmap)]
            title=ax2.set_title('Data')            
            plt.colorbar(im_array[-1])
            plt.setp(ax2,xticks=[],yticks=[])
            
            ax3 = plt.subplot(a,b,D*kk+3)            
            if transpose_flags[kk]==False:
                pic=residual[ii].max(spatial_dims_ind[kk])
            else:
                pic=np.transpose(residual[ii].max(spatial_dims_ind[kk]))        
            im_array += [ax3.imshow(pic, vmin=mi3, vmax=ma3,cmap=cmap)]
            ax3.set_title('Residual')
            plt.colorbar(im_array[-1])
            plt.setp(ax3,xticks=[],yticks=[])

#        fig.tight_layout()
        plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
            
        def update(ii):
            for kk in range(D):
                if transpose_flags[kk]==False:
                    im_array[kk*D].set_data(max_intensity(denoised_data[ii],spatial_dims_ind[kk]))
                    im_array[kk*D+1].set_data(data[ii].max(spatial_dims_ind[kk]))        
                    im_array[kk*D+2].set_data(residual[ii].max(spatial_dims_ind[kk]))                     
                else:
                    im_array[kk*D].set_data(np.transpose(max_intensity(denoised_data[ii],spatial_dims_ind[kk]),[1,0,2]))
                    im_array[kk*D+1].set_data(np.transpose(data[ii].max(spatial_dims_ind[kk])))        
                    im_array[kk*D+2].set_data(np.transpose(residual[ii].max(spatial_dims_ind[kk])))                     
            
            title.set_text('Data, time = %.1f' % ii)
        
        if save_video==True:
            writer = animation.writers['ffmpeg'](fps=10)
            ani = animation.FuncAnimation(fig, update, frames=len(data), blit=False, repeat=False)
            ani.save(Results_folder + 'Data_Denoised_Residual_Projections'+resultsName+'.mp4',dpi=dpi,writer=writer)
        else:
            ani = animation.FuncAnimation(fig, update, frames=len(data), blit=False, repeat=False)
            plt.show()  
            
    #%% #####  Video Slices Residual    
    z_slices=[0,2,4,6,8] #which z slices to look at slice plots/videos    
#    z_slices=list(range(dims[min_dim+1])) #which z slices to look at slice plots/videos
    
    if video_slices:
        fig = plt.figure(figsize=(16,7))
        mi = 0
        ma = np.percentile(data[data>0],satuartion_percentile)
        mi3 = 0
        ma3 = np.percentil(data[data>0],satuartion_percentile)

        ii=0
        #import colormaps as cmaps
        #cmap=cmaps.viridis
        cmap=color_map
        a=3
        
        D=len(z_slices) #number of spatial dimensions
        im_array=[]
        transpose_flag= True
                
        for kk in range(D):
            ax1 = plt.subplot(a,D,kk+1)            
            temp=np.squeeze(np.take(denoised_data[ii],(z_slices[kk],),axis=min_dim))
            if transpose_flag==False:
                pic=temp
            else:
                pic=np.transpose(temp,[1,0,2])  
            im_array += [ax1.imshow(pic,interpolation='None')]
            ax1.set_title('Denoised, z='+ str(z_slices[kk]+1))
            plt.colorbar(im_array[-1])
            plt.setp(ax1,xticks=[],yticks=[])
            
            ax2 = plt.subplot(a,D,kk+D+1)
            temp=np.squeeze(np.take(data[ii],(z_slices[kk],),axis=min_dim))
            if transpose_flag==False:
                pic=temp
            else:
                pic=np.transpose(temp)
                
            im_array += [ax2.imshow(pic, vmin=mi, vmax=ma,cmap=cmap)]
            title=ax2.set_title('Data')            
            plt.colorbar(im_array[-1])
            plt.setp(ax2,xticks=[],yticks=[])
            
            ax3 = plt.subplot(a,D,kk+2*D+1) 
            temp=np.squeeze(np.take(residual[ii],(z_slices[kk],),axis=min_dim))
            if transpose_flag==False:
                pic=temp
            else:
                pic=np.transpose(temp)       
            im_array += [ax3.imshow(pic, vmin=mi3, vmax=ma3,cmap=cmap)]
            ax3.set_title('Residual')
            plt.colorbar(im_array[-1])
            plt.setp(ax3,xticks=[],yticks=[])

#        fig.tight_layout()
        plt.subplots_adjust(left, bottom, right, top, wspace, hspace)        
        
        def update(ii):
            for kk in range(D):
                temp1=np.squeeze(np.take(denoised_data[ii],(z_slices[kk],),axis=min_dim))
                temp2=np.squeeze(np.take(data[ii],(z_slices[kk],),axis=min_dim))
                temp3=np.squeeze(np.take(residual[ii],(z_slices[kk],),axis=min_dim))
                if transpose_flag==False:                    
                    im_array[a*kk].set_data(temp1)
                    im_array[a*kk+1].set_data(temp2)        
                    im_array[a*kk+2].set_data(temp3)                     
                else:
                    im_array[a*kk].set_data(np.transpose(temp1,[1,0,2]))
                    im_array[a*kk+1].set_data(np.transpose(temp2))        
                    im_array[a*kk+2].set_data(np.transpose(temp3))                     
            
            title.set_text('Data, time = %.1f' % ii)
        
        if save_video==True:
            writer = animation.writers['ffmpeg'](fps=10)
            ani = animation.FuncAnimation(fig, update, frames=len(data), blit=False, repeat=False)
            ani.save(Results_folder + 'Data_Denoised_Residual_Slices'+resultsName+'.mp4',dpi=dpi,writer=writer)
        else:
            ani = animation.FuncAnimation(fig, update, frames=len(data), blit=False, repeat=False)
            plt.show()              
            
    #
    #fig = plt.figure(figsize=(dims[1] , dims[2]))
    #mi = min(data)
    #ma = max(data)
    #dpi=100
    #
    #
    #ax = plt.subplot(131)
    #im = ax.imshow(data[ii], vmin=mi, vmax=ma)
    #ax2 = plt.subplot(132)
    #im2 = ax2.imshow(residual[ii], vmin=mi, vmax=ma)
    #ax3 = plt.subplot(133)
    #im3 = ax3.imshow(denoised_data[ii], vmin=mi, vmax=ma)
    #def update(ii):
    #    im.set_data(data[ii])
    ##    ax.set_title('Data t='+str(ii))
    #    im2.set_data(residual[ii])
    ##    ax2.set_title('Residual')
    #    im3.set_data(denoised_data[ii])
    ##    ax3.set_title('Denoised')
    #
    #
    #ani = animation.FuncAnimation(fig, update, frames=len(data), blit=False, interval=30, repeat=False)
    #ani.save('MaxProjections'+str(dd)+'.mp4',writer=writer,dpi=dpi)
    
     
        #%%  Figure for grant
        #left  = 0.05 # the left side of the subplots of the figure
        #right = 0.95   # the right side of the subplots of the figure
        #bottom = 0.1   # the bottom of the subplots of the figure
        #top = 0.95      # the top of the subplots of the figure
        #wspace = 0.01   # the amount of width reserved for blank space between subplots
        #hspace = 0.01  # the amount of height reserved for white space between subplots
        #font = {'family' : 'normal',
        #        'weight' : 'bold',
        #        'size'   : 22}
        #
        #matplotlib.rc('font', **font)
        #
        #pp = PdfPages('GrantFigure.pdf')
        #components=[20,3,26,24]
        #a=len(components)
        #b=4
        #fig=plt.figure(figsize=(16 , 10.5))
        #fs=10.0 #sampling rate
        #t=array(range(len(activity[0])))/fs
        #for dd in range(3):
        #    for ll in range(a):
        #        ax = plt.subplot(b,a,ll+1+dd*a)  
        #        temp=shapes[components[ll]].max(dd)
        #        temp=temp[:,:-20]
        #        if dd==2:
        #            temp=transpose(temp)
        #        mi=min(temp[temp>0])
        #        ma=max(temp)
        #        plt.imshow(temp,vmin=mi,vmax=ma,cmap=color_map)
        #        plt.setp(ax,xticks=[],yticks=[])
        #        if dd==0:
        #            ax.set_title('Component #'+str(ll+1))
        #        if ll==0:
        #            if dd==0:
        #                plt.ylabel('x-projection')
        #            elif dd==1:
        #                plt.ylabel('y-projection')
        #            elif dd==2:
        #                    plt.ylabel('z-projection')
        #                    
        #for ii in range(a):
        #    ax = plt.subplot(a,b,a*(b-1)+ii+1)
        #    plt.plot(t,activity[components[ii]],color='black',linewidth=2)
        #    plt.xlim([t[0],t[-1]])
        #    plt.setp(ax,yticks=[],xticks=[0,20,40,60,80])
        #    plt.xlabel('t [sec]')
        #    if ii==0:
        #        plt.ylabel('activity [AU]')
        #plt.subplots_adjust(left, bottom, right, top, wspace, hspace)
        #
        #pp.savefig(fig)
        #pp.close()
        
        
        
        
        #%%  Overlay Shapes
        #cmaps =  ['BuGn', 'BuPu','GnBu', 'Greens',
        #          'Greys', 'Oranges', 'OrRd','PuBu', 'PuBuGn',
        #          'PuRd', 'Purples', 'RdPu','Reds', 'YlGn', 
        #          'YlGnBu', 'YlOrBr', 'YlOrRd','Blues']
        #cmaps =  [ 'afmhot', 'bone', 'copper', 'pink']
        #pp = PdfPages('Overlay_Shapes_Thresholded.pdf')
        ##components=[12,20,3,27]
        #for dd in range(len(dims)):
        #    fig=plt.figure(figsize=(dims[1] , dims[2]))
        #    first=True
        #    ind=0
        #    for ll in range(48):            
        #        temp=shapes[ll].max(dd)
        #        mi=min(temp[temp>0])
        #        ma=max(temp)
        #        if first==True:
        #            first=False
        #        else:
        #            temp=np.ma.masked_where(temp==0,temp)            
        ##        plt.imshow(temp,interpolation='bilinear',cmap=plt.get_cmap(cmaps[ind]),vmin=mi,vmax=ma,alpha=1-0.2*ind)
        #        plt.imshow(temp,interpolation='nearest',vmin=mi,vmax=ma,alpha=0.2)
        #        ind+=1
        #    pp.savefig(fig)
        #
        #pp.close()
        
        #def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
#    '''
#    Function to offset the "center" of a colormap. Useful for
#    data with a negative min and positive max and you want the
#    middle of the colormap's dynamic range to be at zero
#
#    Input
#    -----
#      cmap : The matplotlib colormap to be altered
#      start : Offset from lowest point in the colormap's range.
#          Defaults to 0.0 (no lower ofset). Should be between
#          0.0 and `midpoint`.
#      midpoint : The new center of the colormap. Defaults to 
#          0.5 (no shift). Should be between 0.0 and 1.0. In
#          general, this should be  1 - vmax/(vmax + abs(vmin))
#          For example if your data range from -15.0 to +5.0 and
#          you want the center of the colormap at 0.0, `midpoint`
#          should be set to  1 - 5/(5 + 15)) or 0.75
#      stop : Offset from highets point in the colormap's range.
#          Defaults to 1.0 (no upper ofset). Should be between
#          `midpoint` and 1.0.
#    '''
#    cdict = {
#        'red': [],
#        'green': [],
#        'blue': [],
#        'alpha': []
#    }
#    import matplotlib
#    matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
#
#    import numpy as np
#    import matplotlib.pyplot as plt
#
#    # regular index to compute the colors
#    reg_index = np.linspace(start, stop, 257)
#
#    # shifted index to match the data
#    shift_index = np.hstack([
#        np.linspace(0.0, midpoint, 128, endpoint=False), 
#        np.linspace(midpoint, 1.0, 129, endpoint=True)
#    ])
#
#    for ri, si in zip(reg_index, shift_index):
#        r, g, b, a = cmap(ri)
#
#        cdict['red'].append((si, r, r))
#        cdict['green'].append((si, g, g))
#        cdict['blue'].append((si, b, b))
#        cdict['alpha'].append((si, a, a))
#
#    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
#    plt.register_cmap(cmap=newcmap)
#
#    return newcmap





#%% Main    
if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!

    from AuxilaryFunctions import GetFileName      
    from Demo import GetDefaultParams
    
    params,params_dict=GetDefaultParams()
    last_rep=params.repeats
    
    SaveNames=[]
    for rep in range(last_rep):
        name=GetFileName(params_dict,rep)
        SaveNames.append(name)
        
    PlotAll(SaveNames,params)    