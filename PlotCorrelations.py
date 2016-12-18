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
from AuxilaryFunctions import GetFileName  
from scipy.ndimage.filters import gaussian_filter
from Demo import GetDefaultParams

params,params_dict=GetDefaultParams()
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

M=len(shapes_array)

sig=[1,2,3,4,5,6]
cmap='gnuplot'

K=len(sig)
a=np.ceil(np.sqrt(K))
b=2*np.ceil(old_div(K,a))

pp = PdfPages('Components Correlations.pdf')
fig=plt.figure(figsize=(18,11))
for ii in range(K):
    shapes_array=shapes_array.reshape((-1,)+dims_shape)
    shapes_array=gaussian_filter(shapes_array, (0,) + tuple(sig[ii]*np.ones((len(dims_shape),1))))
    shapes_array=shapes_array.reshape(M,-1)
    shapes_cov=np.dot(shapes_array,shapes_array.T)
    shape_vars=np.diag(shapes_cov).reshape(-1,1)
    shapes_corr=np.nan_to_num(old_div(shapes_cov,np.sqrt(np.dot(shape_vars,shape_vars.T))))
    ax=plt.subplot(a,b,2*ii+1)
    im=plt.imshow(shapes_corr, interpolation='none',cmap=cmap)
    plt.colorbar(im)
    plt.title('Spatial Corr. Matrix, Sig='+str(sig[ii]))
    ax2=plt.subplot(a,b,2*ii+2)
    hist, bin_edges = np.histogram(np.ndarray.flatten(shapes_corr),bins=30)
    bins=old_div((bin_edges[:-1]+bin_edges[1:]),2)
    im2=plt.plot(bins,hist,linewidth=2)
    plt.title('Spatial Corr.  Hist., Sig='+str(sig[ii]))
    
pp.savefig(fig)

dims_activity=activity_array[0].shape

fig=plt.figure(figsize=(18,11))
for ii in range(K):
    activity_array=gaussian_filter(activity_array, (0,sig[ii]))
    activity_cov=np.dot(activity_array,activity_array.T)
    activity_vars=np.diag(activity_cov).reshape(-1,1)
    activity_corr=old_div(activity_cov,np.sqrt(np.dot(activity_vars,activity_vars.T)))
    ax2=plt.subplot(a,b,2*ii+1)
    im2=plt.imshow(activity_corr, interpolation='none',cmap=cmap)
    plt.colorbar(im2)
    plt.title('Temporal Corr. Mat, Sig='+str(sig[ii]))
    ax2=plt.subplot(a,b,2*ii+2)
    hist, bin_edges = np.histogram(np.ndarray.flatten(activity_corr),bins=30)
    bins=old_div((bin_edges[:-1]+bin_edges[1:]),2)
    im2=plt.plot(bins,hist,linewidth=2)
    plt.title('Temproa Corr.  Hist., Sig='+str(sig[ii]))
    
pp.savefig(fig)
pp.close()

##%%
#from sklearn.feature_extraction import image
#from sklearn.cluster import spectral_clustering
#
#for ii in range(K):
#    shapes_array=shapes_array.reshape((-1,)+dims_shape)
#
#
#graph = image.img_to_graph(img, mask=mask)
#
## Take a decreasing function of the gradient: we take it weakly
## dependent from the gradient the segmentation is close to a voronoi
#graph.data = np.exp(-graph.data / graph.data.std())
#
## Force the solver to be arpack, since amg is numerically
## unstable on this example
#C=4
#labels = spectral_clustering(graph, n_clusters=C, eigen_solver='arpack')