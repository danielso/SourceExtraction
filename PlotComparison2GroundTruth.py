# -*- coding: utf-8 -*-
"""
Created on Wed May 31 08:44:44 2017

@author: Daniel
"""

import numpy as np
from AuxilaryFunctions import GetFileName
from Demo import GetDefaultParams
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
plt.ion()


params,params_dict=GetDefaultParams()
    
last_rep=params.repeats-1
name=GetFileName(params_dict,last_rep)
ResultName='Results/Comp2GroundTruth_' + name +'.npz'

temp=np.load(ResultName)
ind=temp['ind']
distances=temp['distances']
quality=temp['quality']
print temp['barcodes_existing']
print temp['barcodes_detected']

new_distances=distances[distances<10]
new_quality=quality[distances<10]
new_quality=new_quality[new_quality>0.8]
new_distances=new_distances[new_quality>0.8]
len(new_quality)


H, xedges, yedges =np.histogram2d(new_quality,new_distances,bins=10)
H=H.T
fig = plt.figure(figsize=(10, 9))
X, Y = np.meshgrid(xedges, yedges)
cax=plt.pcolormesh(X,Y, H,vmin=0,vmax=100)
plt.xlabel('Barcode quality [fraction of matching base pairs]')
plt.ylabel('Center Distance [pixels]')
plt.colorbar(cax,ticks=[0,20,40,60,80,100])

pp = PdfPages('Results/Hist'+name+'.pdf')   
pp.savefig(fig)
pp.close()
#plt.hist(new_quality,bins=30)
#plt.hist(new_distances,bins=30)

