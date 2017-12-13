# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 09:34:06 2017

@author: Daniel
"""
import numpy as np
import json
from AuxilaryFunctions import GetFileName, LoadResults,GetDataFolder
from Demo import GetDefaultParams
import scipy
import matplotlib.pyplot as plt
plt.ion()

plot_locations=True

#%% Load ground truth
DataFolder=GetDataFolder()
GroundTruthDataFolder=DataFolder+'FISSEQ_MIT/rolonies_cylinder/1x/'

with open(GroundTruthDataFolder+ 'fixed.json') as data_file:    
    ground_truth = json.load(data_file)

barcodes_truth=ground_truth['cell_barcodes'].values()
locations_truth=ground_truth['barcode_locations'].values()



#%% Load CNMF results
params,params_dict=GetDefaultParams()
    
last_rep=params.repeats

SaveNames=[]
for rep in range(last_rep):
    name=GetFileName(params_dict,rep)
    SaveNames.append(name)
    
shapes, activity, background_shapes, background_activity,resultsName=LoadResults(SaveNames,params.Background_num)
ResultName='Results/Comp2GroundTruth_' + name
        
#%% activity to barcodes
C_fake=2# number of fake colors

def Activity2Barcodes(activity):
    T=15 #length of sequence
    C=6 # number of colors
    barcodes=[]
    for ll in range(len(activity)):
        A=activity[ll]
        barcode=[]
        for ii in range(T):            
            barcode.append(np.argmax(A[ii*C:(ii+1)*C-C_fake]))
#        if (4 not in barcode) and (5 not in barcode):
        barcodes.append(barcode)
    return barcodes

barcodes= Activity2Barcodes(activity)
#new_barcodes=[list(x) for x in set(tuple(x) for x in barcodes)]
#print len(barcodes)-len(new_barcodes), 'duplicate barcodes'
#
#barcodes=new_barcodes
#%% compare lists - quality of detected components
T=15 #length of sequence

quality=[]
ind=[]
for kk in range(len(barcodes)):
    q= 0
    ind_max=0
    for mm in range(len(barcodes_truth)):
        temp=0
        for pp in range(T):
            temp=temp+float(barcodes[kk][pp]==barcodes_truth[mm][pp])/T 
        if q<temp:
          q=temp
          ind_max=mm
    quality.append(round(q,2))
    ind.append(ind_max)

print quality     

threshold=0.7
np.sum(np.asarray(quality)>threshold)

new_ind=set(ind)
print len(ind)-len(new_ind), 'duplicate indices'



#%% compare lists - location of detected components
alot=float("inf")
R=4
dims=[38,142,220] #[Z,X,Y]
#
#start=round(R/2)/R
#finish=round(1+R/2)/R

start=0
finish=1
    
distances=[] # did the barcode location match with the shape


for kk in range(len(ind)):
    location_list=locations_truth[ind[kk]].values()
    flag=False
    best_loc=[alot,alot,alot]
    d_min=alot
    for ii in range(len(location_list[1])):
        loc=np.copy(location_list[1][ii])
        if loc!=[]:
            for rr in range(len(dims)):
                loc[rr]=int(round((loc[rr]-start*dims[rr])))
#            print loc
            cent=scipy.ndimage.measurements.center_of_mass(shapes[kk])
            cent =map(lambda x: isinstance(x, float) and int(round(x)) or x, cent)
            distance=np.sqrt((loc[0]-cent[0])**2+(loc[1]-cent[1])**2+(loc[2]-cent[2])**2)
      #      print distance
            if distance<d_min:
                d_min=distance
                best_loc=loc
    if plot_locations==True:
        pic_data = np.percentile(shapes[kk], 95, axis=0)    
        plt.figure(figsize=(12, 4. * pic_data.shape[0] / pic_data.shape[1]))
        ax = plt.subplot(111)
        if best_loc[0]==alot:
            mark='x'
            color='red'
            best_loc=[10,10,10]
        else:
            mark='o'
            color='white'
        ax.scatter(best_loc[2], best_loc[1],  marker=mark, c=color)
        plt.hold(True)
        ax.set_title('Estimate vs. truth, q='+ str(quality[kk])+', d='+str(round(d_min,1)))
        ax.imshow(pic_data)
#            raw_input()

    distances.append(round(d_min))
# Plot Results    
print 'q=',q
print 'd=',distances
#%% how many barcodes where missed?
barcodes_existing=0
barcodes_detected=0

for kk in range(len(locations_truth)):
#    if kk in ind:
#        continue
    location_list=locations_truth[kk].values() 
    flag=0
    for ii in range(len(location_list[1])): 
        loc=np.copy(location_list[1][ii])
        if loc!=[]:
            temp_flag=1
            for rr in range(len(dims)):
                cond1=(loc[rr]<=int(dims[rr]*start))
                cond2=(loc[rr]>=int(dims[rr]*finish))
                if cond1 or cond2:
                    temp_flag=0
            if temp_flag==1:
                flag=1
    barcodes_existing=barcodes_existing+flag
    if kk in ind:
        key=np.where(np.array(ind)==kk)[0][0]
        if (distances[key]<10) and (quality[key]>0.7):
            barcodes_detected=barcodes_detected+flag
                
barcodes_missing=barcodes_existing-barcodes_detected
            
print barcodes_existing
print barcodes_detected

#np.savez(ResultName,quality=quality,distances=distances,ind=ind,barcodes_detected=barcodes_detected,barcodes_existing=barcodes_existing)
