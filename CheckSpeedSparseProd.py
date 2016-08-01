# -*- coding: utf-8 -*-
"""
Created on Thu Jun 02 13:22:44 2016

@author: Daniel
"""





#%%% Check 1
import timeit
import numpy as np

Reps=1000
setup_str="""import numpy as np
N=1000
M=1000
p=0.01
A=np.random.rand(M,N)
v=np.random.rand(N,1)*(np.random.rand(N,1)<p)
"""

exec(setup_str)
#print np.size(np.squeeze(np.nonzero(v)))

s="""np.dot(A,v)""" 

print timeit.timeit(stmt=s, number=Reps,setup=setup_str)

s="""ind=np.squeeze(v>0)
np.dot(np.compress(ind,A,axis=1),np.compress(ind,v,axis=0))"""

print timeit.timeit(stmt=s, number=Reps,setup=setup_str)

s="""ind=np.squeeze(np.nonzero(v))
np.dot(np.take(A,ind,axis=1),np.take(v,ind,axis=0))"""

print timeit.timeit(stmt=s, number=Reps,setup=setup_str)

s="""
ind=np.squeeze(v>0)
q=np.zeros(v.shape)
v_comp=np.compress(ind,v,axis=0)
for mm in range(M):
    q[mm]=np.dot(np.compress(ind,A[mm,:],axis=0),v_comp)
    """
    
print timeit.timeit(stmt=s, number=Reps,setup=setup_str)

#%%% Check 2
import timeit
import numpy as np


Reps=10

setup_str="""import numpy as np
N=10000
M=1000
L=1000
p=0.001
A=np.random.rand(M,N)
V=np.random.rand(N,L)*(np.random.rand(N,1)<p)
"""

exec(setup_str)
#print np.size(np.squeeze(np.nonzero(v)))

s="""np.dot(A,V)""" 

print timeit.timeit(stmt=s, number=Reps,setup=setup_str)

s="""ind=np.squeeze(np.sum(V,1)>0)
np.dot(np.compress(ind,A,axis=1),np.compress(ind,V,axis=0))
"""    

print timeit.timeit(stmt=s, number=Reps,setup=setup_str)

s="""ind=np.squeeze(np.nonzero(np.sum(V,1)))
V_comp=np.take(V,ind,axis=0)
A_comp=np.take(A,ind,axis=1)
np.dot(A_comp,V_comp)
"""    

print timeit.timeit(stmt=s, number=Reps,setup=setup_str)

s="""ind=np.squeeze(np.sum(V,1)>0)
V_comp=np.compress(ind,V,axis=0)
A_comp=np.compress(ind,A,axis=1)
np.dot(A_comp,V_comp)
"""    


print timeit.timeit(stmt=s, number=Reps,setup=setup_str)

s="""Q=np.zeros((M,L))
for mm in range(M):
    ind=(V[:,mm]>0)
    v_comp=np.compress(ind,V[:,mm],axis=0)
    Q[:,mm]=np.dot(np.compress(ind,A,axis=1),v_comp)
"""
    

print timeit.timeit(stmt=s, number=Reps,setup=setup_str)

s="""import scipy.sparse
SA = scipy.sparse.csc_matrix(A)
SV = scipy.sparse.csc_matrix(V)
A*SV
""" 

print timeit.timeit(stmt=s, number=Reps,setup=setup_str)

#%%% Check 3
#  S[ll, mask[ll]] += nan_to_num((C[ll, mask[ll]] np.dot(D[ll], S[:, mask[ll]])-lam1_s[ll,mask[ll]]-lam2_s*S[ll,mask[ll]])/ D[ll, ll])

import timeit
import numpy as np

Reps=100

setup_str="""import numpy as np
N=1
M=1
L=100000
p=0.0001
A=np.random.rand(M,N)
V=np.random.rand(N,L)*(np.random.rand(N,L)<p)
ind=np.squeeze(np.nonzero(np.sum(V,0)))
ind=np.nonzero(V)[0]
V_comp=np.take(V,ind,axis=1)
import scipy.sparse
SV = scipy.sparse.lil_matrix(V)
indBool=np.squeeze(V>0)
"""

exec(setup_str)

exe_str="""
for ll in range(N):
    V[ll]+=np.dot(A[ll],V)
"""

print timeit.timeit(stmt=exe_str, number=Reps,setup=setup_str)

exe_str="""

V_comp=np.take(V,ind,axis=1)
for ll in range(N):
    V[ll,ind]+=np.dot(A[ll],V_comp)
"""

print timeit.timeit(stmt=exe_str, number=Reps,setup=setup_str)

exe_str="""

V_comp=np.compress(indBool,V,axis=1)
for ll in range(N):
    V[ll,ind]+=np.dot(A[ll],V_comp)
"""

print timeit.timeit(stmt=exe_str, number=Reps,setup=setup_str)

#%%

V_comp=np.take(V,ind,axis=1)
for ll in range(N):
    V[ll,ind]+=np.dot(A[ll],V_comp)
"""

print timeit.timeit(stmt=exe_str, number=Reps,setup=setup_str)




exe_str="""
for ll in range(N):
    V[ll,:]+=np.dot(A[ll],V)
"""

print timeit.timeit(stmt=exe_str, number=Reps,setup=setup_str)

exe_str="""
for ll in range(N):
    ind=np.squeeze(np.nonzero(np.sum(V,0)))
    V_comp=np.take(V,ind,axis=1)

    V[ll,ind]+=np.dot(A[ll],V_comp)
"""

#exec(exe_str)

print timeit.timeit(stmt=exe_str, number=Reps,setup=setup_str)

exe_str="""
import scipy.sparse
SV = scipy.sparse.lil_matrix(V)
for ll in range(N):
    SV[ll]+=A[ll]*SV
"""

#exec(exe_str)

print timeit.timeit(stmt=exe_str, number=Reps,setup=setup_str)

exe_str="""
for ll in range(N):
    ind=np.squeeze(np.sum(V,0)>0)
    V_comp=np.compress(ind,V,axis=1)

    V[ll,ind]+=np.dot(A[ll],V_comp)
"""
print timeit.timeit(stmt=exe_str, number=Reps,setup=setup_str)

#exe_str="""
#for ll in range(N):    
#    for kk in range(N):         
#        ind=np.squeeze(np.nonzero(V[kk,:]))
#        V[ll,ind]+=A[ll,kk]*V[kk,ind]
#"""
#
##exec(exe_str)
#
#print timeit.timeit(stmt=exe_str, number=Reps,setup=setup_str)
