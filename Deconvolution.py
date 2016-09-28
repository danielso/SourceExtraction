# -*- coding: utf-8 -*-
"""A set of routines for estimating the temporal components, given the spatial components and temporal components
@author: agiovann
"""
from scipy.sparse import spdiags,coo_matrix#,csgraph
import scipy
import numpy as np
import sys
import scipy.signal 
import scipy.linalg

from warnings import warn    
#import time

def CNMFSetParms(Y, n_processes, K=30, gSig=[5, 5], ssub=1, tsub=1, p=2, p_ssub=1, p_tsub=1, thr=0.8, method_init= 'greedy_roi', **kwargs):
    """Dictionary for setting the CNMF parameters.
    Any parameter that is not set get a default value specified
    by the dictionary default options
    """

    if type(Y) is tuple:
        dims, T = Y[:-1], Y[-1]
    else:
        dims, T = Y.shape[:-1], Y.shape[-1]

    print 'using ' + str(n_processes) + ' processes'
    n_pixels_per_process = np.prod(dims) / n_processes  # how to subdivide the work among processes

    options = dict()
    options['patch_params'] = {
        'ssub': p_ssub,             # spatial downsampling factor
        'tsub': p_tsub,              # temporal downsampling factor
        'only_init' : False
    }
    options['preprocess_params'] = {'sn': None,                  # noise level for each pixel
                                    # range of normalized frequencies over which to average
                                    'noise_range': [0.25, 0.5],
                                    # averaging method ('mean','median','logmexp')
                                    'noise_method': 'logmexp',
                                    'max_num_samples_fft': 3*1024,
                                    'n_pixels_per_process': n_pixels_per_process,
                                    'compute_g': False,            # flag for estimating global time constant
                                    'p': p,                        # order of AR indicator dynamics
                                    'lags': 5,                     # number of autocovariance lags to be considered for time constant estimation
                                    'include_noise': False,        # flag for using noise values when estimating g
                                    'pixels': None,                 # pixels to be excluded due to saturation
                                    }
    options['init_params'] = {'K': K,                                          # number of components
                              # size of components (std of Gaussian)
                              'gSig': gSig,
                              # size of bounding box
                              'gSiz': list(np.array(gSig, dtype=int) * 2 + 1),
                              'ssub': ssub,             # spatial downsampling factor
                              'tsub': tsub,             # temporal downsampling factor
                              'nIter': 5,               # number of refinement iterations
                              'kernel': None,           # user specified template for greedyROI
                              'maxIter': 5,              # number of HALS iterations
                              'method' : method_init,     # can be greedy_roi or sparse_nmf
                              'max_iter_snmf' : 500,
                              'alpha_snmf' : 10e2,
                              'sigma_smooth_snmf' : (.5,.5,.5),
                              'perc_baseline_snmf': 20
                              }
    options['spatial_params'] = {
        'dims': dims,                   # number of rows, columns [and depths]
        # method for determining footprint of spatial components ('ellipse' or 'dilate')
        'method': 'ellipse',
        'dist': 3,                       # expansion factor of ellipse
        'n_pixels_per_process': n_pixels_per_process,   # number of pixels to be processed by eacg worker
        'medw' : (3, 3),                                # window of median filter
        'thr_method' : 'nrg',                           #  Method of thresholding ('max' or 'nrg')
        'maxthr' : 0.1,                                 # Max threshold
        'nrgthr' : 0.9999,                              # Energy threshold
        'extract_cc' : True,                            # Flag to extract connected components (might want to turn to False for dendritic imaging)
        'se' : np.ones((3, 3), dtype=np.int),           # Morphological closing structuring element
        'ss' : np.ones((3, 3), dtype=np.int)            # Binary element for determining connectivity            
        }
    options['temporal_params'] = {
        'ITER': 2,                   # block coordinate descent iterations
        # method for solving the constrained deconvolution problem ('cvx' or 'cvxpy')
        'method': 'cvxpy',
        # if method cvxpy, primary and secondary (if problem unfeasible for approx
        # solution) solvers to be used with cvxpy, can be 'ECOS','SCS' or 'CVXOPT'
        'solvers': ['ECOS', 'SCS'],
        'p': p,                      # order of AR indicator dynamics
        'memory_efficient': False,
        # flag for setting non-negative baseline (otherwise b >= min(y))
        'bas_nonneg': True,
        # range of normalized frequencies over which to average
        'noise_range': [.25, .5],
        'noise_method': 'logmexp',   # averaging method ('mean','median','logmexp')
                        'lags': 5,                   # number of autocovariance lags to be considered for time constant estimation
                        # bias correction factor (between 0 and 1, close to 1)
                        'fudge_factor': .98,
                        'verbosity': False
    }
    options['merging'] = {
        'thr': thr,
    }
    return options
    
#%%
def constrained_foopsi(fluor, bl = None,  c1 = None, g = None,  sn = None, p = None, method = 'cvxpy', bas_nonneg = True,  
                     noise_range = [.25,.5], noise_method = 'logmexp', lags = 5, fudge_factor = 1., 
                    verbosity = False, solvers=None,**kwargs):
    
    """ Infer the most likely discretized spike train underlying a fluorescence trace 
    
    It relies on a noise constrained deconvolution approach
    
    
    Parameters
    ----------
    fluor: np.ndarray
        One dimensional array containing the fluorescence intensities with
        one entry per time-bin.
    bl: [optional] float
        Fluorescence baseline value. If no value is given, then bl is estimated 
        from the data.
    c1: [optional] float
        value of calcium at time 0
    g: [optional] list,float 
        Parameters of the AR process that models the fluorescence impulse response.
        Estimated from the data if no value is given
    sn: float, optional
        Standard deviation of the noise distribution.  If no value is given, 
        then sn is estimated from the data.        
    p: int
        order of the autoregression model
    method: [optional] string
        solution method for basis projection pursuit 'cvx' or 'cvxpy'
    bas_nonneg: bool
        baseline strictly non-negative        
    noise_range:  list of two elms
        frequency range for averaging noise PSD
    noise_method: string
        method of averaging noise PSD
    lags: int 
        number of lags for estimating time constants    
    fudge_factor: float
        fudge factor for reducing time constant bias
    verbosity: bool     
         display optimization details
    solvers: list string
            primary and secondary (if problem unfeasible for approx solution) solvers to be used with cvxpy, default is ['ECOS','SCS']
    Returns
    -------
    c: np.ndarray float
        The inferred denoised fluorescence signal at each time-bin.
    bl, c1, g, sn : As explained above
    sp: ndarray of float
        Discretized deconvolved neural activity (spikes)
    
    References
    ----------
    * Pnevmatikakis et al. 2016. Neuron, in press, http://dx.doi.org/10.1016/j.neuron.2015.11.037
    * Machado et al. 2015. Cell 162(2):338-350
    """
    
    if p is None:
        raise Exception("You must specify the value of p")
        
    if g is None or sn is None:        
        g,sn = estimate_parameters(fluor, p=p, sn=sn, g = g, range_ff=noise_range, method=noise_method, lags=lags, fudge_factor=fudge_factor)
    if p == 0:       
        c1 = 0
        g = np.array(0)
        bl = 0
        c = np.maximum(fluor,0)
        sp = c.copy()
    else:
        if method == 'cvx':
            c,bl,c1,g,sn,sp = cvxopt_foopsi(fluor, b = bl, c1 = c1, g=g, sn=sn, p=p, bas_nonneg = bas_nonneg, verbosity = verbosity)
                    
        elif method == 'cvxpy':
    
            c,bl,c1,g,sn,sp = cvxpy_foopsi(fluor,  g, sn, b=bl, c1=c1, bas_nonneg=bas_nonneg, solvers=solvers)
    
        else:
            raise Exception('Undefined Deconvolution Method')
    
    return c,bl,c1,g,sn,sp


    

def G_inv_mat(x,mode,NT,gs,gd_vec,bas_flag = True, c1_flag = True):
    """
    Fast computation of G^{-1}*x and G^{-T}*x 
    """
    from scipy.signal import lfilter
    if mode == 1:
        b = lfilter(np.array([1]),np.concatenate([np.array([1.]),-gs]),x[:NT]) + bas_flag*x[NT-1+bas_flag] + c1_flag*gd_vec*x[-1]
        #b = np.dot(Emat,b)
    elif mode == 2:
        #x = np.dot(Emat.T,x)
        b = np.hstack((np.flipud(lfilter(np.array([1]),np.concatenate([np.array([1.]),-gs]),np.flipud(x))),np.ones(bas_flag)*np.sum(x),np.ones(c1_flag)*np.sum(gd_vec*x)))
            
    return b

  
def cvxopt_foopsi(fluor, b, c1, g, sn, p, bas_nonneg, verbosity):
    """Solve the deconvolution problem using cvxopt and picos packages
    """
    try:
        from cvxopt import matrix, spmatrix, spdiag, solvers
        import picos
    except ImportError:
        raise ImportError('Constrained Foopsi requires cvxopt and picos packages.')
    
    T = len(fluor)

    # construct deconvolution matrix  (sp = G*c) 
    G = spmatrix(1.,range(T),range(T),(T,T))

    for i in range(p):
        G = G + spmatrix(-g[i],np.arange(i+1,T),np.arange(T-i-1),(T,T))
        
    gr = np.roots(np.concatenate([np.array([1]),-g.flatten()])) 
    gd_vec = np.max(gr)**np.arange(T)  # decay vector for initial fluorescence
    gen_vec = G * matrix(np.ones(fluor.size))  
    
    # Initialize variables in our problem
    prob = picos.Problem()
    
    # Define variables
    calcium_fit = prob.add_variable('calcium_fit', fluor.size)    
    cnt = 0
    if b is None:
        flag_b = True
        cnt += 1
        b = prob.add_variable('b', 1)
        if bas_nonneg:
            b_lb = 0
        else:
            b_lb = np.min(fluor)
            
        prob.add_constraint(b >= b_lb)
    else:
        flag_b = False

    if c1 is None:
        flag_c1 = True
        cnt += 1
        c1 = prob.add_variable('c1', 1)
        prob.add_constraint(c1 >= 0)
    else:
        flag_c1 = False
    
    # Add constraints    
    prob.add_constraint(G * calcium_fit >= 0)
    res = abs(matrix(fluor.astype(float)) - calcium_fit - b*matrix(np.ones(fluor.size)) - matrix(gd_vec) * c1)
    prob.add_constraint(res < sn * np.sqrt(fluor.size))
    prob.set_objective('min', calcium_fit.T * gen_vec)
    
    # solve problem
    try:
        prob.solve(solver='mosek', verbose=verbosity)
        sel_solver = 'mosek'
#        prob.solve(solver='gurobi', verbose=verbosity)
#        sel_solver = 'gurobi'
    except ImportError:
        warn('MOSEK is not installed. Spike inference may be VERY slow!')
        sel_solver = []
        prob.solver_selection()
        prob.solve(verbose=verbosity)
        
    # if problem in infeasible due to low noise value then project onto the cone of linear constraints with cvxopt
    if prob.status == 'prim_infeas_cer' or prob.status == 'dual_infeas_cer' or prob.status == 'primal infeasible':
        warn('Original problem infeasible. Adjusting noise level and re-solving')   
        # setup quadratic problem with cvxopt        
        solvers.options['show_progress'] = verbosity
        ind_rows = range(T)
        ind_cols = range(T)
        vals = np.ones(T)
        if flag_b:
            ind_rows = ind_rows + range(T) 
            ind_cols = ind_cols + [T]*T
            vals = np.concatenate((vals,np.ones(T)))
        if flag_c1:
            ind_rows = ind_rows + range(T)
            ind_cols = ind_cols + [T+cnt-1]*T
            vals = np.concatenate((vals,np.squeeze(gd_vec)))            
        P = spmatrix(vals,ind_rows,ind_cols,(T,T+cnt))
        H = P.T*P
        Py = P.T*matrix(fluor.astype(float))
        sol = solvers.qp(H,-Py,spdiag([-G,-spmatrix(1.,range(cnt),range(cnt))]),matrix(0.,(T+cnt,1)))
        xx = sol['x']
        c = np.array(xx[:T])
        sp = np.array(G*matrix(c))
        c = np.squeeze(c)
        if flag_b:
            b = np.array(xx[T+1]) + b_lb
        if flag_c1:
            c1 = np.array(xx[-1])
        sn = np.linalg.norm(fluor-c-c1*gd_vec-b)/np.sqrt(T)   
    else: # readout picos solution
        c = np.squeeze(calcium_fit.value)
        sp = np.squeeze(np.asarray(G*calcium_fit.value))        
        if flag_b:    
            b = np.squeeze(b.value)        
        if flag_c1:    
            c1 = np.squeeze(c1.value)                    

    return c,b,c1,g,sn,sp

def cvxpy_foopsi(fluor,  g, sn, b=None, c1=None, bas_nonneg=True,solvers=None):
    '''Solves the deconvolution problem using the cvxpy package and the ECOS/SCS library. 
    Parameters:
    -----------
    fluor: ndarray
        fluorescence trace 
    g: list of doubles
        parameters of the autoregressive model, cardinality equivalent to p        
    sn: double
        estimated noise level
    b: double
        baseline level. If None it is estimated. 
    c1: double
        initial value of calcium. If None it is estimated.  
    bas_nonneg: boolean
        should the baseline be estimated        
    solvers: tuple of two strings
        primary and secondary solvers to be used. Can be choosen between ECOS, SCS, CVXOPT    

    Returns:
    --------
    c: estimated calcium trace
    b: estimated baseline
    c1: esimtated initial calcium value
    g: esitmated parameters of the autoregressive model
    sn: estimated noise level
    sp: estimated spikes 
        
    '''
    try:
        import cvxpy as cvx
    except ImportError:
        raise ImportError('cvxpy solver requires installation of cvxpy.')
    if solvers is None:
        solvers=['ECOS','SCS']
        
    T = fluor.size
    
    # construct deconvolution matrix  (sp = G*c)     
    G=scipy.sparse.dia_matrix((np.ones((1,T)),[0]),(T,T))
    
    for i,gi in enumerate(g):
        G = G + scipy.sparse.dia_matrix((-gi*np.ones((1,T)),[-1-i]),(T,T))
        
    gr = np.roots(np.concatenate([np.array([1]),-g.flatten()])) 
    gd_vec = np.max(gr)**np.arange(T)  # decay vector for initial fluorescence
    gen_vec = G.dot(scipy.sparse.coo_matrix(np.ones((T,1))))                          
 
    c = cvx.Variable(T) # calcium at each time step
    constraints=[]
    cnt = 0
    if b is None:
        flag_b = True
        cnt += 1
        b =  cvx.Variable(1) # baseline value
        if bas_nonneg:
            b_lb = 0
        else:
            b_lb = np.min(fluor)            
        constraints.append(b >= b_lb)
    else:
        flag_b = False

    if c1 is None:
        flag_c1 = True
        cnt += 1
        c1 =  cvx.Variable(1) # baseline value
        constraints.append(c1 >= 0)
    else:
        flag_c1 = False    
    
    thrNoise=sn * np.sqrt(fluor.size)
    
    try:
        objective=cvx.Minimize(cvx.norm(G*c,1)) # minimize number of spikes
        constraints.append(G*c >= 0)
        constraints.append(cvx.norm(-c + fluor - b - gd_vec*c1, 2) <= thrNoise) # constraints
        prob = cvx.Problem(objective, constraints) 
        result = prob.solve(solver=solvers[0])    
        
        if  not (prob.status ==  'optimal' or prob.status == 'optimal_inaccurate'):
            raise ValueError('Problem solved suboptimally or unfeasible')            
        
        print 'PROBLEM STATUS:' + prob.status 
        sys.stdout.flush()
    except (ValueError,cvx.SolverError) as err:     # if solvers fail to solve the problem           
#         print(err) 
#         sys.stdout.flush()
         lam=sn/500;
         constraints=constraints[:-1]
         objective = cvx.Minimize(cvx.norm(-c + fluor - b - gd_vec*c1, 2)+lam*cvx.norm(G*c,1))
         prob = cvx.Problem(objective, constraints)
         try: #in case scs was not installed properly
             try:
                 print('TRYING AGAIN ECOS') 
                 sys.stdout.flush()
                 result = prob.solve(solver=solvers[0]) 
             except:
                 print(solvers[0] + ' DID NOT WORK TRYING ' + solvers[1])
                 result = prob.solve(solver=solvers[1]) 
         except:             
             sys.stderr.write('***** SCS solver failed, try installing and compiling SCS for much faster performance. Otherwise set the solvers in tempora_params to ["ECOS","CVXOPT"]')
             sys.stderr.flush()
             #result = prob.solve(solver='CVXOPT')
             raise
             
         if not (prob.status ==  'optimal' or prob.status == 'optimal_inaccurate'):
            print 'PROBLEM STATUS:' + prob.status 
            sp=fluor
            c=fluor
            b=0
            c1=0                      
            return c,b,c1,g,sn,sp
            #raise Exception('Problem could not be solved')
            
    
    
    sp = np.squeeze(np.asarray(G*c.value))    
    c = np.squeeze(np.asarray(c.value))                
    if flag_b:    
        b = np.squeeze(b.value)        
    if flag_c1:    
        c1 = np.squeeze(c1.value)
        
    return c,b,c1,g,sn,sp
    
    
    
    
def estimate_parameters(fluor, p = 2, sn = None, g = None, range_ff = [0.25,0.5], method = 'logmexp', lags = 5, fudge_factor = 1):
    """
    Estimate noise standard deviation and AR coefficients if they are not present
    p: positive integer
        order of AR system  
    sn: float
        noise standard deviation, estimated if not provided.
    lags: positive integer
        number of additional lags where he autocovariance is computed
    range_ff : (1,2) array, nonnegative, max value <= 0.5
        range of frequency (x Nyquist rate) over which the spectrum is averaged  
    method: string
        method of averaging: Mean, median, exponentiated mean of logvalues (default)
    fudge_factor: float (0< fudge_factor <= 1)
        shrinkage factor to reduce bias
    """
    
    if sn is None:
        sn = GetSn(fluor,range_ff,method)
        
    if g is None:
        if p == 0:
            g = np.array(0)
        else:
            g = estimate_time_constant(fluor,p,sn,lags,fudge_factor)

    return g,sn

def estimate_time_constant(fluor, p = 2, sn = None, lags = 5, fudge_factor = 1.):
    """    
    Estimate AR model parameters through the autocovariance function    
    Inputs
    ----------
    fluor        : nparray
        One dimensional array containing the fluorescence intensities with
        one entry per time-bin.
    p            : positive integer
        order of AR system  
    sn           : float
        noise standard deviation, estimated if not provided.
    lags         : positive integer
        number of additional lags where he autocovariance is computed
    fudge_factor : float (0< fudge_factor <= 1)
        shrinkage factor to reduce bias
        
    Return
    -----------
    g       : estimated coefficients of the AR process
    """    
    

    if sn is None:
        sn = GetSn(fluor)
        
    lags += p
    xc = axcov(fluor,lags)        
    xc = xc[:,np.newaxis]
    
    A = scipy.linalg.toeplitz(xc[lags+np.arange(lags)],xc[lags+np.arange(p)]) - sn**2*np.eye(lags,p)
    g = np.linalg.lstsq(A,xc[lags+1:])[0]
    gr = np.roots(np.concatenate([np.array([1]),-g.flatten()]))
    gr = (gr+gr.conjugate())/2.
    gr[gr>1] = 0.95 + np.random.normal(0,0.01,np.sum(gr>1))
    gr[gr<0] = 0.15 + np.random.normal(0,0.01,np.sum(gr<0))
    g = np.poly(fudge_factor*gr)
    g = -g[1:]    
        
    return g.flatten()
    
def GetSn(fluor, range_ff = [0.25,0.5], method = 'logmexp'):
    """    
    Estimate noise power through the power spectral density over the range of large frequencies    
    Inputs
    ----------
    fluor    : nparray
        One dimensional array containing the fluorescence intensities with
        one entry per time-bin.
    range_ff : (1,2) array, nonnegative, max value <= 0.5
        range of frequency (x Nyquist rate) over which the spectrum is averaged  
    method   : string
        method of averaging: Mean, median, exponentiated mean of logvalues (default)
        
    Return
    -----------
    sn       : noise standard deviation
    """
    

    ff, Pxx = scipy.signal.welch(fluor)
    ind1 = ff > range_ff[0]
    ind2 = ff < range_ff[1]
    ind = np.logical_and(ind1,ind2)
    Pxx_ind = Pxx[ind]
    sn = {
        'mean': lambda Pxx_ind: np.sqrt(np.mean(Pxx_ind/2)),
        'median': lambda Pxx_ind: np.sqrt(np.median(Pxx_ind/2)),
        'logmexp': lambda Pxx_ind: np.sqrt(np.exp(np.mean(np.log(Pxx_ind/2))))
    }[method](Pxx_ind)

    return sn

def axcov(data, maxlag=5):
    """
    Compute the autocovariance of data at lag = -maxlag:0:maxlag
    Parameters
    ----------
    data : array
        Array containing fluorescence data
    maxlag : int
        Number of lags to use in autocovariance calculation
    Returns
    -------
    axcov : array
        Autocovariances computed from -maxlag:0:maxlag
    """
    
    data = data - np.mean(data)
    T = len(data)
    bins = np.size(data)
    xcov = np.fft.fft(data, np.power(2, nextpow2(2 * bins - 1)))
    xcov = np.fft.ifft(np.square(np.abs(xcov)))
    xcov = np.concatenate([xcov[np.arange(xcov.size - maxlag, xcov.size)],
                           xcov[np.arange(0, maxlag + 1)]])
    #xcov = xcov/np.concatenate([np.arange(T-maxlag,T+1),np.arange(T-1,T-maxlag-1,-1)])
    return np.real(xcov/T)
    
def nextpow2(value):
    """
    Find exponent such that 2^exponent is equal to or greater than abs(value).
    Parameters
    ----------
    value : int
    Returns
    -------
    exponent : int
    """
    
    exponent = 0
    avalue = np.abs(value)
    while avalue > np.power(2, exponent):
        exponent += 1
    return exponent        
    

#%%

def update_order(A):
    '''Determines the update order of the temporal components given the spatial
    components by creating a nest of random approximate vertex covers
     Input:
     -------
     A:    np.ndarray
          matrix of spatial components (d x K)

     Outputs:
     ---------
     O:   list of sets
          list of subsets of components. The components of each subset can be updated in parallel
     lo:  list
          length of each subset

    Written by Eftychios A. Pnevmatikakis, Simons Foundation, 2015
    '''
    K = np.shape(A)[-1]
    AA = A.T * A
    AA.setdiag(0)
    F = (AA) > 0
    F = F.toarray()
    rem_ind = np.arange(K)
    O = []
    lo = []
    while len(rem_ind) > 0:
        L = np.sort(app_vertex_cover(F[rem_ind, :][:, rem_ind]))
        if L.size:
            ord_ind = set(rem_ind) - set(rem_ind[L])
            rem_ind = rem_ind[L]
        else:
            ord_ind = set(rem_ind)
            rem_ind = []

        O.append(ord_ind)
        lo.append(len(ord_ind))

    return O[::-1], lo[::-1]


def app_vertex_cover(A):
    ''' Finds an approximate vertex cover for a symmetric graph with adjacency matrix A.

     Parameters
     -----------
     A:    boolean 2d array (K x K)
          Adjacency matrix. A is boolean with diagonal set to 0

     Returns
     --------
     L:   A vertex cover of A
     Written by Eftychios A. Pnevmatikakis, Simons Foundation, 2015
    '''

    L = []
    while A.any():
        nz = np.nonzero(A)[0]          # find non-zero edges
        u = nz[np.random.randint(0, len(nz))]
        A[u, :] = False
        A[:, u] = False
        L.append(u)

    return np.asarray(L)
#%%


def make_G_matrix(T,g):
    ''' create matrix of autoregression to enforce indicator dynamics
    Inputs: 
    T: positive integer
        number of time-bins
    g: nd.array, vector p x 1
        Discrete time constants
        
    Output:
    G: sparse diagonal matrix
        Matrix of autoregression
    '''    
    if type(g) is np.ndarray:    
        if len(g) == 1 and g < 0:
            g=0
    
#        gs=np.matrix(np.hstack((-np.flipud(g[:]).T,1)))
        gs=np.matrix(np.hstack((1,-(g[:]).T)))
        ones_=np.matrix(np.ones((T,1)))
        G = spdiags((ones_*gs).T,range(0,-len(g)-1,-1),T,T)    
        
        return G
    else:
        raise Exception('g must be an array')
#%%
def constrained_foopsi_parallel(arg_in):
    """ necessary for parallel computation of the function  constrained_foopsi
    """  
   
    Ytemp, nT, jj_, bl, c1, g, sn, argss = arg_in
    T=np.shape(Ytemp)[0]
    #cc_,cb_,c1_,gn_,sn_,sp_ = constrained_foopsi(Ytemp/nT, bl = bl,  c1 = c1, g = g,  sn = sn, **argss)
    cc_,cb_,c1_,gn_,sn_,sp_ = constrained_foopsi(Ytemp, bl = bl,  c1 = c1, g = g,  sn = sn, **argss)
    gd_ = np.max(np.roots(np.hstack((1,-gn_.T))));  
    gd_vec = gd_**range(T)   
    

    C_ = cc_[:].T + cb_ + np.dot(c1_,gd_vec)
    Sp_ = sp_[:T].T
    #Ytemp_ = Ytemp - np.dot(nT,C_).T
    Ytemp_ = Ytemp - C_.T
    
    return C_,Sp_,Ytemp_,cb_,c1_,sn_,gn_,jj_
    
     
#%%
def update_temporal_components(Y, A, b, Cin, fin, bl = None,  c1 = None, g = None,  sn = None, ITER=2, method_foopsi='constrained_foopsi', memory_efficient=False, debug=False, dview=None,**kwargs):
    """Update temporal components and background given spatial components using a block coordinate descent approach.
    
    Parameters
    -----------    

    Y: np.ndarray (2D)
        input data with time in the last axis (d x T)
    A: sparse matrix (crc format)
        matrix of temporal components (d x K)
    b: ndarray (dx1)
        current estimate of background component
    Cin: np.ndarray
        current estimate of temporal components (K x T)   
    fin: np.ndarray
        current estimate of temporal background (vector of length T)
    g:  np.ndarray
        Global time constant (not used)
    bl: np.ndarray
       baseline for fluorescence trace for each column in A
    c1: np.ndarray
       initial concentration for each column in A
    g:  np.ndarray       
       discrete time constant for each column in A
    sn: np.ndarray
       noise level for each column in A       
    ITER: positive integer
        Maximum number of block coordinate descent loops. 
    method_foopsi: string
        Method of deconvolution of neural activity. constrained_foopsi is the only method supported at the moment.               
    n_processes: int
        number of processes to use for parallel computation. Should be less than the number of processes started with ipcluster.
    backend: 'str'
        single_thread no parallelization
        ipyparallel, parallelization using the ipyparallel cluster. You should start the cluster (install ipyparallel and then type 
        ipcluster -n 6, where 6 is the number of processes). 
        SLURM: using SLURM scheduler
    memory_efficient: Bool
        whether or not to optimize for memory usage (longer running times). nevessary with very large datasets  
    **kwargs: dict
        all parameters passed to constrained_foopsi except bl,c1,g,sn (see documentation). Some useful parameters are      
    p: int
        order of the autoregression model
    method: [optional] string
        solution method for constrained foopsi. Choices are
            'cvx':      using cvxopt and picos (slow especially without the MOSEK solver)
            'cvxpy':    using cvxopt and cvxpy with the ECOS solver (faster, default)
    
    solvers: list string
            primary and secondary (if problem unfeasible for approx solution) solvers to be used with cvxpy, default is ['ECOS','SCS']
            
    Note
    --------

    The temporal components are updated in parallel by default by forming of sequence of vertex covers.  
    
    Returns
    --------
    
    C:   np.ndarray
            matrix of temporal components (K x T)
    f:   np.array
            vector of temporal background (length T) 
    S:   np.ndarray            
            matrix of merged deconvolved activity (spikes) (K x T)
    bl:  float  
            same as input    
    c1:  float
            same as input    
    g:   float
            same as input    
    sn:  float
            same as input 
    YrA: np.ndarray
            matrix of spatial component filtered raw data, after all contributions have been removed.            
            YrA corresponds to the residual trace for each component and is used for faster plotting (K x T)
    
    """
    
    if not kwargs.has_key('p') or kwargs['p'] is None:
        raise Exception("You have to provide a value for p")

    d,T = np.shape(Y);    
    nr = np.shape(A)[-1]
    
    
    if  bl is None:
        bl=np.repeat(None,nr)
        
    if  c1 is None:
        c1=np.repeat(None,nr)

    if  g is None:
        g=np.repeat(None,nr)

    if  sn is None:
        sn=np.repeat(None,nr)                        
    
    A = scipy.sparse.hstack((A,coo_matrix(b)))
    S = np.zeros(np.shape(Cin));
    Cin =  np.vstack((Cin,fin));
    C = Cin;
    nA = np.squeeze(np.array(np.sum(np.square(A.todense()),axis=0)))
    #import pdb
    #pdb.set_trace()
    Cin=coo_matrix(Cin)
    #YrA = ((A.T.dot(Y)).T-Cin.T.dot(A.T.dot(A)))
    YA = (A.T.dot(Y).T)*spdiags(1./nA,0,nr+1,nr+1)
    AA = ((A.T.dot(A))*spdiags(1./nA,0,nr+1,nr+1)).tocsr()
    YrA = YA - Cin.T.dot(AA)
    #YrA = ((A.T.dot(Y)).T-Cin.T.dot(A.T.dot(A)))*spdiags(1./nA,0,nr+1,nr+1)
    
    
    Cin=np.array(Cin.todense())    
    for iter in range(ITER):
        O,lo = update_order(A.tocsc()[:,:nr])
        P_=[];
        for count,jo_ in enumerate(O):
            jo=np.array(list(jo_))           
            #Ytemp = YrA[:,jo.flatten()] + (np.dot(np.diag(nA[jo]),Cin[jo,:])).T
            Ytemp = YrA[:,jo.flatten()] + Cin[jo,:].T
            Ctemp = np.zeros((np.size(jo),T))
            Stemp = np.zeros((np.size(jo),T))
            btemp = np.zeros((np.size(jo),1))
            sntemp = btemp.copy()
            c1temp = btemp.copy()
            gtemp = np.zeros((np.size(jo),kwargs['p']));
            nT = nA[jo]            
                        
#            args_in=[(np.squeeze(np.array(Ytemp[:,jj])), nT[jj], jj, bl[jo[jj]], c1[jo[jj]], g[jo[jj]], sn[jo[jj]], kwargs) for jj in range(len(jo))]
            args_in=[(np.squeeze(np.array(Ytemp[:,jj])), nT[jj], jj, None, None, None, None, kwargs) for jj in range(len(jo))]
#            import pdb
#            pdb.set_trace()
            if dview is not None:                    
                #
                if debug:                

                    results = dview.map_async(constrained_foopsi_parallel,args_in)  

                    results.get()

                    for outp in results.stdout:   

                        print outp[:-1]  

                        sys.stdout.flush()            
                                     
                    for outp in results.stderr:   

                        print outp[:-1]  

                        sys.stderr.flush()            
                    
                else:
                    
                    results = dview.map_sync(constrained_foopsi_parallel,args_in)
                
            else:
                
                results = map(constrained_foopsi_parallel,args_in)            
                
                
            for chunk in results:
               
                pars=dict()

                C_,Sp_,Ytemp_,cb_,c1_,sn_,gn_,jj_=chunk                    
               
                Ctemp[jj_,:] = C_[None,:]
                                
                Stemp[jj_,:] = Sp_               
                
                Ytemp[:,jj_] = Ytemp_[:,None]            

                btemp[jj_] = cb_

                c1temp[jj_] = c1_

                sntemp[jj_] = sn_   

                gtemp[jj_,:] = gn_.T  
                   
                bl[jo[jj_]] = cb_

                c1[jo[jj_]] = c1_

                sn[jo[jj_]] = sn_

                g[jo[jj_]]  = gn_.T if kwargs['p'] > 0 else [] #gtemp[jj,:]
                                             
                pars['b'] = cb_

                pars['c1'] = c1_                 

                pars['neuron_sn'] = sn_

                pars['gn'] = gtemp[jj_,np.abs(gtemp[jj,:])>0] 

                pars['neuron_id'] = jo[jj_]

                P_.append(pars)
            
            YrA -= (Ctemp-C[jo,:]).T*AA[jo,:]
            #YrA[:,jo] = Ytemp
            C[jo,:] = Ctemp.copy()            

            S[jo,:] = Stemp
            
#            if (np.sum(lo[:jo])+1)%1 == 0:
            print str(np.sum(lo[:count+1])) + ' out of total ' + str(nr) + ' temporal components updated'
        
        ii=nr        

        #YrA[:,ii] = YrA[:,ii] + np.atleast_2d(Cin[ii,:]).T
        #cc = np.maximum(YrA[:,ii],0)        
        cc = np.maximum(YrA[:,ii] + np.atleast_2d(Cin[ii,:]).T,0)
        YrA -= (cc-np.atleast_2d(Cin[ii,:]).T)*AA[ii,:]      
        C[ii,:] = cc.T
        #YrA = YA - C.T.dot(AA)
        #YrA[:,ii] = YrA[:,ii] - np.atleast_2d(C[ii,:]).T                
        
        if dview is not None:       
            dview.results.clear()   
            

        if scipy.linalg.norm(Cin - C,'fro')/scipy.linalg.norm(C,'fro') <= 1e-3:
            # stop if the overall temporal component does not change by much
            print "stopping: overall temporal component not changing significantly"
            break
        else:
            Cin = C
        
    f = C[nr:,:]
    C = C[:nr,:]
    YrA = np.array(YrA[:,:nr]).T    
    P_ = sorted(P_, key=lambda k: k['neuron_id']) 
    
    
    return C,f,S,bl,c1,sn,g,YrA #,P_