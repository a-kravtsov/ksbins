import numpy as np 
from scipy.special import kolmogorov 

def nbin_geom_prior(ns, nb, gamma=0.35): 
    """
    "geometric" prior for the number of bins, similar to the prior used in Bayesian blocks (Scargle et al. 2013)
    
    Parameters:
    -----------
    ns: integer, number of samples
    nb: integer, number of bins
    gamma: float, parameter of the prior pdf 
    
    Returs: prior for a given ns, nb and gamma
    """
    return (1.-gamma) * gamma**nb / (1. - gamma**(ns+1))  


def nbin_uniform_prior(ns, nb):
    return 1.

def ks_uniformity_test(x, xint=None):
    """
    test whether sample in 1d array x is uniformly distributed in the interval [xint[0], xint[1]]
    or [min(x), max(x)] if xint is None
    
    Parameters:
    -----------
    x - 1d numpy array containing sample values 
    xbin - list of size 2 containing limits of x within which to test uniformity of the x distribution
        if None, the range is defined as [x.min(), x.max()]
    
    Returns:
    --------
    float - Kolmogorov probability for D_KS statistic computed using sample CDF and uniform cdf
    """
    x = np.array(x)
    if xint is None: 
        xint = [x.min(), x.max()]

    nx = x.size
    if nx <= 1:
        return 1. # impossible to tell for such small number of samples 
    
    # cdf for the x sample 
    xcdf = np.arange(1, nx+1, 1) / (nx-1)
    
    # compute D_KS statistic 
    dks = nx**0.5 *  np.max(np.abs(xcdf -(np.sort(x) - xint[0])/(xint[1] - xint[0])))
    return kolmogorov(dks) # return value of Kolmogorov probability for this D_KS


def ks_optimal_nbin(x, x_interval=None, nbmin=4, nbmax=500, nfd_min=50, prior_func=nbin_uniform_prior, prior_kwargs={}):
    """
    Find the optimal number of equal-size bins to use based on the KS uniformity test
    
    Parameters:
    -----------
        x: array like, 1d array with values to histogram
        x_interval: list of 2 values with the interval of x to use for bin testing
                 If None, the interval will be defined to be [min(x), max(x)]
        nbmin: int, the minimum number of bins to use. Default is 4.
        nbmax: int, the maximum number of bins to explore. Default is 500.
        nfd_min: int, if the number of values in x is < nfd_min: Freeman-Diaconis rule will 
                be used instead. Default is 50. 
        prior_func: Python function, function implementing prior for the number of bins. 
                    Default is nbin_uniform_prior function. 
                    The format of the function is prior_func(ns, nb) where ns is the number
                    of data samples and nb is the number of bins 
        prior_kwargs: dictionary, keyword parameters to be passed on to prior_func
                    Default is empty dictionary
                
    Returns:
    --------
        nopt: int, the optimal number of bins found
        edges: array like, edges of the identified bins
        pksmax: maximum KS probability found for the optimal number of bins
               if Freeman-Diaconis rule was used instead, pksmax will be -1
    
    """
    x = np.array(x)
    ns = x.size
    if x_interval is None:
        xmin, xmax = x.min(), x.max()
        x_interval = [xmin, xmax]
    else: 
        xmin, xmax = x_interval
    
    pmax = -1 # initialize vari
    
    # Freedman-Diaconis rule
    if ns <= nfd_min:
        hfd = 2. * (np.percentile(x,75) - np.percentile(x,25)) * float(ns)**(-1./3)
        if hfd > 0: 
            nfd = int(np.ceil((x.max()-x.min()) / hfd)) + 1
            counts, edges = np.histogram(x, bins=nfd, range=x_interval)
            return nfd, edges, pmax
            
    # check that maximum number of bins is smaller than minimum number 
    assert(nbmax >= nbmin)
    
    for nbins in range(nbmin, nbmax+1):
        hist, bedges = np.histogram(x, bins=nbins)
        pks = np.ones(nbins)
        for i in range(nbins):
            xd = x[(bedges[i] <= x) & (x <= bedges[i+1])]
            # only non-empty bins affect probability
            if xd.size > 0:
                pks[i] = ks_uniformity_test(xd, xint=[bedges[i], bedges[i+1]]) 
                
        ptot = np.prod(pks) * prior_func(ns, nbins, **prior_kwargs)
        if ptot > pmax:
            pmax, nopt, pksmax = ptot, nbins, pks

    counts, edges = np.histogram(x, bins=nopt, range=x_interval)
    
    return nopt, edges, pksmax


                                     
def ks_adaptive_bins(x, nbmin=4, nbmax=500, x_interval=None, nminb=10, 
                     prior_func=nbin_uniform_prior, prior_kwargs={}):
    """
    function to add local adaptivity to the output by iteratively splitting the bins produced by ks_optimal_nbin
    and checking whether the combine p_ks of the new bins is larger than the previous single bin
    the probability
    
    Parameters:
    -----------
    x: ndarray of dimension ns, locations of ns samples
    nbmin, nbmax: int, the minimum and maximum number of bins to try when determining optimal equal size bins
                       Default values are 4 and 500.
    x_interval: list of 2 elements or None, if list, should contain xmin, xmax values defining binning range
    nminb: integer, the minimum number of particles in bins
    prior_func: Python function, function implementing prior for the number of bins. 
                Default is nbin_uniform_prior function. 
                The format of the function is prior_func(ns, nb) where ns is the number
                of data samples and nb is the number of bins 
    prior_kwargs: dictionary, keyword parameters to be passed on to prior_func
                Default is empty dictionary
    
    Returns:
    --------
    nbins: integer, number of new non-equal width bins
    edges: edges corresponding to these bins, edges can be used with plot.hist or np.histogram as input to bins
        keyword parameter
        
    """
    x = np.sort(x)
    ns = x.size

    # start by finding the optimal number of equal-sized bins using KS method
    nbins, edges, pks = ks_optimal_nbin(x, x_interval=x_interval, nbmin=nbmin, nbmax=nbmax,
                                       prior_func=prior_func, prior_kwargs=prior_kwargs)

    if x_interval is None:
        xmin, xmax = x.min(), x.max()
        x_interval = [xmin, xmax]
    else: 
        xmin, xmax = x_interval
    
    # iterate over bins and test whether splitting each bin at the median sample location
    # improves local KS probability that distribution of samples within each bin is uniform
    nedges_old = 0 
    while edges.size > nedges_old: # iterate until bin edges stop changing 
        nedges_old = edges.size
        pks_new = np.copy(pks)
        edges_new = np.copy(edges)
        
        # examin each bin in the current bin split
        for i in range(nbins):
            xl, xr, pksi = edges[i], edges[i+1], pks[i]
            xd = x[(xl <= x) & (x <= xr)] # select samples in the bin
            nd = xd.size
            # do this only if bin is not empty
            if nd: 
                xmed = xd[nd//2] # bin split location is at the median value of the sample values
                xld = xd[(xl <= xd) & (xd <= xmed)] # samples to the left of the split
                xrd = xd[(xmed <= xd) & (xd <= xr)] # samples to the right of the split
                
                if xld.size > nminb and xrd.size > nminb: # make sure ns in proposed bins is > nminb
                    pl = ks_uniformity_test(xld, xint=[xl, xmed]) # compute KS probabilities for proposed bins
                    pr = ks_uniformity_test(xrd, xint=[xmed, xr])
                    # get counts in current bin configuration 
                    dcounts, dbins = np.histogram(xd, bins=np.sort(edges_new)) 
                    nb = np.size(dcounts>0) # number of non-empty bins
                    pnb1 = prior_func(ns, nb+1, **prior_kwargs) # compute prior pdf for adding one bin
                    if pl * pr * pnb1 > pksi:  # if KS posterior is improved
                        edges_new = np.insert(edges_new, i+1, xmed) # accept new split into edge list 
                        pks_new[i] = pl * pnb1 # record KS posterior for both newly formed bins
                        pks_new = np.insert(pks_new, i+1, pr*pnb1)
                        
        pks = np.copy(pks_new)
        edges = np.copy(edges_new)

    # clean duplicate edges, if any 
    edges = np.sort(np.unique(edges))
    cd, bd = np.histogram(x, bins=edges)
    
    # final stage, clean up, handle empty bins, combine bins that have fewer than nminb samples where possible
    nscount = 0 
    edges_new = [edges[0]]
    for i, cdd in enumerate(cd):
        nscount += cdd
        if nscount >= nminb:
            nscount = 0
            edges_new.append(edges[i])
        elif cdd > 0 and cd[i-1] == 0:
            edges_new.append(edges[i])
            
        elif cdd == 0 and i > 0:
            if cd[i-1] > 0:
                edges_new.append(edges[i])
                nscount = 0
                
    edges_new.append(edges[-1])
    
    return np.size(edges_new)-1, np.sort(np.unique(edges_new))

