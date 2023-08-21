import scipy.optimize as opt
from matplotlib.colors import LogNorm

import numpy as np 

# use jupyter "magic" command to tell it to embed plot into the notebook 
import matplotlib.pyplot as plt

def plot_prettier(dpi=200, fontsize=10, usetex=False): 
    '''
    Function to change Matplotlib defaults to make plots look nicer
    
    Parameters: 
    -----------
        dpi - int, "dots per inch" - controls resolution of PNG images that are produced
                by Matplotlib
        fontsize - int, font size to use overall
        usetex - bool, whether to use LaTeX to render fonds of axes labels 
                use False if you don't have LaTeX installed on your system
                
    Returns:
    --------
        None
                
    '''
    plt.rcParams['figure.dpi']= dpi
    plt.rc("savefig", dpi=dpi)
    plt.rc('font', size=fontsize)
    plt.rc('xtick', direction='in')
    plt.rc('ytick', direction='in')
    plt.rc('xtick.major', pad=5)
    plt.rc('xtick.minor', pad=5)
    plt.rc('ytick.major', pad=5)
    plt.rc('ytick.minor', pad=5)
    plt.rc('lines', dotted_pattern = [2., 2.])
    # if you don't have LaTeX installed on your laptop and this statement
    # generates error, comment it out
    if usetex:
        plt.rc('text', usetex=usetex)
    else:
        import matplotlib.font_manager as fm
        plt.rcParams['mathtext.fontset'] = 'cm'
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    
    
def plot_histogram(data, bins=None, density=False, xlabel=' ', ylabel=' ', tickmarks = False, 
                   plot_title=None, figsize=(3.,3.)):
    '''
    A helper function for some customization of histogram plots 

    Parameters:
    -----------
        data: array like, 1d array of data to histogram
        bins: int or list of bin edges to be passed to the Matplotlib's hist function. 
              Default is None
        density: bool, passed to Matplotlib hist; if True plots pdf instead of counts.
              Default is False
        xlabel, ylabel: str, strings with labels for x- and y- axes. 
              If not supplied, plot will not be labeled
        tickmarks: bool, if True tickmarks of individual data values will be plotted along the x-axis.
                   Default is False
        plot_title: str, plot title if needed. Default is None.
        figsize: tuple of floats, figure size 
        
    Returns:
    --------
        None
    '''
    fig = plt.figure(figsize=figsize) # define figure environment
    plt.xlabel(xlabel) # define axis labels
    plt.ylabel(ylabel) 
    
    # plot histogram of values in data
    plt.hist(data, bins=bins, histtype='stepfilled', density=density, 
             facecolor='slateblue', alpha=0.5)
    
    # plot individual values in data as little ticks along x-axis if tickmarks is True
    if tickmarks: 
        plt.plot(data, np.full_like(data, data.max()*0.1), '|k', 
                markeredgewidth=1)
        
    if plot_title is not None:
        plt.title(plot_title, fontsize=3.*figsize[0])

    plt.show()
    


def conf_interval(x, pdf, conf_level):
    return np.sum(pdf[pdf > x])-conf_level

def plot_2d_dist(x, y, xlim, ylim, nxbins, nybins, figsize=(5,5), 
                cmin=1.e-4, cmax=1.0, smooth=None, xpmax=None, ypmax=None, 
                log=False, weights=None, xlabel='x', ylabel='y', 
                clevs=None, fig_setup=None, savefig=None):
    """
    construct and plot a binned, 2d distribution in the x-y plane 
    using nxbins and nybins in x- and y- direction, respectively
    
    log = specifies whether logged quantities are passed to be plotted on log-scale outside this routine
    """
    if fig_setup is None:
        fig, ax = plt.subplots(figsize=figsize)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.xlim(xlim[0], xlim[1])
        plt.ylim(ylim[0], ylim[1])
    else:
        ax = fig_setup
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim); ax.set_ylim(ylim)

    if xlim[1] < 0.: ax.invert_xaxis()

    if weights is None: weights = np.ones_like(x)
    H, xbins, ybins = np.histogram2d(x, y, weights=weights, bins=(np.linspace(xlim[0], xlim[1], nxbins),np.linspace(ylim[0], ylim[1], nybins)))
    
    H = np.rot90(H); H = np.flipud(H); 
             
    X,Y = np.meshgrid(xbins[:-1],ybins[:-1]) 

    if smooth != None:
        from scipy.signal import wiener
        H = wiener(H, mysize=smooth)
        
    H = H/np.sum(H)        
    Hmask = np.ma.masked_where(H==0,H)
    
    if log:
        X = np.power(10.,X); Y = np.power(10.,Y)

    pcol = ax.pcolormesh(X, Y,(Hmask), vmin=cmin*np.max(Hmask), vmax=cmax*np.max(Hmask), cmap=plt.cm.BuPu, norm = LogNorm(), linewidth=0., rasterized=True)
    pcol.set_edgecolor('face')
    
    # plot contours if contour levels are specified in clevs 
    if clevs is not None:
        lvls = []
        for cld in clevs:  
            sig = opt.brentq( conf_interval, 0., 1., args=(H,cld) )   
            lvls.append(sig)
        
        ax.contour(X, Y, H, linewidths=(1.0,0.75, 0.5, 0.25), colors='black', levels = sorted(lvls), 
                norm = LogNorm(), extent = [xbins[0], xbins[-1], ybins[0], ybins[-1]])
    if xpmax is not None:
        ax.scatter(xpmax, ypmax, marker='x', c='orangered', s=20)
    if savefig:
        plt.savefig(savefig,bbox_inches='tight')
    if fig_setup is None:
        plt.show()
    return
