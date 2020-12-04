#
# 2/11/2020: Version 3, where I only use one method (Schiebers), and as base case use float(Z)
#          : using integers only if requested.
# Program implements double reptation model (DRM) for
# (a) Blends of monodisperse fractions (phiFxn) given (w, Z)
# (b) Polydisperse blend with logNormal distribution given (Zw and pd)
#	  The latter module plotPhiPD works well only when pd >= 1.01 (which is rather polydisperse)
#	  when pd is small, the distribution formally becomes a Dirac delta, leading to poor numerical integration


import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['seaborn-white', 'myjournal'])
from scipy.special import erf
from scipy.integrate import trapz
from scipy.stats import lognorm
from scipy.interpolate import interp1d

#np.set_printoptions(precision=4)
#clr = [p['color'] for p in plt.rcParams['axes.prop_cycle']]

# ~ import seaborn as sns
# ~ plt.style.use(['seaborn-ticks', 'myjournalFONT'])
# ~ clr = [p['color'] for p in plt.rcParams['axes.prop_cycle']]
#clr = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']


beta = 2.25

####### kernels for TDD computation #########
def gSpecial(x):
    """Better approx to g(x) in deCloiseux; used by Ftdd"""
    return np.pi**2/6. * erf(np.sqrt(0.7477*x))

def Ftdd(Z, t):
    """This is the total kernel Fsr = Fmono^{1/beta}"""

    Zstar = 10.
    tau_d = Z**3
    H     = Z/Zstar
    U     = t/tau_d + 1./H * gSpecial(H*t/tau_d)
    F     = np.zeros(len(U))
    
    for p in range(1, 21,2):
        F += 1/p**2 * np.exp(-p**2*U)
            
    F = F*8/np.pi**2
    
    return F

# phi(t) computation
def pdIntegrand(LZ, t, mu, sigma, w1):
    """
       Here send logZ
       Schieber W(Z) = wL(Z) = Zw(Z) is Gaussian in Log(Z)
       This is the integrand that is numerically integrated in getPhiPD
    """
    wL1 = 1./(sigma[0]*np.sqrt(2. * np.pi)) * np.exp(-(LZ-mu[0])**2/(2.*sigma[0]**2))
    wL2 = 1./(sigma[1]*np.sqrt(2. * np.pi)) * np.exp(-(LZ-mu[1])**2/(2.*sigma[1]**2))
    wL  = w1 * wL1 + (1-w1) * wL2
    intg  = wL * Ftdd(np.exp(LZ), t)
    return intg

def getPhiPD(Zw, pd, w1, isPlot = False, threshold=1e-6):
    """
        Main routine that computes the response of bidisperse LN distributed samples
        w1 is weight fraction of component 1; Zw and pd are two element arrays.
    """
    
    # translate to standard LN distribution specfiers
    sigma  = np.sqrt(np.log(pd))
    mu     = np.log(Zw) - 0.5*np.log(pd)
    
    LTmax  = max(np.log10(pd * 30 * Zw**3))
    t      = np.logspace(0, LTmax)
    phi    = np.zeros(len(t))

    eps    = 1e-16
    
    LZlow  = min(np.log(lognorm.ppf(eps, s=sigma[0], scale=np.exp(mu[0]))),
                 np.log(lognorm.ppf(eps, s=sigma[1], scale=np.exp(mu[1]))), 0)
    LZhi   = max(np.log(lognorm.ppf(1-eps, s=sigma[0], scale=np.exp(mu[0]))),
                 np.log(lognorm.ppf(1-eps, s=sigma[1], scale=np.exp(mu[1]))))
    
    LZgrid = np.linspace(LZlow, LZhi, 100)

    # trapezoidal rule to integrate
    for i, ti in enumerate(t):
        integGrid = pdIntegrand(LZgrid, ti, mu, sigma, w1)
        phi[i]    = (trapz(integGrid, LZgrid))**beta

    cnd = phi > 1e-6
    phi = phi[cnd]
    t   = t[cnd]
        
    if isPlot:
        original_w = w1
        w1  = 1.0
        t1, phi1 = getPhiPD(Zw, pd, w1, isPlot = False)

        w1  = 0.0
        t2, phi2 = getPhiPD(Zw, pd, w1, isPlot = False)
        
        plt.plot(t, phi,label="W1 = "+str(original_w))
        plt.plot(t1, phi1, alpha=0.5,label="W1 = "+str(1.0))
        plt.plot(t2, phi2, alpha=0.5,label="W1 = "+str(0.0))
        
        plt.xlabel(r'$t$',fontsize=18)
        plt.ylabel(r'$\phi(t)$',fontsize=18)

        plt.ylim(1e-4, None)
        plt.xscale('log')
        plt.yscale('log')
        plt.tight_layout()
        plt.legend()
        plt.show()

    return t, phi

# plot MWD
def plotMWD(Zw, pd, w1, out='show'):
    
    """
        Plot the MWD and the contributions of the individual components
    """
    
    np.random.seed(1234)
    
    sigma  = np.sqrt(np.log(pd))
    mu     = np.log(Zw) - 1.5*np.log(pd)
    tiny   = 1e-6
    
    ### plot
    xmin = min(lognorm.ppf(tiny, s=sigma[0], scale=np.exp(mu[0])),
               lognorm.ppf(tiny, s=sigma[1], scale=np.exp(mu[1])))
                 
    xmax = max(lognorm.ppf(1-tiny, s=sigma[0], scale=np.exp(mu[0])),
               lognorm.ppf(1-tiny, s=sigma[1], scale=np.exp(mu[1])))
    
    x = np.linspace(xmin, xmax, 100)

    plt.axhline(y=0, c='gray', lw=1)
    plt.xlim(min(x), max(x))

    # plot PDF
    W1 = lognorm.pdf(x, s=sigma[0], scale=np.exp(mu[0]))
    W2 = lognorm.pdf(x, s=sigma[1], scale=np.exp(mu[1]))
    W = w1 * W1 + (1-w1) * W2
    
    plt.plot(x, w1*W1, lw=3, alpha=0.5)
    plt.plot(x, (1-w1)*W2, lw=3, alpha=0.5)
    plt.plot(x, W, '--', lw=3, c='k')

    #~ plt.xscale('log')
    plt.xlabel('$Z$')
    plt.ylabel('$w(Z)$')

    plt.legend()
    plt.tight_layout()
    
    if out == 'tiff' or out == 'tif':
        plt.savefig('pdfSample.tiff', dpi=600, fmt="tiff")
    elif out == 'pdf':
        plt.savefig('pdfSample.pdf')
    else:
        plt.show()
    return
    
#########################
# MAIN ROUTINE CALLERS  #
#########################

if __name__ == '__main__':
    
    #~ # Polydisperse LN distribution: polymer settings
    if True:
        
        Zw  = np.array([50.0, 20.0])
        pd  = np.array([1.1, 1.50])
        w1  = 0.5
        
        #lotMWD(Zw, pd, w1)
        #t, phi = getPhiPD(Zw, pd, w1, isPlot = True)
        # ~ np.savetxt("../relax.dat", np.c_[t, phi])
