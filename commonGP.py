#
# This is a python3 script which takes in a bidisperse blend: Z = [x, x], pd = [x, x], w1 = x
#
#  (1) runs TDD and outputs phi(t)
#  (2) runs pyReSpect on fixed "s" grid and output H(s)
#
# Note: The version has two additional function, kernal_derivative and partial_derivative_R1R2, in case
#       if we want to work with jacobian.. Currently we are not using jacobian so these function don't matter.
#

import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['seaborn-white', 'myjournal'])
from matplotlib.offsetbox import AnchoredText
import os
import sys

# plotting setup; can comment all these lines out
#import seaborn as sns
#plt.style.use(['seaborn-ticks', 'myjournalFONT'])
#clr = [p['color'] for p in plt.rcParams['axes.prop_cycle']]

### I/O routines

def readTrainData():
    """"
        Read Training Data from File; 
        normalize MWs so that they are of order 1
    """
    
    # read data files
    xtrain = np.loadtxt(os.path.join("TrainData","xtrain.dat"))

    # divide MWs
    Zmax   = xtrain[:,0:2].max()
    xtrain[:, 0:2] = xtrain[:, 0:2]/Zmax

    n      = len(xtrain)
    
    # sd grid is a constant
    sd, hd = np.loadtxt(os.path.join("TrainData","h0.dat"), unpack=True)
    N      = len(sd)

    # initialize data vector
    Zd      = np.zeros(N*n)
    Zd[0:N] = hd 
    
    # read rest of data
    for i in range(1, n):
        sd, hd = np.loadtxt(os.path.join("TrainData","h{}.dat".format(i)), unpack=True)
        Zd[i*N:(i+1)*N] = hd            

    # making it mean zero    
    meanZd = np.mean(Zd)
    Zd = Zd - meanZd
        
    # column vector
    Zd = Zd.reshape(-1,1)
    Zd.shape #N*n

    return xtrain, sd, Zd, meanZd


#### Helper Routines ####

def kernelMatern(v, gamma):
    """Matern Correlation Kernel
       v is an array (of length m), returns m*m matrix"""
    absdist = np.absolute(v.reshape(-1,1) - v)
    rho     = np.sqrt(6.)/gamma * absdist
    return (1. + rho)*np.exp(-rho)

def kernelSE(v, gamma):
    """Squared-Exponential Kernel
       v is an array (of length m), returns m*m matrix"""    
    absdist = np.absolute(v.reshape(-1,1) - v)
    rho     = 0.5/gamma**2 * absdist**2
    return np.exp(-rho)
    
    
def kernal_derivative(v, gamma):
    """To compute dELTA(R)/dELTA(gamma) for kernelMatern.
       v is an array (of length m), returns m*m matrix"""
    
    absdist         = np.absolute(v.reshape(-1,1) - v)
    absdist_square  = np.square(absdist)
    rho             = np.sqrt(6)/gamma * absdist
    pre_factor      = (6/np.power(gamma,3))
    return pre_factor*np.exp(-rho)*absdist_square


def getR1R2(gamma, xtrain, sv):
    
    """    gamma1  = gamma[0]
           gamma21 = gamma[1] [MW]
           gamma22 = gamma[2] [MW]
           gamma23 = gamma[3] [pd]
           gamma24 = gamma[4] [pd]          
           gamma25 = gamma[5] [w1]
           
           EDIT: removing Z from arg list because was not used"""
           
    #R1  = kernelMatern(sv, gamma[0])
    #R1  = kernelMatern(np.log(sv), gamma[0])
    R1  = kernelSE(np.log(sv), gamma[0]) + 1e-8 * np.identity(len(sv))   # gamma1  = gamma[0], Regularizer
    # the product over x1, x2 and x3
    n   = len(xtrain)
    #R2  = np.ones((n,n))
    R2 = kernelMatern(xtrain[:,0], gamma[1])
    # dimension of input data (-1)
    for i in range(1,xtrain.shape[1]):
        R2  = R2 * kernelMatern(xtrain[:,i], gamma[i+1])
    
    #R2 = R2 + 1e-10 * np.identity(n) #Regularizer
    
    return R1, R2


def getR1R2Inv(R1, R2, logDet=True):
    """       
       returns with or without logDet. Can later improve this to use the function InvLogDet
    """    
    R1I = np.linalg.inv(R1); 
    R2I = np.linalg.inv(R2); 
    
    if logDet:    
        logDetR1 = np.linalg.slogdet(R1)[1]
        logDetR2 = np.linalg.slogdet(R2)[1]
    
        return R1I, R2I, logDetR1, logDetR2
    else:
        return R1I, R2I

def invRZ(R1I, R2I, Z):
    """Computes R^{-1}Z
       Takes in inverse of R1 (N*N) and R2 (n*n) matrices, and
       returns a Nn*1 vector inv(R)*Z
       Z does not literally have to be the datavector; could be r for instance"""

    N, _ = R1I.shape
    n, _ = R2I.shape
    
    Zstr = Z.reshape(N, n, order='F')     # N by n matrix
    return np.dot(np.dot(R1I, Zstr), R2I).reshape(N*n, 1, order='F')
    
    
def invRv1(R1I, R2I):
    """Takes in inverse of R1 (N*N) and R2 (n*n) matrices, and return a Nn*1 vector inv(R)*1"""
    v1 = np.sum(R1I, 1)
    v2 = np.sum(R2I, 1)
    return np.outer(v2, v1).reshape(-1,1)


def partial_derivative_R1R2(gamma, xtrain, sv, index = 0):
    
    """    
       gamma1  = gamma[0]    
       gamma21 = gamma[1]
       gamma22 = gamma[2]
       gamma23 = gamma[3] 
       gamma24 = gamma[4]
       gamma25 = gamma[5]

       index decides to return partial derivative of R_1 or R_2,

       if (index == 1) calculate del(R1)\del(gamma1); 
       if (index == 2) calculate del(R2)\del(gamma21);
       if (index == 3) calculate del(R2)\del(gamma22);
       if (index == 4) calculate del(R2)\del(gamma23);
       if (index == 5) calculate del(R2)\del(gamma24);
       if (index == 6) calculate del(R2)\del(gamma25);
    """
    if(index == 1 ):
        R1  = kernal_derivative(sv, gamma[0])      
        return R1 
    if(index > 1):
        n   = len(xtrain)
        R2  = np.ones((n,n))
        for i in range(xtrain.shape[1]):   
            if (i == index-2):
                R2 = R2* kernal_derivative(xtrain[:,i], gamma[i+1])
            else:
                R2  = R2 * kernelMatern(xtrain[:,i], gamma[i+1])
        return R2