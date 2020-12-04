#     This Python Script mainly contain the routine required to plot the prediction. The Differene betwenn 
#     this and original version is 1) i have included a functon that genratws phi,t back from h,s. 2)  
#     modification in function plotPredTrian,(Basically now it has a routine to check for whether  
#     original data lie within the range (prediction + 2.5*SD,prediction - 2.5*SD)  

from commonGP import *
import tdd
import contSpec as spec
import argparse
import os
from line_profiler import LineProfiler

def get_rvOptim(x, xtrain, param, R1, ind_f):
    """"
        This furnishes the 'r' vector; Matern is hardcoded into local R2 definition
    """
    #
    # Furnish internal R2: this is different from the 
    # original ones defined only on the training data
    # Using previously computer R1 here; ind_f indexes the requested value of Phi

    
    N, _   = R1.shape
    n      = len(xtrain)
    gamma2 = param[2:]
    
    # internally, R2 is a n*1 vector
    # the product over Z1 (i=0), Z2 (i=1), pd1 (i=2), pd2(i=3), and w1 (i=4)
        
    R2 = np.ones(n)
    for i in range(xtrain.shape[1]):
        rho = np.absolute(x[i] - xtrain[:,i]) * np.sqrt(6.)/gamma2[i]
        R2  = R2 * (1. + rho)*np.exp(-rho)   # Matern is hard-coded in the prediction
    
    # the crucial r vector; block of N at a time
    rv     = np.zeros(N*n)        
    for i in range(n):
        rv[i*N:(i+1)*N] = R1[:,ind_f] * R2[i]
        
    return rv.reshape(-1,1)

def predict(x, sv, xtrain, Zv, param):

    N      = len(sv)
    n      = len(xtrain)

    sig2   = param[0]
    gamma  = param[1:]    
    
    R1, R2   = getR1R2(gamma, xtrain, sv)
    R1I, R2I = getR1R2Inv(R1, R2, logDet=False)

    RIZ      = invRZ(R1I, R2I, Zv)
    
    h_pred    = np.zeros(N)
    dh_pred   = np.zeros(N)

    # actual prediction engine
    for i in range(len(sv)):

        rv    = get_rvOptim(x, xtrain, param, R1, i)
        invRr = invRZ(R1I, R2I, rv)

        h_pred[i]  = np.dot(rv.T, RIZ)
        dh_pred[i] = sig2 * (1. - np.dot(rv.T, invRr))
        
    return h_pred, np.sqrt(np.abs(dh_pred))





def plotPredTrain(hp, dhp, sv, xp, meanZ,plot=False):
    
    N = len(sv)
    
    hp = hp + meanZ

    # run true dynamics using TDD and get spectrum: Note unnormalize the MWs
    Zmax = 50.0
    t, phi = tdd.getPhiPD(Zmax*xp[0:2], xp[2:4], xp[4], isPlot = False)
    np.savetxt("relax.dat", np.c_[t, phi])
    par  = spec.readInput('inpReSpect.dat')    
    H, _ = spec.getContSpec(par)
    h_true = np.exp(H)
    
    
    #*****The code below calculate results that are used for test error analysis*****************
    
    #Error b/w true and prediction
    error = np.sum(np.abs(h_true-hp))/len(h_true)
    
    # Create a dict placeholder to store ==> whether prediction is in range, fraction of partical out and
    # which points are out
    Prediction_InRange = {}
    IsUp_out   = hp + 2.5*dhp >= h_true
    IsDown_out = h_true >= hp - 2.5*dhp
    Prediction_InRange['isInRange']   = np.all([np.all(IsUp_out),np.all(IsDown_out)])
    Union_upout_and_downout = np.r_[IsUp_out,IsDown_out]
    Value , Counts = np.unique(Union_upout_and_downout,return_counts=True)
    if len(Value)>1:
        Prediction_InRange['OutRange_fraction'] = Counts[0]/(Counts[0]+Counts[1])
    else:
        Prediction_InRange['OutRange_fraction'] = 0
    Prediction_InRange['Which_Out'] = [idx for idx in range(len(Union_upout_and_downout)) if Union_upout_and_downout[idx] == False]
    
    
    #***************************************************************************************#
    if plot:
        plt.plot(sv, hp, linewidth=4, label='est')
        plt.fill_between(sv, hp - 2.5*dhp, hp + 2.5*dhp, alpha=0.1)

        plt.plot(sv, h_true, 'gray',linewidth=4, alpha=0.5, label='true')   
        #plt.plot(sv, abs(h_true-hp),label="Prediction_error")
        plt.xscale('log')
        plt.xlabel('$s$',fontsize=22)
        plt.ylabel('$h$',fontsize=22)
        plt.xticks(fontsize= 14)
        plt.yticks(fontsize= 14)

        plt.legend()
        plt.tight_layout()
        plt.show()
        #plt.plot(np.abs((h_true-hp)))
        #plt.plot(np.exp(h_true-hp)/np.exp(h_true))
        #plt.show()
    
    #print(error)
    #print(Prediction_InRange)
    return error,hp,h_true,Prediction_InRange


def Gt(s, hs, dhp):
    
    smin = min(s)
    smax = max(s)
    ns   = len(s)

    tmin = smin * np.exp(+np.pi/2)
    tmax = smax / np.exp(+np.pi/2)
    t    = np.geomspace(tmin, tmax, ns)

    hsv         = np.zeros(ns);
    hsv[0]      = 0.5 * np.log(s[1]/s[0])
    hsv[ns-1]   = 0.5 * np.log(s[ns-1]/s[ns-2])
    hsv[1:ns-1] = 0.5 * (np.log(s[2:ns]) - np.log(s[0:ns-2]))
    S, T        = np.meshgrid(s, t);

    Gt   =  np.dot(np.exp(-T/S) * hsv, hs)
    
    dphi =  np.dot(np.exp(-T/S) * hsv, dhp)

    return t, Gt,dphi

xtrain, sd, Zd, meanZd = readTrainData()
param = np.loadtxt("TrainData/hyper.dat")[1:-2]

### MAIN ROUTINES ###
if __name__ == "__main__":

    #xp = np.array([43.0, 18, 1.15, 1.34, 0.38])
    #xp[0:2] = xp[0:2]/50.0
    
    #xp = np.array([0.49, 0.01, 1.08, 1.06, 0.41])
    #xp = np.array([0.73, 0.03, 1.46, 1.21, 0.1])
    #xp = np.array([0.46, 0.33, 1.39, 1.15, 0.62])
    xp = np.array([0.81,   0.29,   1.13,   1.39,   0.84])
    
    #xp = xtrain[0]
    #print("xp =", 50*xp[0:2], xp[2:])
    # ~ xp = xtrain[2]
    hp, dhp = predict(xp, sd, xtrain, Zd, param)
    plotPredTrain(hp, dhp, sd, xp, meanZd,True)
    
    #lp = LineProfiler()
    #lp_wrapper = lp(plotPredTrain2)
    #lp_wrapper(xp)
    #lp.print_stats()