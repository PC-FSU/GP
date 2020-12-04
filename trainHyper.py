#
# This is a python3 script which takes in a bidisperse blend: Z = [x, x], pd = [x, x], w1 = x
#  (1) reads in input data from TrainData/ [xtrain and h*.dat]
#  (2) using functions from commonGP.py, optimizes the hyper-parameters
#
#
# This python script takes in the following argument:
#  1. include_jacobian: If yes, Jacobian will be used in optimization process
#  2. nreplica: # of time optimization routine is ran 
#  3. isSave:  if yes, the hyperparam's will be saved in a file hyper.dat in path TrainData/
#  4. isDebug : if true, the decompose plot will be plotted for each param, default is false
#  
# Note: For this script to work on some data corresponding .dat file should be in  
#       present in TrainData/ Workspace. I do the following thing before running the script:
#   -->    python load_Wokspace.py --ClearWorkspace True
#   -->    python load_Workspace.py --load True --Folder (Folder name)
# Then run trainHyper.py
#   -->   python trainHyper.py --nreplica 10 --isSave False

from commonGP import *
import argparse
import time
import pandas as pd
#import cProfile
#from line_profiler import LineProfiler
# ~ np.set_printoptions(precision=2, suppress=True)

### objective function with jacobian ###########
def objParam(param, xtrain, sv, Z, dprint=False,method=0,index=0):
    """
       This is the cost function that has to be minimized by varying 7 kernel parameters
       param = (sig2,gamma1, gamma21, gamma22, gamma23, gamma24, gamma25)

       dprint = True returns datafit, complexity penalty, and obj
       can be turned on for debugging related printing
       
       This param choose that we include partial derivatives or not
       method =  0  { return (datafit, complexity penality, and obj) or (obj),depending upon dprint parameter} 
                 1  { return partial derivative of cost function w.r.t to paramater given by index} 

       index = if 0 {return partial derivative of cost function w.r.t sigma_2}        
               if 1 {return partial derivative of cost function w.r.t gamma_1}
               if 2 {return partial derivative of cost function w.r.t gamma_21}
               if 3 {return partial derivative of cost function w.r.t gamma_22}
               if 4 {return partial derivative of cost function w.r.t gamma_23}
               if 5 {return partial derivative of cost function w.r.t gamma_24}
               if 6 {return partial derivative of cost function w.r.t gamma_25}    
       
    """
    
    sig2   = param[0]
    gamma  = param[1:]

    N   = len(sv); n   = len(xtrain)

    R1, R2 = getR1R2(gamma, xtrain, sv)

    R1I, R2I, logDetR1, logDetR2 = getR1R2Inv(R1, R2, logDet=True)

    RIZ = invRZ(R1I, R2I, Z)

    # RI1 = invRv1(R1I, R2I)


    if(method == 0):

        # datafit
        df   = np.dot(Z.T, RIZ)
        df   = df.item()/sig2

        # complexity penalty
        cp   = N*n * np.log(sig2) + N * logDetR2 + n * logDetR1

        # together
        obj = df + cp

        if dprint:
            return df, cp, obj
            #return  N*n * np.log(sig2),N * logDetR2 + n * logDetR1,cp
        else:
            return obj  
    else:
        if(index == 0):
            obj = N*n/sig2 - np.dot(Z.T, RIZ)/np.square(sig2)
            return int(obj)
        if(index > 0):
            alpha = RIZ
            #start_time = time.time()
            inv_R = np.kron(R2I,R1I)
            first_term =  np.dot(alpha,alpha.T)/sig2 - inv_R
            partial_R  = partial_derivative_R1R2(gamma, xtrain, sv, index)
            if(index == 1):
                second_term = np.kron(R2,partial_R)
            else:
                second_term =np.kron(partial_R,R1)
            obj = -np.trace(np.matmul(first_term,second_term))
            #end_time   = time.time() - start_time
            #print(end_time,"\n")
            
            return obj
        
### Jacobian #############     
def jacobian(param,xtrain,sv,Zv):
    # This function can be pass in optimizer as jocobian
    param = np.array(param)
    jac   = np.zeros(xtrain.shape[1]+2)
    #start_time = time.time()
    for i in range(len(jac)):
        jac[i] = objParam(param,xtrain,sv,Zv, False,1,i)
    #end_time   = time.time() - start_time
    #print(end_time)
    return jac

# Nfeval is a global param which traces how many total call has been made for function objparam
Nfeval = 0
#the callback function for counting objparam call, it's passsed as argument in minimization routine
def callbackF(param0):
    global Nfeval
    #print(param0)
    Nfeval +=1



def getBestFit(xtrain, sv, Z, nreplica=10, include_jacobian = False):
    """This workhorse routine orchestrates the minimization of the objective function
       It returns the MLE parameters param (sig2, and gamma),likelihood,Average computational time, and length of xtrain"""
    from scipy.optimize import minimize
    
    bestobj   = 1e3
    bestPar   = np.zeros(7)
    
    tiny      = 1e-12  # want sigma^2 and gamma to be positive
    bnds      = ((tiny, None), (tiny, None), (tiny, None), (tiny, None),
                 (tiny, None), (tiny, None), (tiny, None))
    opt       = {'maxiter': 500}
    total_time = 0  # a placeholder for counting the total time spent by minimzation routine
    
    for ireplica in range(nreplica):

        param0    = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        param0    = param0 * np.random.uniform(0.01, 2.0, len(param0))  # randomize a little
        
        start_time = time.time()
        if include_jacobian:
            dprint,method = False,0
            res = minimize(objParam, param0, args=(xtrain, sv, Z,dprint,method),
                           jac = jacobian,bounds=bnds, method='TNC', options=opt)  #
        else:
            dprint,method = False,0
            res = minimize(objParam, param0, args=(xtrain, sv, Z,dprint,method),
                           bounds=bnds, method='TNC', options=opt,callback=callbackF)
        end_time   = time.time() - start_time
        
        param = res.x
        # ~ print(res)  # verbose

        if res.fun < bestobj:
            bestobj = res.fun
            bestPar = param.copy()
            print("ireplica  : ",ireplica)
        #To track computation time
        total_time += end_time

        if ireplica==0:
            #screen prinitng
            print("{:<25} {:<25} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12}".
                   format("Convergence","-LogLiklihood","sigma_2","gamma_1","gamma_21","gamma_22","gamma_23","gamma_24","gamma_25","Time"))
        
        # screen printing
        #print("\n{0}  {1:8.1f} ".format(res.success, res.fun), end="")
        #for x in np.r_[param,besttime]:
        #    print("{:7.2f}".format(x), end=" ")
        print("{:<25} {:<25.1f} {:<12.2f} {:<12.2f} {:<12.2f} {:<12.2f} {:<12.2f} {:<12.2f} {:<12.2f} {:<12.2f}".format(res.success, res.fun, param[0],param[1],param[2],param[3],param[4],param[5],param[6],end_time))    
                   
    print("\n")
    
    print("best Param =", bestPar)
    avg_time = total_time / nreplica
    #Average number of time optimize routine is ran
    global Nfeval
    avg_ran = Nfeval/nreplica #Nfeval is toal time function evaluation call is made (For example, in our case a single optimzation convergence evaluate objparam 30~40 times.  
    print(Nfeval)
    print("\nAverage time routine is ran :%f,and avg_time to ran a single routine:%f"%(avg_ran,avg_time/avg_ran))
    #print("Objec : ",objParam(bestPar,xtrain, sv, Z), " , ",bestobj)
    return np.r_[bestobj,bestPar,avg_time,len(xtrain)]


#### Utilities ####
def decomposePlot( ax=None, paramNum=3, SpanLogLim=[-2, 0.01], param = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])):
    
    # initialize
    #paramBase = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    
    # beta is special case, which can be negative
#    if paramNum == 0:
#        paramSpan = np.mean(Zv) + np.linspace(-10**SpanLogLim[1], 10**SpanLogLim[1])
#    else:
#        paramSpan = np.logspace(SpanLogLim[0], SpanLogLim[1], 20)

    paramSpan = np.logspace(SpanLogLim[0], SpanLogLim[1], 20)

    objSpan = np.zeros(len(paramSpan))
    cpSpan  = np.zeros(len(paramSpan))
    dfSpan  = np.zeros(len(paramSpan))


    # calculate
    for i, p in enumerate(paramSpan):
        param[paramNum] = p
        dfSpan[i], cpSpan[i], objSpan[i] = objParam(param, xtrain, sd, Zd, True)

    # plot
    xlab = [r'$\sigma^2$', r'$\gamma_1$', r'$\gamma_{21}$', r'$\gamma_{22}$', r'$\gamma_{23}$',r'$\gamma_{24}$', r'$\gamma_{25}$']
    
    
    ax.plot(paramSpan, dfSpan,label='df')#label="1st"
    ax.plot(paramSpan, cpSpan,label='cp')#label="2nd"
    ax.plot(paramSpan, objSpan, '--',label='obj')#label="cp"

    # beta has linear scale
    if paramNum > 0:
        ax.set_xscale('log')

    ax.set_ylabel('obj')
    ax.set_xlabel(xlab[paramNum])
    ax.legend()
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))    
    return ax

############# MAIN CODE ###########
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Optimization routine")
    parser.add_argument("--include_jacobian", type=bool, default=False,
                        help='Flag to include jacobian in hyperparam or not')
    parser.add_argument("--nreplica",   type=int, default=10,
                        help='Number of time optimization routine gonna run')
    parser.add_argument("--isSave", type = bool, default = False,
                        help = "Flag to save hyper param")
    parser.add_argument("--isDebug", type = bool, default = False,
                        help = "Flag to print  decompose plot")
    
    argspar  = parser.parse_args()
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')
    
    nreplica = argspar.nreplica
    include_jacobian = argspar.include_jacobian
    isSave   = argspar.isSave
    isDebug  = argspar.isDebug
    
    xtrain, sd, Zd, meanZd = readTrainData()
    
    print("\nTraining on %d Samples"%len(xtrain),"\n")
    param = getBestFit(xtrain, sd, Zd, nreplica, include_jacobian)
    if isSave:
        np.savetxt("TrainData/hyper.dat", param)
        
    if isDebug:
        #for decompose plot
        fig, axs = plt.subplots(2,4, figsize=(12, 7), facecolor='w', edgecolor='k')
        fig.subplots_adjust()
        axs = axs.ravel()

        for i in range(7):
            axs[i] = decomposePlot(ax=axs[i],paramNum=i,param=param)
            axs[i].plot()
        plt.tight_layout()
        plt.show()
