#
# This is a python3 script which generates bidisperse blend: Z = [x, x], pd = [x, x], w1 = x by either uniform 
# or hypercubic sampling and then:
#               (1) runs TDD and outputs phi(t)
#               (2) runs pyReSpect on fixed "s" grid and output H(s)
#

#  This Script Generate Train Data (h.dat and xtrain.dat). I am going to include training data in Zip file so in general you don't have to run this script.
#    s
#   
#  This script takes argument:
#    1. Sample_Size : Number of sample tha you want to draw from hypercubic sampling for train data
#          Note: The generated Sample are then subjected to constrained offer by our problem, like M1>M2,
#                This lead to elimination of half of the genearted sample, so in reality we actual size is 
#                ~Sample_Size/2.
#    2. nw :  value for param w
#    3. npd : value for param pd
#    4. nz :  value for param Z
#    Note: The above three argument is used only in generating uniform sampling data not hypercubic sampling 
#    5. Param_Value: "value for (Z_min,Z_max,pdmin,pdmax), fixed for our case, but in case.
#    6. isPlot: Flag to plot the hypercubic or uniform test sample 
#    7. isSave: if True save the *.dat file in TrainData  
#
############ To Run ######################
#   python genTrain.py --Sample_Size 100 --isSave True/False --(Other_Flags as desired)
#

import os
import pyDOE as doe
import sys
import argparse
import shutil
import argparse
from matplotlib.offsetbox import AnchoredText

from scipy.interpolate import interp1d
from timeit import default_timer as timer

# relaxation spectra
import tdd
from contSpec import *
import time



def RunSingleCalc(x, itag):
    """
            Orchestrates calculations for a single sample
                    (1) runs TDD model to obtain (t, phi(t))
                    (2) runs pyReSpect to obtain  s, Hs
    """
    Zw = x[0:2]
    pd = x[2:4]
    w1 = x[4]

    print(itag, Zw, pd, w1)

    # run dynamics using TDD, gwet phi(t)
    t, phi = tdd.getPhiPD(Zw, pd, w1, isPlot = False)
    np.savetxt("relax.dat", np.c_[t, phi])

    # run pyReSpect
    times = 0.
    print('Running pyReSpect: ...', end="")
    start = timer()
    par = readInput('inpReSpect.dat')
    _, _ = getContSpec(par)
    times = timer() - start
    print("cpu {0:.2f} s".format(times))
    
    # Source path
    #At every iteration move the generated h.dat file to TrainData Workspace
    source = 'C://Users//18503//Dropbox//RA//Code//RA//PatchUp//PatchUp'
    source_file  = os.path.join(source,"h.dat")
    # Destination path  
    destination_file = os.path.join(source,"TrainData","h{}.dat".format(itag))

    # move and print times to screen
    # ~ tgt = str(Zw[0]) + '-' + str(Zw[1]) + '_' + "{:.2f}".format(pd[0]) +  '-' + \
          # ~ "{:.2f}".format(pd[1]) + '_' + "{:.2f}".format(w1)+'.dat'

    # ~ os.system("mv relax.dat TrainData/r{}.dat".format(itag))
    #os.system("mv h.dat TrainData/h{}s.dat".format(itag))
    shutil.move(source_file, destination_file)
    
    if not isSave:
        os.remove(destination_file)
    return


def genTrainSamples(xtrain):

    # run calculation, and save results in folder
    for itag, x in enumerate(xtrain):
        RunSingleCalc(x, itag)
        time.sleep(0.1)
        # To give some time to clear the overhead while generating the thousand's of file. When you run the           # program for a large sample size sometimes file opening and moving overhead don't get clear and it           # creates an error
        
    return

def uniformMeshSamples(nw, nZ, npd, Zmin=5., Zmax=50., pdmin=1.01, pdmax=1.50, isPlot=False, isSave = False):

    """creates samples on a uniform cubic mesh"""

    Z_  = np.linspace(Zmin, Zmax, nZ)
    pd_ = np.linspace(pdmin, pdmax, npd)
    w_  = np.linspace(0., 1., nw+1)

    w_ = w_[1:-1] # w1 = 0 not allowed, and w1 = 1 treated separately

    xfull = np.array(np.meshgrid(Z_, Z_, pd_, pd_, w_)).T.reshape(-1,5)     # cube of meshpoints
    xfull = xfull[xfull[:,1] < xfull[:,0]]                        # select only M1 > M2
    xpure = np.array(np.meshgrid(Z_, [0.], [pdmin, pdmax], [1.01], [1.])).T.reshape(-1,5) # add w1=1 samples

    # TDD has some problems if Z = 0; so making it same as component 1, but w1 = 1.0 anyway
    xpure[:,1] = xpure[:,0]


    xtrain = np.vstack((xfull, xpure))
    
    if isPlot:
        plt.subplots(1,3, figsize=(15,5))

        plt.subplot(131)
        plt.plot(xtrain[:,0], xtrain[:,1],'s', alpha=0.5)
        plt.axis('equal')
        plt.xlabel(r'$Z_1$')
        plt.ylabel(r'$Z_2$')
        plt.tight_layout()

        plt.subplot(132)
        plt.plot(xtrain[:,0], xtrain[:,4],'s', alpha=0.5)
        plt.xlabel(r'$Z_1$')
        plt.ylabel(r'$w_1$')

        plt.subplot(133)
        plt.plot(xtrain[:,2], xtrain[:,3],'s', alpha=0.5)
        plt.xlabel(r'$\rho_1$')
        plt.ylabel(r'$\rho_2$')

        plt.legend()
        plt.title("Uniform Samples")
        plt.tight_layout()
        plt.show()
    
    if isSave:
        np.savetxt("TrainData/xtrain.dat", xtrain, fmt="%6.2f")

    return xtrain


def hypercubic_sampling(nw, nZ, npd, Zmin=5., Zmax=50., pdmin=1.01, pdmax=1.50, isPlot=False, isSave = False, Sample_Size = 10, Seed = 0):
        
    np.random.seed(Seed)
    xfull = doe.lhs(5, samples=Sample_Size)
    
    xfull[:,0] = Zmin + (Zmax-Zmin)*xfull[:,0]     #scale m1 to have value between Zmin and Zmax
    xfull[:,1] = Zmin + (Zmax-Zmin)*xfull[:,1]     #scale m2 to have value between Zmin and Zmax
    
    xfull[:,2] = pdmin + (pdmax-pdmin)*xfull[:,2]  
    xfull[:,3] = pdmin + (pdmax-pdmin)*xfull[:,3]
    
    xfull = xfull[xfull[:,1] < xfull[:,0]]                 # constraint M1>M2
    xfull = xfull[xfull[:,4] > 0.0001]                     # w1>0
    
    Z_ = np.linspace(Zmin, Zmax, nZ)
    #Z_ = xfull[:,0]
    xpure = np.array(np.meshgrid(Z_, 0, [pdmin, pdmax], [1.01], [1.])).T.reshape(-1,5) # add w1=1 samples
    xpure[:,1] = xpure[:,0]
    xtrain = np.vstack((xfull, xpure))

    if isPlot:
        #plt.subplots(1,3,figsize=(9,2))
        fig,ax =  plt.subplots(1,3,figsize=(12,5))
        ax[0].plot(xtrain[:,0], xtrain[:,1],'s', alpha=0.5)
        ax[0].set_xlabel(r'$Z_1$')
        ax[0].set_ylabel(r'$Z_2$')
        anchored_text = AnchoredText("A", loc=2)
        ax[0].add_artist(anchored_text)
        ax[0].tick_params(axis='both', which='both', length=5) 
        #plt.ylim(1e-4,None)
        #plt.tight_layout()

        
        ax[1].plot(xtrain[:,0], xtrain[:,4],'s', alpha=0.5)
        ax[1].set_xlabel(r'$Z_1$')
        ax[1].set_ylabel(r'$w_1$')
        anchored_text = AnchoredText("B", loc=2)
        ax[1].add_artist(anchored_text)
        ax[1].tick_params(axis='both', which='both', length=5) 
        #plt.ylim(1e-4,None)

        ax[2].plot(xtrain[:,2], xtrain[:,3],'s', alpha=0.5)
        ax[2].set_xlabel(r'$\rho_1$')
        ax[2].set_ylabel(r'$\rho_2$')
        anchored_text = AnchoredText("C", loc=2)
        ax[2].add_artist(anchored_text)
        ax[2].tick_params(axis='both', which='both', length=5) 
        #plt.ylim(1e-4,None)

        plt.legend()
        #plt.suptitle("Hypercubic Samples")
        #plt.subplots_adjust(bottom=0.25,wspace=0.2)
        plt.tight_layout(pad=1.5)
        #plt.savefig("images//Hypercubic Samples.png",bbox_inches='tight', pad_inches=0.10)
        plt.show()
        
    
    if isSave:
        np.savetxt("TrainData/xtrain.dat", xtrain, fmt="%6.2f")
    
    return xtrain
    

#
# Create training data; Store results (spectra and input params) in TrainData/
#
if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate training data")
    parser.add_argument("--nw", type=int, default=2,
                        help='value for param w')
    parser.add_argument("--npd",   type=int, default=2,
                        help='value for param pd')
    parser.add_argument("--nZ", type=int, default=2,
                        help='')
    parser.add_argument("--isPlot", type=bool, default=False,
                        help='Flag to print the drawn sample')
    
    parser.add_argument("--Seed", type=int, default = 0,
                        help='seed use to generate hypercubic sampling')
    
    parser.add_argument("--isSave", type=bool, default=False,
                        help='Flag to save the generated data')
    
    parser.add_argument("--Sample_Size", type=int, default=20,
                        help='Number of sample tha you want to draw from hypercubic sampling')
    
    parser.add_argument("--Param_Value", nargs="+", default=[5, 50, 1.01, 1.50],
                       help = "value for (Z_min,Z_max,pdmin,pdmax)")
    
    argspar = parser.parse_args()
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    nw     = argspar.nw
    npd    = argspar.npd
    nZ     = argspar.nZ
    isPlot = argspar.isPlot
    isSave = argspar.isSave
    Seed   = argspar.Seed
    Sample_Size = argspar.Sample_Size
    Zmin,Zmax,pdmin,pdmax = argspar.Param_Value
    #xtrain = uniformMeshSamples(nw, nZ, npd, Zmin, Zmax, pdmin, pdmax, isPlot, isSave)
    #n      = len(xtrain)
    #print("Number of training samples\n", len(xtrain))
    #print(xtrain)
    
    
    
    print("\n Hypercubic samples \n")
    xtrain = hypercubic_sampling(nw, nZ, npd, Zmin, Zmax, pdmin, pdmax, isPlot, isSave, Sample_Size, Seed)
    n      = len(xtrain)
    print("Number of training samples\n", len(xtrain))
    print(xtrain)
    genTrainSamples(xtrain)
    
   # 
   # xtrain = np.array([[35.0, 20, 1.1, 1.3, 0.5],
   #                    [35.0, 20, 1.1, 1.3, 0.1],
   #                    [35.0, 20, 1.1, 1.3, 0.9]])
   # #xtrain[:,0:2] = xtrain[:,0:2]/50
   # print(xtrain)
   # genTrainSamples(xtrain)
   # 
   # 