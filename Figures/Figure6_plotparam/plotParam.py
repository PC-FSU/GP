# This script plot optimize param (SIGMA,GAMMA,TIME, LIKLIHOOD )for different length of taraining data,n
#
# The script takes in argument:
# 1. Folder_list: Pass the different subfolder name for plotting the corresponding params. For example if you 
#                 pass 50 100 200, param will be plotted for these three n values
# To run:
#        python plotParam.py --Folder_list 50 100 200 400


import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
plt.style.use(['seaborn-white', 'myjournal'])
from matplotlib.offsetbox import AnchoredText

# The following command force matplotlib to use asmath package. Needed to plot the label ax[1] defined below in the code.
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command

#This routine is to call file in higher level directory from file in subdirectory. Example The path of file tdd.py is  "~//Sachin//" but since we are calling it from a file located in lower level subdirectory "~//Sachin//Figures//Figure_2_spectra_illustration" we need to add this PYTHONPATH routine.
#**************************
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..\\..'))


### MAIN ROUTINES ###

if __name__ == "__main__":
    # ************Parse arguments********************
    parser = argparse.ArgumentParser(description="Plot param for different n")
    parser.add_argument("--Folder_list", type=int, nargs='+', default=[50, 100, 200, 400],
                   help = "List of folder for which you want to plot hyper-params")
    
    argspar = parser.parse_args()
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')
    
    Folder_list = argspar.Folder_list
    # *********************************************
    #Read the data
    #Placeholder that will be used for plotting, where each row correspond to param of one n, 
    # and we have total of 10 param [bestobj,bestPar(7 params),avg_time,len(xtrain)]
    All_Param  = np.zeros(shape=(len(Folder_list),10))
    Parent_dir  = "C:\\Users\\18503\\Dropbox\\RA\\Code\\RA\\PatchUp\\PatchUp\\Sachin"
    #Read the param
    for index,value in enumerate(Folder_list):
        All_Param[index] = np.loadtxt(os.path.join(Parent_dir,"TrainData",str(value),"hyper.dat"))
        
    xlab = [r"$-\frac{LogLiklihood}{n}$",r'$\sigma^2$', r'$\gamma_1$', r'$\gamma_{21}$', r'$\gamma_{22}$',
           r'$\gamma_{23}$',r'$\gamma_{24}$',r'$\gamma_{25}$',r"$Time$"]
    
    # *********************************************
    #Ploting routine
    fig, ax = plt.subplots(2,2,figsize=(10, 8))
    ax = ax.ravel()
    
    #plot sigma_2
    ax[0].scatter(All_Param[:,-1],All_Param[:,1],alpha = 1,s=60)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel('$n$')
    ax[0].set_ylabel(xlab[1])
    ax[0].set_ylim(1e-4, 1)
    anchored_text = AnchoredText("A", loc=3)
    ax[0].add_artist(anchored_text)
    ax[0].tick_params(axis='both', which='both', length=5)
    
    #plot gamma_1 and gamma_25
    ax[1].scatter(All_Param[:,-1],All_Param[:,2],alpha = 1,s=60,label = xlab[2])
    ax[1].scatter(All_Param[:,-1],All_Param[:,7],alpha = 1,s=60,label = xlab[7])
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel('$n$')
    ax[1].set_ylabel(str(xlab[2])+","+str(xlab[7]))
    ax[1].set_ylim(1e-4, 1e2)
    anchored_text = AnchoredText("B", loc=3)
    ax[1].add_artist(anchored_text)
    ax[1].legend(loc=4,frameon = True)
    ax[1].tick_params(axis='both', which='both', length=5)

    #plot gamma_21 and gamma_22
    ax[2].scatter(All_Param[:,-1],All_Param[:,3],alpha = 1,s=60,label = xlab[3])
    ax[2].scatter(All_Param[:,-1],All_Param[:,4],alpha = 1,s=60,label = xlab[4])
    ax[2].set_xscale('log')
    ax[2].set_yscale('log')
    ax[2].set_xlabel('$n$')
    ax[2].set_ylabel(str(xlab[3])+","+str(xlab[4]))    
    ax[2].set_ylim(1e-3, 1e2)
    anchored_text = AnchoredText("C", loc=3)
    ax[2].add_artist(anchored_text)
    ax[2].legend(loc=4,frameon = True)
    ax[2].tick_params(axis='both', which='both', length=5)
    
    #plot gamma_23 and gamma_24
    ax[3].scatter(All_Param[:,-1],All_Param[:,5],alpha = 1,s=60,label = xlab[5])
    ax[3].scatter(All_Param[:,-1],All_Param[:,6],alpha = 1,s=60,label = xlab[6])    
    ax[3].set_xscale('log')
    ax[3].set_yscale('log')
    ax[3].set_xlabel('$n$')
    ax[3].set_ylabel(str(xlab[5])+","+str(xlab[6]))       
    ax[3].set_ylim(1e-3, 1e2)
    anchored_text = AnchoredText("D", loc=3)
    ax[3].add_artist(anchored_text)
    ax[3].legend(loc=4,frameon = True)
    ax[3].tick_params(axis='both', which='both', length=5)    
    
    
    #fig.subplots_adjust(wspace=0.5, hspace=0)
    fig.tight_layout(pad=2.0)
    plt.savefig("Param.pdf",bbox_inches='tight', pad_inches=0.25)
    plt.show()     
