# This script plot optimize param (SIGMA,GAMMA,TIME, LIKLIHOOD )for different length of taraining data,n
#
# The script takes in argument:
# 1. Folder_list: Pass the different subfolder name for plotting the corresponding params. For example if you 
#                 pass 50 100 200, param will be plotted for these three n values
# To run:
#        python plotParam.py --Folder_list 50 100 200 400

import matplotlib.pyplot as plt
plt.style.use(['seaborn-white', 'myjournal'])
from matplotlib.offsetbox import AnchoredText

import argparse
import numpy as np
import os

### MAIN ROUTINES ###

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Plot param for different n")
    parser.add_argument("--Folder_list", type=int, nargs='+', default=[50, 100, 200, 400],
                   help = "List of folder for which you want to plot hyper-params")
    
    argspar = parser.parse_args()
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')
    
    Folder_list = argspar.Folder_list
    
    All_Param  = np.zeros(shape=(len(Folder_list),10))
    #Placeholder that will be used for plotting, where each row correspond to param of one n, and we have 10 
    # param ==> bestobj,bestPar( 6 params) ,avg_time,len(xtrain).
    

    #Read the param
    for index,value in enumerate(Folder_list):
        All_Param[index] = np.loadtxt(os.path.join("TrainData",str(value),"hyper.dat"))
    
    fig, axs = plt.subplots(2,4, figsize=(10, 5), facecolor='w', edgecolor='k')
    fig.subplots_adjust( hspace=0.2 )
    axs = axs.ravel()
    xlab = [r"$-\frac{LogLiklihood}{n}$",r'$\sigma^2$', r'$\gamma_1$', r'$\gamma_{21}$', r'$\gamma_{22}$',
           r'$\gamma_{23}$',r'$\gamma_{24}$',r'$\gamma_{25}$',r"$Time$"]
    
    #All_Param[:,0] = All_Param[:,0]/All_Param[:,-1]

    
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
    #plt.savefig("images//Param.png",bbox_inches='tight', pad_inches=0.25)
    plt.show()     
    
    #I haven;t included the cide to plot the time vs n, but it can be copy from the commented code down below
    
    
    
    
    
    
    
    
    #    for i in range(8):
#        axs[i].scatter(All_Param[:,-1],All_Param[:,i],alpha = 1)
#        axs[i].set_xlabel('$n$')
#        axs[i].set_ylabel(xlab[i])
#        #axs[i].set_title(xlab[i] + " vs n")
#        #axs[i].legend()#loc = "lower left",prop={'size': 10}
#    
#    plt.tight_layout()
#    plt.show()
#    
#    fig, ax = plt.subplots(figsize=(5, 4))
#    ax.loglog(All_Param[:,-1],All_Param[:,8],'-o',markersize=10)
#    #ax.loglog(All_Param[:,-1],np.exp(3*np.log(All_Param[:,-1])))
#    #temp = All_Param[:,8][1:]/All_Param[:,8][0]
#    print(All_Param[:,-1],All_Param[:,8])
#    
#    ax.set_xlabel('$n$',fontsize=18)
#    ax.set_ylabel(xlab[-1],fontsize=18)
#    ax.tick_params(axis="x", labelsize=16)
#    ax.tick_params(axis="y", labelsize=16)
#    #plt.legend(loc = "lower left",prop={'size': 10})
#    #ax.set_title(xlab[-1] + " vs n")
#    plt.tight_layout()
#    plt.savefig("images//Time vs n.png")
#    plt.show()

    
    
#    for i in range(8):
#        fig, ax = plt.subplots(figsize=(5, 4))
#        plt.scatter(All_Param[:,-1],All_Param[:,i],alpha = 1,s=60)
#        plt.xlabel('$n$',fontsize=18)
#        plt.ylabel(xlab[i],fontsize=18)
#        plt.xticks(fontsize= 16)
#        plt.yticks(fontsize= 16)
#        plt.xscale('log')
#        #plt.xticks([0,200,400,600,800,1000])
#        plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
#        #plt.title(xlab[i] + " vs n")
#        plt.tight_layout()
#        plt.savefig("images//"+str(i)+" vs n.png")
#        plt.show()
#     
        #plt.clf()
    
    
    
    
    
#    fig, ax = plt.subplots(1,2,figsize=(10, 4))
#    ax[0].scatter(All_Param[:,-1],All_Param[:,0],alpha = 1,s=60)
#    ax[0].set_xscale('log')
#    #ax[0].set_yscale('log')
#    ax[0].set_xlabel('$n$')
#    ax[0].set_ylabel(xlab[0])
#    #ax[0].set_ylim(1e-4, 1)
#    anchored_text = AnchoredText("A", loc=3)
#    ax[0].add_artist(anchored_text)
#    
#  
#    ax[1].plot(All_Param[:,-1],All_Param[:,8],"o-",alpha = 1)
#    ax[1].set_xscale('log')
#    ax[1].set_yscale('log')
#    ax[1].set_xlabel('$n$')
#    ax[1].set_ylabel(str(xlab[-1]))
#    #ax[1].set_ylim(1e-4, 1e2)
#    anchored_text = AnchoredText("B", loc=3)
#    ax[1].add_artist(anchored_text)
#    #ax[1].legend(loc=4,frameon = True)
#    
#    fig.tight_layout(pad=2.0)
#    #plt.savefig("images//TimeAndLiklihood.png",bbox_inches='tight', pad_inches=0.25)
#    plt.show()   
 