#
# This python script plot the RMSE and Median Absolute deviation between mean prediction and actual result.
#
# This script takes in following argument:
# 1. Folder_list: List of train data folder used for running, for example, if you want to consider n = 50,100
#                 200 and 400 you need to pass Folder_list argument as ( --Folder_list 50 100 200 400).
#
# 2. isSave    :  If true the result of RMSE and MAD analysis will be overwritten in the exisiting .txt file. 
#
# To run the script: Normal example:-
#         python plotRMSE.py --Folder_list 50 100 200 --isSave True/False
# 
# To reproduce the result:
#         python plotRMSE.py          #Default Folder_list is set to n = [50,,,,3200] (reported in paper)
#                                     #This will only plot the result from the existing .txt file
#                                     #Read instruction in __main__ section.
#
#  Note: All the loading and cleaning of workspace is automated in this script.
#

import matplotlib.pyplot as plt
plt.style.use(['seaborn-white', 'myjournal'])
from matplotlib.offsetbox import AnchoredText

# The following command force matplotlib to use asmath package. Needed to plot the labels.
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command

#This routine is to call file in higher level directory from file in subdirectory. Example The path of file tdd.py is  "~//Sachin//" but since we are calling it from a file located in lower level subdirectory "~//Sachin//Figures//Figure_2_spectra_illustration" we need to add this PYTHONPATH routine.
#**************************
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..\\..'))

#**************************
import glob 
import subprocess
import time
import seaborn as sns
import pandas as pd
from predictGP import *
from scipy.stats import median_abs_deviation
from scipy.special import erf,erfinv
import shutil


# Utility function 1

def move_file(Parent_dir):
    #Move some file for plotPredTrain2 to work properly
    # Source path 
    source = os.path.join(Parent_dir,"inpReSpect.dat")
    # Destination path 
    destination = os.path.join(os.getcwd(),"inpReSpect.dat")
    # Copy the content of source to destination 
    dest = shutil.copy(source, destination) 
    return None

# Utility function 3
def del_file():
    try:
        os.remove("relax.dat")
        os.remove("h.dat")
        os.remove("inpReSpect.dat")
        os.rmdir("output")
    except:
        print("Directory doesn't have .dat file")
    
    return None

# Utility function 3

def plotPredTrain2(hp, dhp, sv, xp, meanZ, getPhifromH=False, plot=False):
    #
    # getPhiFromH: if true, the test error will also run for phi vs T
    # 
    N = len(sv)
    hp = hp + meanZ
    # run true dynamics using TDD and get spectrum: Note unnormalize the MWs
    Zmax = 50.0
    t, phi = tdd.getPhiPD(Zmax*xp[0:2], xp[2:4], xp[4], isPlot = False)
    
    np.savetxt(r"relax.dat", np.c_[t, phi])
    while not os.path.exists(r"relax.dat"):
        time.sleep(1)
    par  = spec.readInput('inpReSpect.dat')
    H, _ = spec.getContSpec(par)
    h_true = np.exp(H)
    
    if getPhifromH:
        #*********************for phi vs t******************************        
        #call GT function which gives back phi and t from H and S
        #Predicted Phi
        t,phi_predicted,dphi_predicted = Gt(sv, hp, dhp)
        #get back true phi
        _,phi_true,dphi_true = Gt(sv, h_true, dhp) 
        #**************************************************************
        if plot:      
            plt.plot(t, phi_predicted, linewidth=4, label='est')
            plt.fill_between(t, phi_predicted - 2.5*dphi_predicted, hp + 2.5*dphi_predicted, alpha=0.1)
            plt.plot(t, phi_true, 'gray',linewidth=4, alpha=0.5, label='true')   
            #plt.plot(sv, abs(h_true-hp),label="Prediction_error")
            
            plt.xscale('log')
            plt.xlabel('$t$',fontsize=22)
            plt.ylabel('$\Phi$',fontsize=22)
            plt.xticks(fontsize= 14)
            plt.yticks(fontsize= 14)

            plt.legend()
            plt.tight_layout()
            plt.show()    
            
    else:   
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
    
    #Choose what to return
    if getPhifromH:
        return h_true-hp, dhp, phi_true-phi_predicted, dphi_predicted
    else:
        return h_true-hp, dhp, None, None

# Ploting routine, make change here to edit the figure appearance
    
def Plot():
    
    #Hardcoding n here, and using the actual value not the folder name since they are going to use in plot labels
    n = [53,402,1606]
    #Load data
    #Load Eta for H
    Eta_h = np.loadtxt("Eta_h.txt")
    #Load Eta for Phi
    Eta_phi = np.loadtxt("Eta_phi.txt")
    
    #Define Alpha
    alpha = np.linspace(0.01,0.99,50)
    Eta_alpha  = np.sqrt(2)*erfinv(2*alpha-1)
    
    #Placeholder for DAlpha
    Dalpha  = np.zeros(shape=(2,len(n),len(alpha))) 
    #Shape => (#2 for H,phi; #n for train_set; #alpha we are checking)
    
    #Loop over different n,i.e Training dataset
    for i in range(len(n)):   
        #Loop over different alpha value
        plt.hist(Eta_h[i])
        plt.show()
        plt.hist(Eta_phi[i])
        plt.show()
        
        for index,element in enumerate(alpha):
            # For h
            Dalpha[0][i][index] = np.mean(Eta_h[i] <= element)
            # For Phi
            Dalpha[1][i][index] = np.mean(Eta_phi[i] <= element)
            
    
    #************plotting rouitne****************************
    fig, axs = plt.subplots(1,2, figsize=(14, 6), facecolor='w', edgecolor='k')
    axs = axs.ravel()
   
    for i in range(len(n)):
        #For H
        axs[0].scatter(alpha,Dalpha[0][i],label = r"$n = $"+" "+str(n[i]))
        #For Phi
        axs[1].scatter(alpha,Dalpha[1][i],label = r"$n = $"+" "+str(n[i]))
        
    #axs[0].plot(alpha[10:],alpha[10:],linestyle='--',label="")
    axs[0].set_xlabel(r"$\alpha$")
    axs[0].set_ylabel(r"$(D_{\alpha},h(s))$")
    axs[0].legend()
    
    #axs[1].plot(alpha[10:],alpha[10:],linestyle='--',)
    axs[1].set_xlabel(r"$\alpha$")
    axs[1].set_ylabel(r"$(D_{\alpha},\phi(t))$")
    axs[1].legend()
    
    plt.savefig("Dalpha.pdf",bbox_inches='tight', pad_inches=0.25)
    plt.show()

    return None
    
#Routine to run/save analysis.
    
def run_analysis(Folder_list,isSave):
    
    Parent_dir  = "C:\\Users\\18503\\Dropbox\\RA\\Code\\RA\\PatchUp\\PatchUp\\Sachin"
    Test_data   = np.loadtxt(os.path.join(os.path.join(Parent_dir,"TestData","test.dat")))
    
    #Hardcoding N here
    N = 100
    eta  = np.zeros(shape=(2,len(Folder_list),N*len(Test_data)))  #placeholder for etas,In shape argument 2 is for =>(For H, For Phi)

    move_file(Parent_dir)
    #Iterate over each training set
    for F_idx,folder in enumerate(Folder_list):
            #Clean Workspace
            print("Routine running for Folder {0} \n".format(folder))
            cmd_1 = "python Load_workspace.py --ClearWorkspace True"
            out_str = subprocess.check_output(cmd_1, shell=True,cwd = Parent_dir)
            print(out_str)
            time.sleep(2.5)
            
            #Load workspace 
            cmd_2 = "python Load_workspace.py --load True --Folder {0}".format(folder)
            out_str = subprocess.check_output(cmd_2, shell=True,cwd = Parent_dir)
            print(out_str)
            time.sleep(2.5)
            
            #Load the training data and hyperparam
            xtrain, sd, Zd, meanZd = readTrainData(Parent_dir)
            param = np.loadtxt(os.path.join(Parent_dir,"TrainData","hyper.dat"))[1:-2]

            for index,test_point in enumerate(Test_data):
                
                     print("Routine Running for Folder,TestPoint# ",folder," ",index,"\n")
                    
                     hp, dhp = predict(test_point, sd, xtrain, Zd, param)
                     Dhp,Sigma_h,DPhi,Sigma_Phi = plotPredTrain2(hp, dhp, sd, \
                                                                       test_point ,meanZd,True,False)  
                     #*****Eta Calculation***********
                     #For h
                     eta[0][F_idx][index*N:(index+1)*N] =   Dhp/Sigma_h
                     #For Phi
                     eta[1][F_idx][index*N:(index+1)*N] =   DPhi/Sigma_Phi
                     time.sleep(0.1)

    if isSave:
        np.savetxt("Eta_h.txt", eta[0])
        np.savetxt("Eta_phi.txt", eta[1])
    del_file()    
    return None


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Plot RMSE and MAD")
                     
    parser.add_argument("--Folder_list", type=int, nargs='+', default=[100,800,3200],
               help = "List of training folder")
    parser.add_argument("--isSave", type=bool,default=False,
           help = "If true, save the analysis result in .txt files")
    
    argspar = parser.parse_args()
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')
    
    Folder_list = argspar.Folder_list
    isSave      = argspar.isSave
    
    #Uncomment line below if you want to run the analysis
    #run_analysis(Folder_list,isSave)
    Plot()
    