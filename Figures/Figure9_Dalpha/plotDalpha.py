# This python script plot the Dalpha for Phi and H.
#
# This script takes in following argument:
# 1. Folder_list: List of train data folder used for running, for example, if you want to consider n = 50,100
#                 200 and 400 you need to pass Folder_list argument as ( --Folder_list 50 100 200 400).
#
# 2. isSave    :  If true the result of Dalpha analysis will be overwritten in the exisiting .txt file. 
#
# To run the script: Normal example:-
#         python plotDalpha.py --Folder_list 50 100 200 --isSave True/False
# 
# To reproduce the result:
#         python plotDalpha.py        #Default Folder_list is set to n = [100,800,3200] (reported in paper)
#                                     #This will only plot the result from the existing .txt file
#                                     #Read instruction in __main__ section.

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

# Utility function 2
def del_file():
    try:
        os.remove("relax.dat")
        os.remove("h.dat")
        os.remove("inpReSpect.dat")
        os.rmdir("output")
    except:
        print("")
    
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
            
    try:
        np.savetxt(r"relax.dat", np.c_[t, phi])
    except:
        while not os.path.exists(r"relax.dat"):
            time.sleep(1)
        np.savetxt(r"relax.dat", np.c_[t, phi])
        
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
    
def Plot(Eta_h = None,Eta_phi = None, From_txt=False,SavePlot=False):

        
    """
    ETA_h    = Eta_alpha result for h.  Note: Eta should be Xi greek symbol, but to avoid confusion i used eta
    ETA_Phi  = Eta_alpha result for Phi.
    From_txt = If true load value of above 4 argument from existing .txt file.
    SavePlot = If true, save the figure.
    
    """
    
    if From_txt == True:
        #Load data
        #Load Eta for H
        Eta_h = np.loadtxt("Eta_h.txt")
        #Load Eta for Phi
        Eta_phi = np.loadtxt("Eta_phi.txt")
        print(Eta_h.shape,Eta_phi.shape) 
        
    try:
        n = Eta_h[:,0]  #The first column of all .txt file contain n
        Eta_h = Eta_h[:,1:]
        Eta_phi = Eta_phi[:,1:]
    except:
        n = np.array([Eta_h[0]])
        Eta_h = Eta_h[1:].reshape(1,-1)
        Eta_phi = Eta_phi[1:].reshape(1,-1)     
        print(Eta_h.shape,Eta_phi.shape) 
        
    #Define Alpha
    alpha = np.linspace(0.01,0.99,50)
    Eta_alpha  = np.sqrt(2)*erfinv(2*alpha-1)
    
    #Placeholder for DAlpha
    Dalpha  = np.zeros(shape=(2,len(n),len(alpha))) 
    #Shape => (#2 for H,phi; #n for train_set; #alpha we are checking)
    print(Dalpha.shape)
    #Loop over different n,i.e Training dataset
    for i in range(len(n)):   
        #Loop over different alpha value
        for index,element in enumerate(Eta_alpha):
            # For h
            Dalpha[0][i][index] = np.mean(Eta_h[i] <= element)
            # For Phi
            Dalpha[1][i][index] = np.mean(Eta_phi[i] <= element)
            
    
    #************plotting rouitne****************************
    fig, axs = plt.subplots(1,1, figsize=(10, 5), facecolor='w', edgecolor='k')
    #axs = axs.ravel()
    
    for i in range(len(n)):
        #For H
        #axs[0].scatter(alpha,Dalpha[0][i],label = r"$n = $"+" "+str(n[i]))
        axs.scatter(alpha,Dalpha[0][i],label = r"$n = $"+" "+str(n[i]))
        #For Phi
        #axs[1].scatter(alpha,Dalpha[1][i],label = r"$n = $"+" "+str(n[i]))
        
    axs.plot(alpha,alpha,linestyle='--',color = 'k')
    axs.set_xlabel(r"$\alpha$")
    axs.set_ylabel(r"$(D_{\alpha},h(s))$")
    axs.legend()
    
    #axs[1].plot(alpha,alpha,linestyle='--',color = 'k')
    #axs[1].set_xlabel(r"$\alpha$")
    #axs[1].set_ylabel(r"$(D_{\alpha},\phi(t))$")
    #axs[1].legend()
    
    if SavePlot == True:
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
    n = []
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
            n.append(len(xtrain)) # used to plot n vs error
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
    
    
    Eta_h    =   np.c_[np.array(n).T, eta[0].reshape(len(Folder_list),N*len(Test_data))]
    Eta_phi  =   np.c_[np.array(n).T, eta[1].reshape(len(Folder_list),N*len(Test_data))]
    del_file() 
    
    if isSave == True:
        np.savetxt("Eta_h.txt",   Eta_h)
        np.savetxt("Eta_phi.txt", Eta_phi)
        
    return Eta_h,Eta_phi


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Plot Dalpha vs alpha, for different n")
                     
    parser.add_argument("--Folder_list", type=int, nargs='+', default=[100,800,3200],
               help = "List of training folder")
    
    parser.add_argument("--isSave", type=bool,default=False,
           help = "If true, save the analysis result in .txt files, and save the plot")
    
    parser.add_argument("--RunAnalysis", type=bool, default=False,
           help = "If true, The Dalpha vs Alpha will be caluclated for passed Folder_list argument and result will be plotted. If False, the result will be plotted from the exisitng .txt files (Note: This result will be plotted for the Folder_list argument used to create the saved .txt file, not the current Folder_list argument.")
    
    
    argspar = parser.parse_args()
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')
    
    Folder_list = argspar.Folder_list
    isSave      = argspar.isSave
    RunAnalysis = argspar.RunAnalysis

    if RunAnalysis == True:
        Eta_h,Eta_phi = run_analysis(Folder_list,isSave)
        Plot(Eta_h,Eta_phi,From_txt=False,SavePlot = isSave)
    else:
        Plot(From_txt = True, SavePlot = isSave)