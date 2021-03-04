# This python script plot the RMSE and Median Absolute deviation between mean prediction and actual result.
#
# This script takes in following argument:
# 1. Folder_list   : List of train data folder used for running, for example, if you want to consider n = 
#                 50,100 200 and 400 you need to pass Folder_list argument as ( --Folder_list 50 100 200 400).
#
# 2. RunAnalysis  : If true, The RMSE and MAD will be caluclated for passed Folder_list argument and 
#                    result  will be plotted. If False, the result will be plotted from the exisitng 
#                    .txt  files (Note: This result will be plotted for the Folder_list argument used to
#                    create the saved .txt file, not the current Folder_list argument. Note this 
#                    command doesn't save the result, to save the result check out the isSave flag.
#
# 3. isSave       : If true the result of RMSE and MAD analysis will be overwritten in the exisiting
#                   .txt file, and the plot will be saved.
#
# To run the script: Normal example:-
#         python plotRMSE.py --Folder_list 50 100 200 --isSave True/False --RunAnalysis True/False
# 
# To reproduce the result:
#         python plotRMSE.py                            
#         (Default Folder_list is set to n = [50,100,200,400,1600,3200] (reported in paper)                    #         This will only plot the result (Not save) from the existing .txt file)
# Note: All the loading and cleaning of workspace is automated in this script.


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
    t, phi = tdd.getPhiPD(Zmax*xp[0:2], xp[2:4], xp[4], isPlot = False)\
    
    try:
        np.savetxt(r"relax.dat", np.c_[t, phi])
    except:
        while not os.path.exists(r"relax.dat"):
            time.sleep(1)
        np.savetxt(r"relax.dat", np.c_[t, phi])
        
    par  = spec.readInput(r"inpReSpect.dat")    
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
        return hp,h_true,phi_predicted,phi_true
    else:
        return hp,h_true,None,None

# Ploting routine, make change here to edit the figure appearance
    
def Plot(RMSE_h=None,RMSE_phi=None,MAD_h=None,MAD_phi=None,From_txt=False,SavePlot=False):
    
    """
    RMSE_h    = Root Mean Squared Error for h(s).
    RMSE_Phi  = Root Mean Squared Error for Phi(t).
    Mad_h     = Medain Absolute deviation for h(s).
    Mad_Phi   = Medain Absolute deviation for Phi(st).
    From_txt  = If true load value of above 4 argument from existing .txt file.
    SavePlot  = If true, save the figure.
    
    """
    
    
    if From_txt == True:
        #Load data
        RMSE_h = np.loadtxt("RMSE_h.txt")
        RMSE_phi = np.loadtxt("RMSE_phi.txt")

        MAD_h = np.loadtxt("MAD_h.txt")
        MAD_phi = np.loadtxt("MAD_phi.txt")

    ################# Plotting routine for median and mean absoulute deviation ###################
    fig, axs = plt.subplots(1,2, figsize=(14, 7), facecolor='w', edgecolor='k')
    axs = axs.ravel()
    #read labels for n, to plot
    n = RMSE_h[:,0].astype(int)  #The first column of all .txt file contain n
    
    df1 = pd.DataFrame(MAD_h[:,1:].T,columns=n).assign(Data=r"$h$")  #Skip first column
    df2 = pd.DataFrame(MAD_phi[:,1:].T,columns=n).assign(Data=r"$\Phi$")
    df3 = pd.DataFrame(RMSE_h[:,1:].T,columns=n).assign(Data=r"$h$")
    df4 = pd.DataFrame(RMSE_phi[:,1:].T,columns=n).assign(Data=r"$\Phi$")
    
    
    #cdf = pd.concat([df1, df2])
    cdf = df1                               #Plot only h, if you want to plot h & phi uncomment above line
    mdf = pd.melt(cdf, id_vars=['Data'], var_name=['Letter'])
    #print(mdf)
    sns.boxplot(x="Letter", y="value", hue="Data", data=mdf, ax=axs[0])
    axs[0].set_xlabel(r'$n$')
    axs[0].set_ylabel('MAD')
    axs[0].set_yscale('log')
    axs[0].get_legend().remove()
    
    #cdf = pd.concat([df3, df4])  
    cdf = df3                                #Plot only h, if you want to plot h & phi uncomment above line
    mdf = pd.melt(cdf, id_vars=['Data'], var_name=['Letter'])
    #print(mdf)
    sns.boxplot(x="Letter", y="value", hue="Data", data=mdf, ax=axs[1])    
    axs[1].set_xlabel(r'$n$')
    axs[1].set_ylabel("RMSE")
    axs[1].set_yscale('log')
    axs[1].get_legend().remove()

    if SavePlot == True:
        plt.savefig("TestAnalysis.pdf",bbox_inches='tight', pad_inches=0.25)
    plt.show()
    
#Routine to run/save analysis.
    
def run_analysis(Folder_list , isSave):
    """
    Folder_list = List of training folder, i.e n
    isSave      = If true, save the analysis result in .txt files, and save the plot.
    """
    
    Parent_dir  = "C:\\Users\\18503\\Dropbox\\RA\\Code\\RA\\PatchUp\\PatchUp\\Sachin"
    Test_data   = np.loadtxt(os.path.join(os.path.join(Parent_dir,"TestData","test.dat")))
    #In shape argument 2 is for=>(For H, For Phi)
    RMSE  = np.zeros(shape=(2,len(Folder_list),len(Test_data)))  #placeholder for RMSE
    MAD   = np.zeros(shape=(2,len(Folder_list),len(Test_data))) #placeholder for Median absolute deviation
                             
    n = []
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
            n.append(len(xtrain)) # used to plot n vs error
            param = np.loadtxt(os.path.join(Parent_dir,"TrainData","hyper.dat"))[1:-2]

            for index,test_point in enumerate(Test_data):
                
                     print("Routine Running for Folder,TestPoint# ",folder," ",index,"\n")
                     #print(test_point,"\n")
                     hp, dhp = predict(test_point, sd, xtrain, Zd, param)
                     hp,h_true,phi_predicted,phi_true = plotPredTrain2(hp, dhp, sd, \
                                                                       test_point ,meanZd,True,False)  
                     #RMSE Calculation
                     RMSE[0][F_idx][index] = np.linalg.norm(hp-h_true)/len(h_true)
                     RMSE[1][F_idx][index] = np.linalg.norm(phi_predicted-phi_true)/len(phi_true)
                    
                     #MAD Calculation
                     MAD[0][F_idx][index] = np.median(abs(hp-h_true))
                     MAD[1][F_idx][index] = np.median(abs(phi_predicted-phi_true))
                        
                     time.sleep(0.1)

    
    RMSE_h    =   np.c_[np.array(n).T,RMSE[0]]
    RMSE_phi  =   np.c_[np.array(n).T,RMSE[1]]
    MAD_h     =   np.c_[np.array(n).T,MAD[0]]
    MAD_phi   =   np.c_[np.array(n).T,MAD[1]]
    del_file()   
    
    if isSave == True:
        #Save n, and analysis result    
        np.savetxt("RMSE_h.txt",   RMSE_h)  
        np.savetxt("RMSE_phi.txt", RMSE_phi)
        np.savetxt("MAD_h.txt",  MAD_h)
        np.savetxt("MAD_phi.txt", MAD_phi)
        
    return RMSE_h,RMSE_phi,MAD_h,MAD_phi

    


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Plot RMSE and MAD")
                     
    parser.add_argument("--Folder_list", type=int, nargs='+', default=[50, 100, 200, 400, 800,1600,3200],
               help = "List of training folder, i.e n")
    parser.add_argument("--isSave", type=bool,default=False,
           help = "If true, save the analysis result in .txt files, and save the plot")
    parser.add_argument("--RunAnalysis", type=bool, default=False,
           help = "If true, The RMSE and MAD will be caluclated for passed Folder_list argument and result will be plotted, to save pass the isave argument. If False, the result will be plotted from the exisitng .txt files (Note: This result will be plotted for the Folder_list argument used to create the saved .txt file, not the current Folder_list argument.")
    
    argspar = parser.parse_args()
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')
    
    Folder_list = argspar.Folder_list
    isSave      = argspar.isSave
    RunAnalysis = argspar.RunAnalysis
    
    if RunAnalysis == True:
        RMSE_h,RMSE_phi,MAD_h,MAD_phi = run_analysis(Folder_list,isSave)
        Plot(RMSE_h,RMSE_phi,MAD_h,MAD_phi,From_txt=False,SavePlot = isSave)
    else:
        Plot(From_txt = True, SavePlot = isSave)