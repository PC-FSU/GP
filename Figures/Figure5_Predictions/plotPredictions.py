#This Scripy plot 3 subplots on a single figure instance
# first subplot plot prediction on training data
# second subplot plot predition ( H vs S) on a unknown test sample
# Third plot plot's phi vs T for same unknown data




#This routine is to call file in higher level directory from file in subdirectory. Example The path of file tdd.py is  "~//Sachin//" but since we are calling it from a file located in lower level subdirectory "~//Sachin//Figures//Figure_2_spectra_illustration" we need to add this PYTHONPATH routine.
#**************************
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..\\..'))
#***************************
from predictGP import *
import subprocess
import time
import shutil 

def Visualize(xp):
    
     #************************************************************    
    #Clear The workspace and load the params. You don't need to worry about this part, it has nothing to do with the plot, Just change the parent_dir as per your computer path.
    
    folder = "100" #Update folder if you want to check prediction for different n,(Training data size)
    
    print("Routine running for Folder {0} \n".format(folder))
    Parent_dir  = "C:\\Users\\18503\\Dropbox\\RA\\Code\\RA\\PatchUp\\PatchUp\\Sachin"
    cmd_1 = "python Load_workspace.py --ClearWorkspace True"
    out_str = subprocess.check_output(cmd_1, shell=True,cwd = Parent_dir)
    print(out_str)
    time.sleep(0.5) #time given to clear the processes overhead

    cmd_2 = "python Load_workspace.py --load True --Folder {0}".format(folder)
    out_str = subprocess.check_output(cmd_2, shell=True,cwd = Parent_dir)
    print(out_str)
    time.sleep(0.5)
    
    #Load the training data and hyperparam
    xtrain, sd, Zd, meanZd = readTrainData(Parent_dir)  #Path to specify where to load and read h.dat file
    param = np.loadtxt(os.path.join(Parent_dir,"TrainData","hyper.dat"))[1:-2]
    #print(param)
    #*************************************************************
    #Plotting Routine
    
    fig, ax = plt.subplots(1,3, figsize=(15,5))
    #****************For first plot(on training  data)****************
    temp_xp = xtrain[0]
    hp, dhp = predict(temp_xp, sd, xtrain, Zd, param)
    #Move some file for plotPredTrain to work properly
    # Source path 
    source = os.path.join(Parent_dir,"inpReSpect.dat")
    # Destination path 
    destination = os.path.join(Parent_dir,"Figures","Figure5_Predictions","inpReSpect.dat")
    # Copy the content of source to destination 
    dest = shutil.copy(source, destination) 
    
    #Call plotPredTrain
    _,hp,h_true,_ = plotPredTrain(hp, dhp, sd, temp_xp, meanZd,False)
    
    #predicted
    ax[0].plot(sd,hp,label='est')
    ax[0].fill_between(sd, hp - 2.5*dhp, hp + 2.5*dhp, alpha=0.1) #UQ band
    #True
    ax[0].plot(sd, h_true, 'gray',alpha=0.5, label='true')   
     
    ax[0].set_xscale('log')
    ax[0].set_xlabel('$s$')
    ax[0].set_ylabel('$h$')
    anchored_text = AnchoredText("A", loc=1)
    ax[0].add_artist(anchored_text)
    ax[0].legend(loc=3,prop={'size': 15})
    ax[0].tick_params(axis='both', which='both', length=5)
    
    #***************For second plot(on unseen data)*******************
    
    hp, dhp = predict(xp, sd, xtrain, Zd, param)
    _,hp,h_true,_ = plotPredTrain(hp, dhp, sd, xp, meanZd,False)
    
    #Predicted
    ax[1].plot(sd,hp,label='est')
    ax[1].fill_between(sd, hp - 2.5*dhp, hp + 2.5*dhp, alpha=0.1) #UQ band
    #True
    ax[1].plot(sd, h_true, 'gray',alpha=0.5, label='true')   
    
    ax[1].set_xscale('log')
    ax[1].set_xlabel('$s$')
    ax[1].set_ylabel('$h$')
    anchored_text = AnchoredText("B", loc=1)
    ax[1].add_artist(anchored_text)
    ax[1].legend(loc=3,prop={'size': 15})
    ax[1].tick_params(axis='both', which='both', length=5)
    
    #***************For Third plot(Get back phi vs T)*****************
    
    #Prediction 
    t,phi,dphi = Gt(sd, hp, dhp)  #call GT function which gives back phi(t) from h(s)
    ax[2].plot(t,phi,label='est') #This will plot mean 
    
    #For the upper estimate of SD
    h_upper = hp + 2.5*dhp
    _,phi_upper,_ = Gt(sd,h_upper, dhp) #This will give upper band
    
    #For the lower estimate of SD and to drop where h<0.
    h_lower = hp - 2.5*dhp
    h_lower = np.where(h_lower<0,0,h_lower) #This will give lower band
    _,phi_lower,_ = Gt(sd,h_lower, dhp)
    
    ax[2].fill_between(t, phi_upper, phi_lower, alpha=0.1)   #This will plot the band
    
    #Plot True
    t,phi,_ = Gt(sd, h_true, 0)
    
    ax[2].plot(t,phi,'gray',label='true')
    ax[2].set_xscale('log')
    ax[2].set_yscale('log')
    ax[2].set_xlabel(r'$t$')
    ax[2].set_ylabel(r'$\phi(t)$')
    ax[2].set_ylim(1e-5, 1e1)
    anchored_text = AnchoredText("C", loc=1)
    ax[2].add_artist(anchored_text)
    ax[2].legend(loc=3,prop={'size': 15})
    ax[2].tick_params(axis='both', which='both', length=5)
    
    
    #Save figure
    plt.tight_layout(pad=3.0)
    plt.savefig("predictions.pdf",bbox_inches='tight', pad_inches=0.10)
    plt.show()
    
    #remove the extra file
    os.remove("inpReSpect.dat"),os.remove("h.dat"),os.remove("relax.dat")
    os.rmdir("output") #This dir is created when call to constspec.py is made. We don't need it.
    
    return 0

if __name__ == "__main__":
    xp = np.array([43.0, 18, 1.15, 1.34, 0.38])
    xp[0:2] = xp[0:2]/50.0
    Visualize(xp)
