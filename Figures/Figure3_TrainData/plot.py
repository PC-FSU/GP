
import os
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['seaborn-white', 'myjournal'])
from matplotlib.offsetbox import AnchoredText
#***************************
#Imports for latex-like label in matplotlib
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' #for \text command

def plot():
    
    #load data
    Parent_dir  = "C:\\Users\\18503\\Dropbox\\RA\\Code\\RA\\PatchUp\\PatchUp\\Sachin"
    xtrain  =  np.loadtxt(os.path.join(Parent_dir,"TrainData","100","xtrain.dat"))
    
    
    fig,ax =  plt.subplots(1,3,figsize=(12,5))
    
    ax[0].plot(xtrain[:,0], xtrain[:,1],'s', alpha=0.5)
    ax[0].set_xlabel(r'$Z_1$')
    ax[0].set_ylabel(r'$Z_2$')
    anchored_text = AnchoredText("A", loc=2)
    ax[0].add_artist(anchored_text)
    ax[0].tick_params(axis='both', which='both', length=5) 

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

    #plt.legend()
    plt.tight_layout(pad=1.5)
    plt.savefig("Hypercubic_Samples.pdf",bbox_inches='tight', pad_inches=0.10)
    plt.show()
       
    return 0
    

if __name__ == "__main__":
    plot()
  