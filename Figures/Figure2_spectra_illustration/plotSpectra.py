# 
#This script plot relaxation modulus t,phi and corresponding relaxation spectra s,h for a given x
#


#This routine is to call file in higher level directory from file in subdirectory. Example The path of file tdd.py is  "~//Sachin//" but since we are calling it from a file located in lower level subdirectory "~//Sachin//Figures//Figure_2_spectra_illustration" we need to add this PYTHONPATH routine.

#**************************
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..\\..'))
#***************************

#Import 
import tdd
from contSpec import *
import subprocess
import shutil
from matplotlib.offsetbox import AnchoredText

#***************************
#Imports for latex-like label in matplotlib
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' #for \text command


def plot_spectra(x):
    """
            plot spectra for a single sample
                    (1) runs and print TDD model to obtain (t, phi(t))
                    (2) runs and print pyReSpect to obtain  s, Hs
            Note: The graph is gonna plot for three set of w1 (0.1,passed_w1,0.9), all other param will be same
    """

    print(x)
    
    #Placeholder to save t,phi(t),s,h(s). 
    #Note: Since we are passing 4 diff x, the len(t_list) = 4, ith place will hold the data for ith x.
    t_list   = []
    phi_list = []
    s_list   = []
    h_list   = []
    
    #Clear Workspace
    Parent_dir  = "C:\\Users\\18503\\Dropbox\\RA\\Code\\RA\\PatchUp\\PatchUp\\Sachin"
    cmd_1 = "python Load_workspace.py --ClearWorkspace True"
    out_str = subprocess.check_output(cmd_1, shell=True, cwd = Parent_dir )
    print(out_str)
    count = 0
    
    #loop over all data-point
    for xp in x:
        Zw = xp[0:2]
        pd = xp[2:4]
        w1 = xp[4]
        print(xp)
        
        # run dynamics using TDD, get phi(t)
        t, phi = tdd.getPhiPD(Zw, pd, w1, isPlot = False)
        np.savetxt(os.path.join(Parent_dir,"relax.dat"), np.c_[t, phi]) # Save relax.dat file in parent dir
        par = readInput(os.path.join(Parent_dir,"inpReSpect.dat"))      # load inReSpect.dat,located in parent dir
        par['GexpFile'] = os.path.join(Parent_dir,"relax.dat")          # Change the parent dic, to correct loacation of relax.dat
        
        # run constspec to get h(s)
        _,_ = getContSpec(par)
        t_list.append(t),phi_list.append(phi)
        data = np.genfromtxt("h.dat", delimiter='') #load s,h(s) from h.dat
        s_list.append(data[:,0]),h_list.append(data[:,1]) #save it in placeholder to plot later
        
    os.remove("h.dat") #clear the h-file generated 
    os.rmdir("output") #This dir is created when call to constspec.py is made. We don't need it.
    
    #***************Plot phi(t) vs t*************************
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    color = ['b', 'orange', 'g', 'orange']
    linestyle = ['-','-','-','--']
    for i in range(len(x)):
        ax[0].plot(t_list[i], phi_list[i],label=r'$W_1$'+" = " +str((x[i][4])), \
                   color = color[i],linestyle = linestyle[i],alpha = 0.8)

    ax[0].set_xlabel(r'$t$')
    ax[0].set_ylabel(r'$\phi(t)$')
    ax[0].set_ylim(1e-4, None)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    anchored_text = AnchoredText("A", loc=1)
    ax[0].add_artist(anchored_text)
    ax[0].legend(prop={'size': 12.5})
    ax[0].tick_params(axis='both', which='both', length=5) 
    

    #*******************Plot h(s) vs s**************************************
    for i in range(len(x)):
        ax[1].plot(s_list[i], h_list[i], label=r'$W_1$'+" = " +str((x[i][4])),color = color[i], \
                   linestyle = linestyle[i],alpha = 0.8)

    ax[1].set_xlabel(r'$s$')
    ax[1].set_ylabel(r'$h$')
    ax[1].set_ylim(1e-4, None)
    ax[1].set_xscale('log')
    anchored_text = AnchoredText("B", loc=1)
    ax[1].add_artist(anchored_text)
    ax[1].legend(prop={'size': 12.5})
    ax[1].tick_params(axis='both', which='both', length=5) 
    
    
    #Save the figure
    plt.tight_layout(pad=2.0)
    plt.savefig("spectrum.pdf",bbox_inches='tight', pad_inches=0.10)
    plt.show()

    return None

if __name__ == "__main__":
    xp = np.array([[50.0, 5, 1.01, 1.01, 0.1],
                  [50.0, 5, 1.01, 1.01, 0.5],
                  [50.0, 5, 1.01, 1.01, 0.9],
                  [50.0, 5, 1.5, 1.5, 0.5]])
    plot_spectra(xp)
    
    