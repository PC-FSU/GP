# 
#This script plot t,phi and corresponding relaxation spectra s,H for a given input
#



# relaxation spectra
import tdd
from contSpec import *
import subprocess
import shutil
from matplotlib.offsetbox import AnchoredText


def plot_spectra(x):
    """
            plot spectra for a single sample
                    (1) runs and print TDD model to obtain (t, phi(t))
                    (2) runs and print pyReSpect to obtain  s, Hs
            Note: The graph is gonna plot for three set of w1 (0.1,passed_w1,0.9), all other param will be same
    """

    print(x)
    # run dynamics using TDD, gwet phi(t)
    t_list   = []
    phi_list = []
    s_list   = []
    h_list   = []
    
    #Clear Workspace
    cmd_1 = "python Load_workspace.py --ClearWorkspace True"
    out_str = subprocess.check_output(cmd_1, shell=True)
    print(out_str)
    count = 0
    
    for xp in x:
        Zw = xp[0:2]
        pd = xp[2:4]
        w1 = xp[4]
        print(xp)
        t, phi = tdd.getPhiPD(Zw, pd, w1, isPlot = False)
        np.savetxt("relax.dat", np.c_[t, phi])
        par = readInput('inpReSpect.dat')
        _, _ = getContSpec(par)
        t_list.append(t),phi_list.append(phi)

        ## Source path  
        #source = 'C://Users//18503//Dropbox//RA//Code//RA//PatchUp//PatchUp'
        #source_file  = os.path.join(source,"h.dat")
        ## Destination path  
        #destination_file = os.path.join(source,"TrainData","h{}.dat".format(count))
        #shutil.move(source_file, destination_file)
        #count+=1
        data = np.genfromtxt('h.dat', delimiter='')
        #print(data)
        s_list.append(data[:,0]),h_list.append(data[:,1])
        time.sleep(0.02)
        

    #Plot t vs phi
    
    fig, ax = plt.subplots(1,2,figsize=(10,5))
  
    color = ['b', 'orange', 'g', 'orange']
    linestyle = ['-','-','-','--']
    #markersize = [5,5,5,8]
    for i in range(len(x)):
        ax[0].plot(t_list[i], phi_list[i],label=r'$W_1$'+" = " +str((x[i][4])),color = color[i],linestyle = linestyle[i],alpha = 0.8)
#        plt.plot(t_list[1], phi_list[1], linewidth=4,alpha=0.5,label=r'$W_1$'+" =" +str(0.5))
#        plt.plot(t_list[2], phi_list[2], linewidth=4,alpha=0.5,label=r'$W_1$'+" =" +str(0.9))

    ax[0].set_xlabel(r'$t$')
    ax[0].set_ylabel(r'$\phi(t)$')

    ax[0].set_ylim(1e-4, None)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    anchored_text = AnchoredText("A", loc=1)
    ax[0].add_artist(anchored_text)
    ax[0].legend(prop={'size': 12.5})
    ax[0].tick_params(axis='both', which='both', length=5) 
    
    for i in range(len(x)):
        ax[1].plot(s_list[i], h_list[i], label=r'$W_1$'+" = " +str((x[i][4])),color = color[i],linestyle = linestyle[i],alpha = 0.8)
#    plt.plot(s_list[1], h_list[1], linewidth=4,alpha=0.5,label=r'$W_1$'+" =" +str(0.5))
#    plt.plot(s_list[2], h_list[2], linewidth=4,alpha=0.5,label=r'$W_1$'+" =" +str(0.9))

    ax[1].set_xlabel(r'$s$')
    ax[1].set_ylabel(r'$h$')
    ax[1].set_ylim(1e-4, None)
    ax[1].set_xscale('log')
    anchored_text = AnchoredText("B", loc=1)
    ax[1].add_artist(anchored_text)
    ax[1].legend(prop={'size': 12.5})
    ax[1].tick_params(axis='both', which='both', length=5) 
    
    plt.tight_layout(pad=2.0)
    plt.savefig("images/spectrum.png",bbox_inches='tight', pad_inches=0.10)
    plt.show()

    return None


#
# Create training data; Store results (spectra and input params) in TrainData/
#
if __name__ == "__main__":
    xp = np.array([[50.0, 5, 1.01, 1.01, 0.1],
                  [50.0, 5, 1.01, 1.01, 0.5],
                  [50.0, 5, 1.01, 1.01, 0.9],
                  [50.0, 5, 1.5, 1.5, 0.5]])
    plot_spectra(xp)
    
    