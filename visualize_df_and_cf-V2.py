# This python script plot CP,DF with different param.
# The script takes in following argument: 
#   1. paramNum: Plot the CP,DF decomposition for sigma_2,gamma_1,gamma_21,gamma_22,gamma_23,gamma_24 for
#                paramNum 0,1,2,3,4,5,6 respectively.
#   2. Folder: (NOte: dtype = int) It will load the training data of given folder to trainData workspace. 
#   3. SpanLogLim_list : the x-axis limits for plooting the params.       
#
# To run: 
#   For example if you want to plot sigma_2 for folder 100 in loglimit -1 -0.4 you will pass the following 
#   command:  python visualize_df_and_cf-V2.py --Folder 100 --paramNum 0 --SpanLogLim_list -1 -0.4
#
# Note: This script will clear the workspace and load the data for you (from the folder passed above) 
# No need to clear the data and load the data

from trainHyper import *
import subprocess

# The following command force matplotlib to use asmath package. Needed to plot the label ax[1] label defined below in the code.
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}'] #for \text command



def plot_df_cf(folder,paramNum=0, SpanLogLim_list=np.array([-1,1])):
    
    SpanLogLim = SpanLogLim_list[0]
    paramSpan = np.logspace(SpanLogLim[0], SpanLogLim[1], 100) #defined x-axis for plot
    dfSpan  = np.zeros(shape=(len(paramSpan),1))
    cpSpan  = np.zeros(shape=(len(paramSpan),1))
    objSpan = np.zeros(shape=(len(paramSpan),1))
      
    
    print("Routine running for Folder {0} \n".format(folder))
    cmd_1 = "python Load_workspace.py --ClearWorkspace True"
    out_str = subprocess.check_output(cmd_1, shell=True)
    print(out_str)
    time.sleep(0.5) #time given to clear the processes overhead

    cmd_2 = "python Load_workspace.py --load True --Folder {0}".format(folder)
    out_str = subprocess.check_output(cmd_2, shell=True)
    print(out_str)
    time.sleep(0.5)

    #Load the training data and hyperparam
    xtrain, sd, Zd, meanZd = readTrainData()
    param = np.loadtxt("TrainData/hyper.dat")[1:-2]
    #print(param)
    
    xlab = [r'$\sigma^2$', r'$\gamma_1$', r'$\gamma_{21}$', r'$\gamma_{22}$', r'$\gamma_{23}$',r'$\gamma_{24}$', r'$\gamma_{25}$']
    
    for i, p in enumerate(paramSpan):
        param[paramNum] = p
        #print(param)
        dfSpan[i], cpSpan[i], objSpan[i] = objParam(param, xtrain, sd, Zd, dprint = True)
        
    fig, ax1 = plt.subplots()
    ax1.set_xlabel(xlab[paramNum])
    ax1.set_ylabel(r'$\frac{ \text{log probability} }{10^4}$')
    ax1.plot(paramSpan, objSpan/1e4,label ="OBJ",alpha=0.8)
    ax1.plot(paramSpan, dfSpan/1e4,'--',label ="DF",alpha=0.8)
    ax1.plot(paramSpan, cpSpan/1e4,'--',label ="CP",alpha=0.8)
    #minima_index = np.argmin(objSpan)
    plt.axvline(paramSpan[np.argmin(objSpan)],color ='k',linestyle = '--',alpha= 0.7)
    
    anchored_text = AnchoredText("A", loc=2)
    ax1.add_artist(anchored_text)
        
    plt.legend()
    ax1.tick_params(axis='both', which='both', length=5)
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    
    print(np.c_[objSpan,dfSpan,cpSpan])
    plt.savefig("images//Decompose.png",bbox_inches='tight', pad_inches=0.10)
    plt.show()         

    return 0

#
# Create training data; Store results (spectra and input params) in TrainData/
#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=" Visualize df and cp")
    parser.add_argument("--paramNum", type=int, default=0,
                   help = "List of param for which objective function will be decomposed")
    parser.add_argument("--Folder", type=int, default=100,
                   help='Name of Folder of training set')    
    parser.add_argument('-f',"--SpanLogLim_list", type=float, nargs='+', action ='append',
               help = "List of param for which objective function will be decomposed")
    
    
    argspar = parser.parse_args()
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')
    
    paramNum = argspar.paramNum
    Folder    = argspar.Folder
    SpanLogLim_list = argspar.SpanLogLim_list
    SpanLogLim_list = np.array(SpanLogLim_list)
    plot_df_cf(Folder,paramNum,SpanLogLim_list)               
   
