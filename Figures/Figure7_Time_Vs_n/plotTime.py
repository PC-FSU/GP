#     This Python Script mainly contain the routine required to plot the prediction. The Differene betwenn 
#     this and original version is 1) i have included a functon that genratws phi,t back from h,s. 2)  
#     modification in function plotPredTrian,(Basically now it has a routine to check for whether  
#     original data lie within the range (prediction + 2.5*SD,prediction - 2.5*SD)  
import numpy as np
import matplotlib.pyplot as plt
plt.style.use(['seaborn-white', 'myjournal'])
from matplotlib.offsetbox import AnchoredText
#***************************
#Imports for latex-like label in matplotlib
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}' #for \text command

if __name__ == "__main__":
    
    x = np.array([815, 3199,6410,1.248000000000000000e+04])
    #a=np.array([37,36.100000,31,30.28,27])
    b=np.array([12,298,1808,2.556469500765800476e+05])
    plt.loglog(x,b, linestyle = 'None',marker='o',label="True")

    temp = ((x[1:]/x[0])**3)*b[0]
    temp = np.insert(temp,0,b[0])
    plt.loglog(x,temp,label =r"$O(n^3)$")


    plt.xlabel(r"$n$")
    plt.ylabel("Time(sec)")
    plt.legend()
    plt.savefig("time.pdf",bbox_inches='tight', pad_inches=0.25)
    plt.show()
