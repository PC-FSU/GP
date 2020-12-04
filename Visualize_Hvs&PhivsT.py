#This Scripy plot 3 subplots on a single figure instance
# first subplot plot prediction on training data
# second subplot plot predition ( H vs S) on a unknown test sample
# Third plot plot's phi vs T for same unknown data

from predictGP import *

def Visualize(xp):
    
    fig, ax = plt.subplots(1,3, figsize=(15,5))
    #****************For first plot(on training  data)****************
    temp_xp = xtrain[0]
    hp, dhp = predict(temp_xp, sd, xtrain, Zd, param)
    _,hp,h_true,_ = plotPredTrain(hp, dhp, sd, temp_xp, meanZd,False)
    
    #estimated
    ax[0].plot(sd,hp,label='est')
    ax[0].fill_between(sd, hp - 2.5*dhp, hp + 2.5*dhp, alpha=0.1)
    #predicted
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
    
    #estimated
    ax[1].plot(sd,hp,label='est')
    ax[1].fill_between(sd, hp - 2.5*dhp, hp + 2.5*dhp, alpha=0.1)
    #predicted
    ax[1].plot(sd, h_true, 'gray',alpha=0.5, label='true')   
    
    ax[1].set_xscale('log')
    ax[1].set_xlabel('$s$')
    ax[1].set_ylabel('$h$')
    anchored_text = AnchoredText("B", loc=1)
    ax[1].add_artist(anchored_text)
    ax[1].legend(loc=3,prop={'size': 15})
    ax[1].tick_params(axis='both', which='both', length=5)
    
    #***************For Third plot(Get back phi vs T)*****************
    
    #call GT function which gives back phi and t from H and S
    t,phi,dphi = Gt(sd, hp, dhp)
    ax[2].plot(t,phi,label='est')
    #ax[2].fill_between(t, phi + 2.5 * dphi, phi - 2.5 * dphi, alpha=0.1,color = 'b')
    
    #For the upper estimate of SD
    h_upper = hp + 2.5*dhp
    _,phi_upper,_ = Gt(sd,h_upper, dhp)
    
    #For the lower estimate of SD and drop where h<0.
    h_lower = hp - 2.5*dhp
    h_lower = np.where(h_lower<0,0,h_lower)
    _,phi_lower,_ = Gt(sd,h_lower, dhp)
    ax[2].fill_between(t, phi_upper, phi_lower, alpha=0.1)
       
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
    plt.tight_layout(pad=3.0)
    #plt.savefig("images//predictions.png",bbox_inches='tight', pad_inches=0.10)
    plt.show()
    return 0

if __name__ == "__main__":
    #xtrain, sd, Zd, meanZd = readTrainData()
    #param = np.loadtxt("TrainData/hyper.dat")[1:-2]
    xp = np.array([43.0, 18, 1.15, 1.34, 0.38])
    xp[0:2] = xp[0:2]/50.0
    #All the args passed below is called in predictGP.py script so need to call them again
    hp, dhp = predict(xp, sd, xtrain, Zd, param) 
    Visualize(xp)
