#
# This python script plot the mean and median deviation between mean prediction and actual result, also we can 
# use this script to analysis whether the actual value is in range of +-2.5 SD of predicted value. 
#
# This script takes in following argument:
# 1. idx: I have 10 different test set in TrainData folder where each data has ~25 examples, so idx run
#         will consider first idx number of test set to run the analysis, (Note: it should be <= 10,as
#         there's only 10 test dataset.  
#
# 2. Folder_list: List of test data folder used for analysis, for example, if you want to consider n = 50,100
#                 200 and 400 you need to pass Folder_list argument as ( --Folder_list 50 100 200 400).
#
# 3. Adverserial: If true the test will be done on adverserial test set. Adverserial test have example which 
#                 are on extreme end of our input space. Note: You don't need idx flag if you are running it
#                 for Adverserial example.

# To run the script: Normal example:-
#         python Visualize_TestErrorAnalysis.py --idx 5 --Folder_list 50 100 200
#
#         Adverserial example:
#           python Visualize_TestErrorAnalysis.py --Folder_list 50 100 200 --Adverserial True
#
#
#  Note: All the loading and cleaning of workspace is automated in this script.
#



import glob 
import subprocess
import time
import sys
import seaborn as sns
import pandas as pd
from predictGP import *
from scipy.stats import median_abs_deviation


### MAIN ROUTINES ###

#Calculate Median absolute deviation 
def mad(data, axis=None):
    return np.mean(abs(data - np.mean(data, axis)), axis)

def Plot(Error_Median,Error_Mean,n,DICT):
        
    ################# Plotting routine for median and mean absoulute deviation ###################
    fig, axs = plt.subplots(1,3, figsize=(14, 7), facecolor='w', edgecolor='k')
    axs = axs.ravel()
    labels = [str(i) for i in n]
    
    df1 = pd.DataFrame(Error_Median[0].T,columns=n).assign(Data=r"$\Phi$")
    #df1 = df1.reset_index(drop=True, inplace=True)
    df2 = pd.DataFrame(Error_Median[1].T,columns=n).assign(Data=r"$h$")
    df3 = pd.DataFrame(Error_Mean[0].T,columns=n).assign(Data=r"$\Phi$")
    df4 = pd.DataFrame(Error_Mean[1].T,columns=n).assign(Data=r"$h$")
    
    cdf = pd.concat([df1, df2])    
    mdf = pd.melt(cdf, id_vars=['Data'], var_name=['Letter'])
    print(mdf)
    sns.boxplot(x="Letter", y="value", hue="Data", data=mdf, ax=axs[0])
    
    cdf = pd.concat([df3, df4])    
    mdf = pd.melt(cdf, id_vars=['Data'], var_name=['Letter'])
    print(mdf)
    sns.boxplot(x="Letter", y="value", hue="Data", data=mdf, ax=axs[1])    
    
    
    axs[1].set_xlabel(r'$n$')
    axs[1].set_ylabel(r'$Mean Absolute Deviation$')

    Dict_data = np.array([[i['Average_fraction_out'] for i in j] for j in DICT],dtype=object).T
    Dict_data = pd.DataFrame(Dict_data,columns=n)
    sns.boxplot(x="variable", y="value", data=pd.melt(df),ax=axs[2])
    axs[2].set_xlabel(r'$n$')
    axs[2].set_ylabel(r'$Average Fraction Out $')
    
    plt.savefig("images//Test_error.png",bbox_inches='tight', pad_inches=0.25)
    plt.show()
    
    
def run_analysis(Folder_list,idx,Test_Datasets,RunAnalysis_For_Phi):
    
    Error_Median  = np.zeros(shape=(len(Folder_list),idx))  #placeholder for mean_absolute_Deviation
    Error_Mean    = np.zeros(shape=(len(Folder_list),idx))  #placeholder for median_absolute_deviation
    
    print(Test_Datasets)
    n = []
    # We have a dict for each folder that contain test error analysis info, and this list contain all those
    # dicts
    InRangePrediction_list = []
    
    for F_idx,folder in enumerate(Folder_list):
            #Clean Workspace
            print("Routine running for Folder {0} \n".format(folder))
            cmd_1 = "python Load_workspace.py --ClearWorkspace True"
            out_str = subprocess.check_output(cmd_1, shell=True)
            print(out_str)
            time.sleep(2.5)
            
            #Load workspace 
            cmd_2 = "python Load_workspace.py --load True --Folder {0}".format(folder)
            out_str = subprocess.check_output(cmd_2, shell=True)
            print(out_str)
            time.sleep(2.5)
            
            #Load the training data and hyperparam
            xtrain, sd, Zd, meanZd = readTrainData()
            n.append(len(xtrain)) # used to plot n vs error
            param = np.loadtxt("TrainData/hyper.dat")[1:-2]
            
            # Placeholder  for
            #   1: mean + 2.5*SD > actual value > mean-2.5*SD
            #   2:  Average_fraction_out: % of grid point out from band defined in point a.
            
            dict_collect  = {}
            dict_collect['total_count'] = 0
            dict_collect['positive_count'] = 0
            dict_collect['Average_fraction_out'] = 0
            
            for i in range(idx):
                xtest  = np.loadtxt(Test_Datasets[i])
                temp = []
                for xp in xtest:
                    hp, dhp = predict(xp, sd, xtrain, Zd, param)
                    #err,_,_,Prediction_InRange = plotPredTrain(hp, dhp, sd, xp, meanZd,False)
                    err,_,h_true,Prediction_InRange = plotPredTrain(hp, dhp, sd, xp, meanZd,RunAnalysis_For_Phi,True)
                    
                    temp.append(err)
                    print("For Xp ",xp,"Error is ", err," # ",dict_collect['total_count'])
                    time.sleep(0.1)
                    dict_collect['total_count'] += 1
                    if Prediction_InRange['isInRange']:
                        dict_collect['positive_count'] += 1
                    else:
                        #dict_collect[dict_collect['total_count']] = Prediction_InRange
                        dict_collect['Average_fraction_out'] += Prediction_InRange['OutRange_fraction']
                        #dict_collect['Average_fraction_out'].append(Prediction_InRange['OutRange_fraction'])
                        
                Error_Median[F_idx][i] = median_abs_deviation(temp,axis=None)
                Error_Mean[F_idx][i]   = mad(temp,axis=None)
                print("*************Error for dataset %d of Folder %d is %f, %f**************" %
                      (i,folder,Error_Median[F_idx][i],Error_Mean[F_idx][i]))
                time.sleep(0.5)
            dict_collect['Average_fraction_out'] = dict_collect['Average_fraction_out']/dict_collect['total_count']
            InRangePrediction_list.append(dict_collect)   
            time.sleep(0.5)  
    
    print(InRangePrediction_list)
    
    return Error_Median,Error_Mean,n,InRangePrediction_list
  



if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="analyze test data")
    
    parser.add_argument("--idx", type=int, default=0,
                        help='Run on first {idx} test set')
    
    parser.add_argument("--Folder_list", type=int, nargs='+', default=[50, 100, 200, 400],
               help = "List of training folder")
    parser.add_argument("--Adverserial",type=bool,default = False,
                        help = "If true the test will be done on adverserial test set")
    
    argspar = parser.parse_args()
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')
    
    idx =  argspar.idx
    Folder_list = argspar.Folder_list
    Adverserial = argspar.Adverserial
    
    if Adverserial:
        Test_Datasets  = glob.glob(os.path.join("TestData","*.txt"))
        idx = 1
    else:
        Test_Datasets  = glob.glob(os.path.join("TestData","*.dat"))
            #Define PlaceHolder for mean and median to store analysis result for Phi and H
        MEAN = []
        MEDIAN = []
        DICT = []
        
        Temp_mean,Temp_median,n,temp_dict = run_analysis(Folder_list,idx,Test_Datasets,False)
        MEAN.append(Temp_mean)
        MEDIAN.append(Temp_median)
        DICT.append(temp_dict)
        
        Temp_mean,Temp_median,_,temp_dict = run_analysis(Folder_list,idx,Test_Datasets,True)
        MEAN.append(Temp_mean)
        MEDIAN.append(Temp_median)
        DICT.append(temp_dict)
        
        #Plot(MEDIAN,MEAN,n,DICT)

























        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


#    np.save("MEDIAN",MEDIAN)
    #    np.save("MEAN",MEAN)
    #    MEAN = np.load("MEAN.npy")
    #    MEDIAN = np.load("MEDIAN.npy")
    #    n = [50,100,200,400]
    #    Error_Median  = np.zeros(shape=(len(n),2))  #placeholder for mean_absolute_Deviation
    #    Error_Mean    = np.zeros(shape=(len(n),2))  #placeholder for median_absolute_deviation
    #    print(MEAN.shape)
    #    print(MEDIAN)      #Dimesnsion = (2,Folder_list,idx)
    #    print("Hers ", DICT)
    
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#    print(Test_Datasets)
#    n = []
#    # We have a dict for each folder that contain test error analysis info, and this list contain all those
#    # dicts
#    InRangePrediction_list = []
#    
#    for F_idx,folder in enumerate(Folder_list):
#            #Clean Workspace
#            print("Routine running for Folder {0} \n".format(folder))
#            cmd_1 = "python Load_workspace.py --ClearWorkspace True"
#            out_str = subprocess.check_output(cmd_1, shell=True)
#            print(out_str)
#            time.sleep(5)
#            
#            #Load workspace 
#            cmd_2 = "python Load_workspace.py --load True --Folder {0}".format(folder)
#            out_str = subprocess.check_output(cmd_2, shell=True)
#            print(out_str)
#            time.sleep(5)
#            
#            #Load the training data and hyperparam
#            xtrain, sd, Zd, meanZd = readTrainData()
#            n.append(len(xtrain)) # used to plot n vs error
#            param = np.loadtxt("TrainData/hyper.dat")[1:-2]
#            
#            # Placeholder  for
#            #   1: mean + 2.5*SD > actual value > mean-2.5*SD
#            #   2:  Average_fraction_out: % of grid point out from band defined in point a.
#            
#            dict_collect  = {}
#            dict_collect['total_count'] = 0
#            dict_collect['positive_count'] = 0
#            dict_collect['Average_fraction_out'] = 0
#            
#            for i in range(idx):
#                xtest  = np.loadtxt(Test_Datasets[i])
#                temp = []
#                for xp in xtest:
#                    hp, dhp = predict(xp, sd, xtrain, Zd, param)
#                    #err,_,_,Prediction_InRange = plotPredTrain(hp, dhp, sd, xp, meanZd,False)
#                    err,_,h_true,Prediction_InRange = plotPredTrain(hp, dhp, sd, xp, meanZd,False,False)
#                    
#                    temp.append(err)
#                    print("For Xp ",xp,"Error is ", err," # ",dict_collect['total_count'])
#                    time.sleep(0.1)
#                    dict_collect['total_count'] += 1
#                    if Prediction_InRange['isInRange']:
#                        dict_collect['positive_count'] += 1
#                    else:
#                        #dict_collect[dict_collect['total_count']] = Prediction_InRange
#                        dict_collect['Average_fraction_out'] += Prediction_InRange['OutRange_fraction']
#                        
#                Error_Median[F_idx][i] = median_abs_deviation(temp,axis=None)
#                Error_Mean[F_idx][i]   = mad(temp,axis=None)
#                print("*************Error for dataset %d of Folder %d is %f, %f**************" %
#                      (i,folder,Error_Median[F_idx][i],Error_Mean[F_idx][i]))
#            
#            dict_collect['Average_fraction_out'] = dict_collect['Average_fraction_out']/dict_collect['total_count']
#            InRangePrediction_list.append(dict_collect)   
#            time.sleep(5)  
#            
#    
#        ################# Plotting routine for median and mean absoulute deviation ###################
#    fig, axs = plt.subplots(1,2, figsize=(14, 7), facecolor='w', edgecolor='k')
#    axs = axs.ravel()
#    labels = [str(i) for i in n]
#    
#    
#    Data   = [Error_Median[i] for i in range(len(Folder_list))]
#    axs[0].boxplot(Data,
#                vert=True,  # vertical box alignment
#                patch_artist=True,  # fill with color
#                labels=labels)  # will be used to label x-ticks
#    #title = "Median Absolute Deviation on " + str(idx) +" test dataset"
#    #axs[0].set_title(title)
#    axs[0].set_xlabel(r'$n$')
#    axs[0].set_ylabel(r'$Median Absolute Deviation$')
#    axs[0].tick_params(axis="x")
#    axs[0].tick_params(axis="y")
#    axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
#    
#    Data   = [Error_Mean[i] for i in range(len(Folder_list))]
#    axs[1].boxplot(Data,
#                vert=True,  # vertical box alignment
#                patch_artist=True,  # fill with color
#                labels=labels)  # will be used to label x-ticks
#    
#    #title = "Mean Absolute Deviation on " + str(idx) +" test dataset"
#    #axs[1].set_title(title)
#    axs[1].set_xlabel(r'$n$')
#    axs[1].set_ylabel(r'$Mean Absolute Deviation$')
#    axs[1].tick_params(axis="x")
#    axs[1].tick_params(axis="y")
#    axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
#    fig.subplots_adjust(wspace=1)
#    plt.tight_layout(pad=2.0)
#    plt.savefig("images//Test_error.png",bbox_inches='tight', pad_inches=0.25)
#    plt.show()
#    print(InRangePrediction_list)
    