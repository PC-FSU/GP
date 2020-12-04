#
#
#  This Script Generate Test Data, I am gonna include Xtest data in Zip file so in general you don't have to  
#   run this script. 
#   
#  This script takes argument:
#    1. Sample_Size : Number of sample tha you want to draw from hypercubic sampling for test data
#          Note: The generated Sample are then subjected to constrained offer by our problem, like M1>M2,
#                This lead to elimination of half of the genearted sample, so in reality we actual size is 
#                ~Sample_Size/2.
#    2. Param_Value: "value for (Z_min,Z_max,pdmin,pdmax), fixed for our case, but in case."
#    3. Seed:  seed use to generate hypercubic samples
#    4. isPlot: Flag to plot the drawn hypercubic test sample
#    5. isPlot_predictions: 'Flag to visualize prediction on random test data, Note:- The data is not going be saved, only prediction is going to be plotted
#    6. Check_prediction: If true prediction will be done on a single training samples, This is just a 
#                         extra utilities, and can be removed.
#
#########Flags to generate the test data for performing error analysis with different xtrain size,n#########
#    7. nreplica: Number of Copy of xtest data of size Sample_Size, you need to generate
#                 Note: when nreplica == 0, a passed seed or seed=0 is selected and kept same for other 
#                       nreplica - 1. 
#
#    8. Gen_Data: If True, This will generate (nreplica) copy of different xtest data (of size 
#                        Sample_Size) for running test_error_analysis.py and save it in TestData Folder").
#                The data will be saved in TestData Folder, with filename xtest_0.dat...to...
#                xtest_neplica.dat. Note: if Gen_data is True and there are existing file in TestData folder, 
#                it will delete all the exisiting data and then rewrites the data.
#
#
#  How to run:
#     To create data only for Visualizing prediction
#     => python genTest.py --Sample_Size 100 --isPlot_predictions True
#     To create data and save it for future analysis
#     => python genTest.py --Sample_Size 100 --Gen_Data True --nreplica 5

from predictGP import *
import pyDOE as doe
import glob 


def generate_test(Zmin=5., Zmax=50., pdmin=1.01, pdmax=1.50, isPlot=False, Gen_Data = False, Sample_Size = 10, isSeed=True,Seed = 0):
    
    if isSeed:
        np.random.seed(Seed)
        
    xtest = doe.lhs(5, samples=Sample_Size)   #Draw Hypercubic samples
    
    #xtest[:,0] = Zmin + (Zmax-Zmin)*xtest[:,0]     #scale m1 to have value between Zmin and Zmax
    #xtest[:,1] = Zmin + (Zmax-Zmin)*xtest[:,1]     #scale m2 to have value between Zmin and Zmax
    
    xtest[:,2] = pdmin + (pdmax-pdmin)*xtest[:,2]  
    xtest[:,3] = pdmin + (pdmax-pdmin)*xtest[:,3]
    
    xtest = xtest[xtest[:,1] < xtest[:,0]]                 # constraint M1>M2
    xtest[:0] = xtest[xtest[:,0]>0]                        #Constraint that M1>0 and M2>0
    xtest[:1] = xtest[xtest[:,1]>0]
    xtest = xtest[xtest[:,4] > 0.0001]                     # w1>0
    
    if isPlot:
        plt.subplots(1,3, figsize=(15,5))

        plt.subplot(131)
        plt.plot(xtrain[:,0], xtest[:,1],'s', alpha=0.5)
        plt.axis('equal')
        plt.xlabel(r'$Z_1$')
        plt.ylabel(r'$Z_2$')
        plt.tight_layout()

        plt.subplot(132)
        plt.plot(xtrain[:,0], xtest[:,4],'s', alpha=0.5)
        plt.xlabel(r'$Z_1$')
        plt.ylabel(r'$w_1$')

        plt.subplot(133)
        plt.plot(xtrain[:,2], xtest[:,3],'s', alpha=0.5)
        plt.xlabel(r'$\rho_1$')
        plt.ylabel(r'$\rho_2$')


        plt.legend()
        plt.title("Uniform Samples")
        plt.tight_layout()
        plt.show()
    
    #Save Data
    if Gen_Data:
        np.savetxt("xtest.dat", xtest, fmt="%6.2f")
    
    return xtest
    



### MAIN ROUTINES ###

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate test data")

    parser.add_argument("--Sample_Size", type=int, default=20,
                        help='Number of sample tha you want to draw from hypercubic sampling for test data')
    
    parser.add_argument("--Param_Value", type = int, nargs="+", default=[5, 50, 1.01, 1.50],
                       help = "value for (Z_min,Z_max,pdmin,pdmax), fixed for our case, but in case.")
    
    parser.add_argument("--Seed", type=int, default = 0,
                        help='seed use to generate hypercubic samples')
    
    parser.add_argument("--isPlot", type=bool, default=False,
                        help='Flag to plot the drawn hypercubic test sample')

    parser.add_argument("--isPlot_predictions", type=bool, default=False,
                    help='Flag to visualize prediction on random test data, Note:- The data is not going be saved, only prediction is going to be plotted')
    
    parser.add_argument("--Check_prediction", type=bool, default=False,
                        help='If true prediction will be done on a single training samples')
    
    
    #Flags to generate the test data for performing error analysis with differbt xtrain size,n.
    
    parser.add_argument("--nreplica",type=int,default=0,
                        help = "Number of Copy of xtest data of size Sample_Size, you need to generate")
    
    parser.add_argument("--Gen_Data",type=bool,default=False,
                        help="If True, This will generate (nreplica) copy of different xtest data (of size \
                        Sample_Size) for running test_error_analysis.py and save in TestData Folder")
    
    argspar = parser.parse_args()

    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')


    isPlot = argspar.isPlot
    Seed        = argspar.Seed
    Sample_Size = argspar.Sample_Size
    Zmin,Zmax,pdmin,pdmax = argspar.Param_Value
    isPlot_predictions    = argspar.isPlot_predictions
    Check_prediction      = argspar.Check_prediction
    nreplica =  argspar.nreplica
    Gen_Data =  argspar.Gen_Data

    
    # Load data for prediction
    xtrain, sd, Zd, meanZd = readTrainData()
    param = np.loadtxt("TrainData/hyper.dat")[1:-2]
    
    
    #PLOT PREDICTIONS on random test data set
    if isPlot_predictions:
        # Generate test data
        isSeed = True
        xtest = generate_test(Zmin, Zmax, pdmin, pdmax, isPlot, Gen_Data, Sample_Size, isSeed,Seed)
        print("Number of test samples\n", len(xtest))
        #print(xtest)
        for xp in xtest:
            hp, dhp = predict(xp, sd, xtrain, Zd, param)
            print("xp =", 50*xp[0:2], xp[2:])
            plotPredTrain(hp, dhp, sd, xp, meanZd,True)
    
    #Check Prediction on a single training example
    if Check_prediction:
        xp = xtrain[np.random.randint(low=0,high=n-1,size=1).item()]
        hp, dhp = predict(xp, sd, xtrain, Zd, param)
        print("xp =", 50*xp[0:2], xp[2:])
        plotPredTrain(hp, dhp, sd, xp, meanZd,True)
    
    
    #Generate data for test_error analysis
    if Gen_Data:
        #Remove data if already exists
        Files = glob.glob(os.path.join("TestData","*.dat"))
        for file in Files:
            temp = os.remove(file)
        
        #Generate data
        for i in range(nreplica):
            if i==0:
                isSeed = True
                xtest = generate_test(Zmin, Zmax, pdmin, pdmax, isPlot, Gen_Data, Sample_Size, isSeed,Seed)
                n = len(xtest)
                np.savetxt(os.path.join("TestData","xtest_"+str(i)+".dat"), xtest, fmt="%6.2f")
            else:
                isSeed = False
                xtest = generate_test(Zmin, Zmax, pdmin, pdmax, isPlot, Gen_Data, Sample_Size, isSeed,Seed)
                np.savetxt(os.path.join("TestData","xtest_"+str(i)+".dat"), xtest, fmt="%6.2f")
                n = len(xtest)
            print("%d Test example generated"%(n))
                
              