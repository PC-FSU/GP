
#   
#  This script takes argument:
#    1. Sample_Size : Number of sample tha you want to draw from hypercubic sampling for test data
#          Note: The generated Sample are then subjected to constrained offer by our problem, like M1>M2,
#                This lead to elimination of half of the genearted sample, so in reality we actual size is 
#                ~Sample_Size/2.
#    2. Param_Value: "value for (Z_min,Z_max,pdmin,pdmax), fixed for our case, but in case."

#########Flags to generate the test data for performing error analysis with different xtrain size,n#########
#    3. nreplica: Number of Copy of xtest data of size Sample_Size, you need to generate.
#                
#    4. isSave:  "If True, Save the data, and remove the existing .data files"
#
#    5. isPlot:  "IF true, plot the drawn samples"
#
#
#  How to run:
#            python genTest.py --Sample_Size 50 --isPlot True/False --isSave True/False --nreplica int_N
#    


from predictGP import *
import pyDOE as doe
import glob 


def generate_test(Zmin=5., Zmax=50., pdmin=1.01, pdmax=1.50, isPlot=False, Sample_Size = 10):
    
    Seed = np.random.randint(low=0,high=1e3,size=(1,))
    np.random.seed(int(Seed))
    
    xtest = doe.lhs(5, samples=Sample_Size)   #Draw Hypercubic samples
    
    xtest[:,0] = Zmin + (Zmax-Zmin)*xtest[:,0]     #scale m1 to have value between Zmin and Zmax
    xtest[:,1] = Zmin + (Zmax-Zmin)*xtest[:,1]     #scale m2 to have value between Zmin and Zmax
    
    xtest[:,2] = pdmin + (pdmax-pdmin)*xtest[:,2]  
    xtest[:,3] = pdmin + (pdmax-pdmin)*xtest[:,3]
    
    xtest = xtest[xtest[:,0] > xtest[:,1]]                 # constraint M1>M2
    
    xtest = xtest[xtest[:,0]>0]                        # Constraint that M1>0 and M2>0
    xtest = xtest[xtest[:,1]>0]
    xtest = xtest[xtest[:,4] > 0.0001]                     # w1>0
    
     #Since we are working with data value between 0.1 (Z_min = 5) and 1,(Z_max = 50).Note: Zmin doesn't correspond to 0, because on training routine we only kept the data in range (5,50) and divide that by 50, so input range was 0.1 and 1.
    xtest[:,0:2] = xtest[:,0:2]/Zmax
    
    if isPlot:
        plt.subplots(1,3, figsize=(15,5))

        plt.subplot(131)
        plt.plot(xtest[:,0]*Zmax, xtest[:,1]*Zmax,'s', alpha=0.5)
        plt.axis('equal')
        plt.xlabel(r'$Z_1$')
        plt.ylabel(r'$Z_2$')
        plt.tight_layout()

        plt.subplot(132)
        plt.plot(xtest[:,0]*Zmax, xtest[:,4],'s', alpha=0.5)
        plt.xlabel(r'$Z_1$')
        plt.ylabel(r'$w_1$')

        plt.subplot(133)
        plt.plot(xtest[:,2], xtest[:,3],'s', alpha=0.5)
        plt.xlabel(r'$\rho_1$')
        plt.ylabel(r'$\rho_2$')
        
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    return xtest
    

### MAIN ROUTINES ###

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Generate test data")

    parser.add_argument("--Sample_Size", type=int, default=20,
                        help='Number of sample tha you want to draw from hypercubic sampling for test data')
    
    parser.add_argument("--Param_Value", type = int, nargs="+", default=[5, 50, 1.01, 1.50],
                       help = "value for (Z_min,Z_max,pdmin,pdmax), fixed for our case, but in case.")

    parser.add_argument("--isPlot", type=bool, default=False,
                        help='Flag to plot the drawn hypercubic test sample')
    
    #Flags to generate the test data for performing error analysis with differbt xtrain size,n.
    
    parser.add_argument("--nreplica",type=int,default=1,
                        help = "Number of Copy of xtest data of size Sample_Size, you need to generate")
    
    parser.add_argument("--isSave",type=bool,default=False,
                        help="If True, Save the data, and remove the existing .data files")
    
    argspar = parser.parse_args()

    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')


    isPlot      = argspar.isPlot
    Sample_Size = argspar.Sample_Size
    Zmin,Zmax,pdmin,pdmax = argspar.Param_Value
    nreplica    =  argspar.nreplica
    isSave      =  argspar.isSave
           
    #Generate data
    for i in range(nreplica):        
        xtest = generate_test(Zmin, Zmax, pdmin, pdmax, isPlot, Sample_Size)
        n = len(xtest)
        print("%d Test example generated"%(n))
        print(xtest)
        if isSave==True:
            flag = os.path.exists(os.path.join("TestData","xtest_"+str(i)+".dat"))
            if flag == True:
                inp = input("File already exist, to rewrite type y/n  : "  )
                if(inp == "y" or inp == "Y"):
                    os.remove(os.path.join("TestData","xtest_"+str(i)+".dat"))
                    np.savetxt(os.path.join("TestData","xtest_"+str(i)+".dat"), xtest, fmt="%6.2f")
            else:
                np.savetxt(os.path.join("TestData","xtest_"+str(i)+".dat"), xtest, fmt="%6.2f")