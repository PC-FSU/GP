#  This script will run a deterministic profiler on trainHyper.py 
#  The script takes in two argument:
#  1. Whether to include jacobian for trainHyper.py routine
#  2. How many time a optimzation process is ran
#
#  To run the :
#   make sure you have the correct data in workspace, if not run the following two commands 
#   1. python Load_workspace.py --ClearWorkspace True
#   2. python Load_workspace.py --load True --Folder 50 or whatever
#   3. python profiler.py --nreplica 10 or 5 (or your choice)
#
# I have also attached code for running a line profiler, if you wish to use line-profiler (anothr 
# deterministi profiler), please comment the code and uncomment the section at bottom
#

from commonGP import *
from trainHyper import *
import profile

if __name__ == '__main__':
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Profiler routine")
    parser.add_argument("--include_jacobian", type=bool, default=False,
                        help='Flag to include jacobian in hyperparam or not')
    parser.add_argument("--nreplica",   type=int, default=1,
                        help='Number of time optimization routine gonna run')   

    argspar  = parser.parse_args()
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')
    
    nreplica = argspar.nreplica
    include_jacobian = argspar.include_jacobian
    
    xtrain, sd, Zd, meanZd = readTrainData()
    profile.runctx(
        'print(getBestFit(xtrain, sd, Zd, nreplica, include_jacobian)); print()',
        globals(),
        {'xtrain': xtrain,'sd':sd,'Zd':Zd,'nreplica':nreplica,'include_jacobian':include_jacobian},
    )
    
    
#######################line profiler ##########################################3
#from trainHyper import *
#from line_profiler import LineProfiler
#
#if __name__ == "__main__":
#    
#    # Parse arguments
#    parser = argparse.ArgumentParser(description="Profiler routine")
#    parser.add_argument("--include_jacobian", type=bool, default=False,
#                    help='Flag to include jacobian in hyperparam or not')
#    parser.add_argument("--nreplica",   type=int, default=1,
#                    help='Number of time optimization routine gonna run')
#
#    argspar  = parser.parse_args()
#    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
#        print('\t{}: {}'.format(p, v))
#    print('\n')
#
#    nreplica = argspar.nreplica
#    include_jacobian = argspar.include_jacobian
#
#    xtrain, sd, Zd, meanZd = readTrainData()
#    
#    lp = LineProfiler()
#    lp_wrapper = lp(getBestFit)
#    lp.add_function(objParam)
#    lp.add_function(getR1R2Inv)
#    lp.add_function(getR1R2)
#    lp_wrapper(xtrain, sd, Zd, nreplica, include_jacobian)
#    lp.print_stats()




    
