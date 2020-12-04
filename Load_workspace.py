#
#     This python script takes care of Loading, clearing, and saving data available on trainData Workspace. 
#     TrainData Workspace is workspace defined by all .dat file present in TrainData folder, not in it's
#     subfolder (50,200,1600 etc.)
#
#     The script takes in argument : 
#     1. ClearWorkspace,  Pass True if you want to remove all the .dat file from TrainData Folder.
#                        Note :- It's not going to clear any file in subfolders.  
#                        To run: pyhthon LoadWorkspace.py --ClearWorkspace True 
#
#     2. load, Pass True if you want to load data to workspace, note: In order to load to work you need to     #              pass the folder flag along with it, to specify from which subfolder to load the data.
#                        To run: pyhthon LoadWorkspace.py --load True --Folder 50 
#
#     3. SaveCurrent, Pass True if you want to save data to workspace, note: In order to saveCurrent to work   #             you need to  pass the folder flag along with it, to specify to which subfolder to save the data.
#             (Note : if you are saving to a exisiting subfolder it will first clear all the data of subfolder #             and save the current file). This is generally run to save the data file (*.dat) and hyper.dat in #             case of successful convegrnece of optization routine. 
#                        To run: pyhthon LoadWorkspace.py --SaveCurrent True --Folder 100 



import argparse
import shutil
import glob
import os

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="To load the workspace")
    parser.add_argument("--load",type=bool,default=False,
                        help="Flag to remove the current .dat file and load new .dat file in Traindata folder ")
    parser.add_argument("--Folder",type=str,default = "10",
                        help="Folder from which data need to be loaded,NOte:- Pass load flag as true otherwise an error will occur")
    parser.add_argument("--SaveCurrent", type=bool,default=False,
                        help="If true it will save the current file to Folder argument passed above")
    
    parser.add_argument("--ClearWorkspace",type=bool,default=False,
                        help="If true just remove all the .dat file from TrainFolder")
                        
    argspar = parser.parse_args()

    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')
    load = argspar.load
    Folder = argspar.Folder    
    SaveCurrent = argspar.SaveCurrent
    ClearWorkspace = argspar.ClearWorkspace
    
    
    if ClearWorkspace:
        Files = glob.glob(os.path.join("TrainData","*.dat"))
        for file in Files:
            temp = os.remove(file)

    if SaveCurrent:
        #Check if directory exists, if not create one.
        directory =  os.path.join("TrainData",Folder)
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        #Clear the destination(Where you want to save file) in case if there's file there
        if len(os.listdir(directory)) != 0:
            temp = glob.glob(os.path.join("TrainData",Folder,"*.dat"))
            for file in temp:
                os.remove(file)
        
        files  = glob.glob(os.path.join("TrainData","*.dat"))
        destination = os.path.join("TrainData",Folder)
        #save files
        for file in files:
            source = file
            shutil.copy(source, destination)

    if load:
        #remove all .dat file from TrainData folder
        #Remove file
        Files = glob.glob(os.path.join("TrainData","*.dat"))
        for file in Files:
            temp = os.remove(file)
            
        #Load new file
        files = glob.glob(os.path.join("TrainData",Folder,"*.dat"))
        destination = os.path.join("TrainData")
        for file in files:
            source = file
            shutil.copy(source, destination)
     