# This python script plot the Dalpha for Phi and H.
#
# This script takes in following argument:
# 1. Folder_list: List of train data folder used for running, for example, if you want to consider n = 50,100
#                 200 and 400 you need to pass Folder_list argument as ( --Folder_list 50 100 200 400).
#
# 2. isSave    :  If true the result of Dalpha analysis will be overwritten in the exisiting .txt file, and the
                  plot will be saved.

# 3. RunAnalysis : If true, The Dalpha be caluclated for passed Folder_list argument and result will be plotted. If False, the result will be 
#                   plotted from the exisitng .txt files (Note: This result will be plotted for the Folder_list argument used to create the saved .txt file,
#                   not the current Folder_list argument. Note this command doesn't save the result, to save the result check out the isSave flag.
# 
#
# To run the script: Normal example:-
#         python plotDalpha.py --Folder_list 50 100 200 --isSave True/False --RunAnalysis True/False
# 
# To reproduce the result:
#         python plotDalpha.py       (Default Folder_list is set to n = [100,800,3200]                                                      
#                                                        This will only plot the result (Not save) from the existing .txt file)
#
# To Print the result for single case let's say n=53
#         python plotDalpha.py --Folder_list 100   (The result will be plotted from saved .txt files)
  
#*****************************************************.txt file description*************************************************************

Note: I am calling Xi (Greek symbol used in Dalpha notation) Eta. I also didn't drop the label, Figure you would be needing 
      while doing analysis.

# Eta_h.txt ==> Contain the Eta_alpha for h(s). The shape of file => (1 + N*n_test, Len(Folder_list)).
#		 
#		 Len(test_data) = 251 = Constant, N (grid) = 100 = Constant. Therefore 1st Dim = 1 + 25100 = Constant.
#                Note: we have 251 test data insted of 250, Because the script that I used to
#		 Create sample test data subject the sample drawn under same constraint as input point (M_1>M_2,..etc). 
#		 So i ran the script for 10 times with 10 different random seed, and with sample_size of 50. The script produce 50 
# 	         sample, but drop nearly half because of the constraint, so at each run we have sample ~(24-27). Therefore ~(24-27)*10=251.
#
#                The First Row contain the value of n, training set on which Analysis is ran.        
#       	 If the Folder_list argument is 50 100, the shape of Eta_h.txt will be ==> (1+25100,2). 
#                
#                
#                 The current Eta_h.txt has shape ==> (1+25100,7), because the used Folder_list argument for creating Eta_h.txt is
#                 [50,100,200,400,800,1600,3200].
#
#
#  Eta_Phi.txt ==> Contain the Eta_alpha for Phi(t). The shape of file => (1 + N*n_test,Len(Folder_list)).
#                    
#                The First Column contain the value of n, training set on which Analysis is ran. 
#                The current Eta_Phi.txt has shape ==> (1+25100,7), because the used Folder_list argument for creating Eta_Phi.txt is
#                 [50,100,200,400,800,1600,3200].