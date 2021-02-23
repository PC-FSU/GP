# This python script plot the RMSE and Median Absolute deviation between mean prediction and actual result.
#
# This script takes in following argument:
# 1. Folder_list   : List of train data folder used for running, for example, if you want to consider n = 50,100
#                    200 and 400 you need to pass Folder_list argument as ( --Folder_list 50 100 200 400).
#
# 2. RunAnalysis  : If true, The RMSE and MAD will be caluclated for passed Folder_list argument and result will be plotted. If False, the result will be 
#                   plotted from the exisitng .txt files (Note: This result will be plotted for the Folder_list argument used to create the saved .txt file,
#                   not the current Folder_list argument. Note this command doesn't save the result, to save the result check out the isSave flag.
#
# 3. isSave       : If true the result of RMSE and MAD analysis will be overwritten in the exisiting .txt file, and the plot will be saved.
#
# To run the script: Normal example:-
#         python plotRMSE.py --Folder_list 50 100 200 --isSave True/False --RunAnalysis True/False
# 
# To reproduce the result:
#         python plotRMSE.py                            (Default Folder_list is set to n = [50,100,200,400,1600,3200] (reported in paper)                                                         
#                                                        This will only plot the result (Not save) from the existing .txt file)
#                                     
#
#*****************************************************.txt file description*************************************************************
#
#  Mad_h.txt ==> Contain the median absolute deviation for h(s). The shape of file => (Len(Folder_list), 1+ Len(test_data) ).
#		 
#		 Len(test_data) = 261 = Constant. We have 261 test data insted of 250, Because the script that I used to
#		 Create sample test data subject the sample drawn under same constraint as input point (M_1>M_2,..etc). 
#		 So i ran the script for 10 times with 10 different random seed, and with sample_size of 50. The script produce 50 
# 	         sample, but drop nearly half because of the constraint, so at each run we have sample ~(24-27). Therefore ~(24-27)*10=261.
#
#                The First Column contain the value of n, training set on which Analysis is ran.        
#       	 If the Folder_list argument is 50 100, the shape of Mad_h.txt will be ==> (2,1+261). 
#                
#                
#                 The current Mad_h.txt has shape ==> (7,1+261), because the used Folder_list argument for creating Mad_h.txt is
#                 [50,100,200,400,800,1600,3200] (Same as what we have in paper).
#
#
#  Mad_Phi.txt ==> Contain the median absolute deviation for Phi(t). The shape of file => (Len(Folder_list), 1+ Len(test_data) ).
#                    
#                The First Column contain the value of n, training set on which Analysis is ran. 
#                The current Mad_Phi.txt has shape ==> (7,1+261), because the used Folder_list argument for creating Mad_Phi.txt is
#                 [50,100,200,400,800,1600,3200] (Same as what we have in paper).
#
#
# 
#  RMSE_h.txt ==> Contain RMSE for h(s). The shape of file => (Len(Folder_list), 1+ Len(test_data) ).
#
#                 The First Column contain the value of n, training set on which Analysis is ran. 
#                 The current RMSE_h.txt has shape ==> (7,1+261), because the used Folder_list argument for creating RMSE_h.txt is
#                 [50,100,200,400,800,1600,3200] (Same as what we have in paper).
#  
#  
#
#
#  RMSE_Phi.txt ==> Contain RMSE for Phi(t). The shape of file => (Len(Folder_list) , 1+ Len(test_data)).
#
#                 The First Column contain the value of n, training set on which Analysis is ran. 
#                 The current RMSE_Phi.txt has shape ==> (7,1+261), because the used Folder_list argument for creating RMSE_Phi.txt is
#                 [50,100,200,400,800,1600,3200] (Same as what we have in paper).
#