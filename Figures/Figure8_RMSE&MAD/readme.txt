# This python script plot the RMSE and Median Absolute deviation between mean prediction and actual result.
#
# This script takes in following argument:
# 1. Folder_list: List of train data folder used for running, for example, if you want to consider n = 50,100
#                 200 and 400 you need to pass Folder_list argument as ( --Folder_list 50 100 200 400).
#
# 2. isSave    :  If true the result of RMSE and MAD analysis will be overwritten in the exisiting .txt file. 
#
# To run the script: Normal example:-
#         python plotRMSE.py --Folder_list 50 100 200 --isSave True/False
# 
# To reproduce the result:
#         python plotRMSE.py          #Default Folder_list is set to n = [50,,,,3200] (reported in paper)
#                                     #This will only plot the result from the existing .txt file
#                                     #Read instruction in __main__ section.


