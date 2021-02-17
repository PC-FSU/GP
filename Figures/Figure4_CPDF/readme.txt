# This python script plot CP,DF with different param.
# The script takes in following argument: 
#   1. paramNum: Plot the CP,DF decomposition for sigma_2,gamma_1,gamma_21,gamma_22,gamma_23,gamma_24 for
#                paramNum 0,1,2,3,4,5,6 respectively.
#   2. Folder: (NOte: dtype = int) It will load the training data of given folder to trainData workspace. 
#
#   3. SpanLogLim_list : the x-axis limits for plooting the params.       
#
# To run: 
#   For example if you want to plot sigma_2 for folder 100 in loglimit -1 -0.4 you will pass the following 
#   command:  python plotDFCP.py --Folder 100 --paramNum 0 --SpanLogLim_list -1 -0.4
#   
# To reproduce the figure reported in paper run following command
#   For Sigma_2 :
#                  python plotDFCP.py --Folder 100 --paramNum 0 --SpanLogLim_list -4 -2
#
#   For gamma_1 : 
                   python plotDFCP.py --Folder 100 --paramNum 1 --SpanLogLim_list -1 -0.4
