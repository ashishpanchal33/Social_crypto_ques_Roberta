import os
import warnings
import sys


if __name__=="__main__":
    
    par_arg = set(sys.argv[1:])

    
    warnings.filterwarnings("ignore")
    
    
    
    if len(par_arg)>0:
        
        par_arg_to_run = {'1','2','3','4','5','6','7'} & par_arg
        
        unknown_arg = par_arg - {'1','2','3','4','5','6','7'}
        if len(unknown_arg)>0 or len(par_arg_to_run) ==0 :
            print('following processes does not exist , and will not run:',unknown_arg )
            print('please use the following codes for the corresponding processes')
            print('->  1 run social listening script')
            print('->  2 fetch and save price info and fundamental factors')
            print('->  3 run clustering script')
            print('->  4 run prediction script')
            print('->  5 create, saving ADS and save correlation for Tableau')
            print('->  6 create network data for Tableau')
            print('->  7 create post data for Tableau')
            print('\n example : python social_crypto_cues.py 1 2 3')
               
            
    
        if '1' in par_arg_to_run:
            print('1 running social listening script')
            os.system("python Main_social_listening_and_processing_script.py")

        if '2' in par_arg_to_run:
            print('2 fetch and save price info and fundamental factors')
            os.system("python fetch_price_info_and_fundamental_factors.py")

        if '3' in par_arg_to_run:
            print('3 running clustering script')
            os.system("python clustering.py")    

        if '4' in par_arg_to_run:
            print('4 running prediction script')
            os.system("python Price_prediction_xgboost.py")    


        if '5' in par_arg_to_run:
            print('5 create, saving ADS and save correlation for Tableau')
            os.system("python Create_tableau_required_trends_data.py")

        if '6' in par_arg_to_run:
            print('6 creating network data for Tableau') 
            os.system("python Network_builder.py")    

        if '7' in par_arg_to_run:
            print('7 creating post data for Tableau') 
            os.system("python create_post_text_tableau_data.py")


    
    else:
            print('running complete process ---')
            print('running clustering script')
            os.system("python Main_social_listening_and_processing_script.py.py")

            print('running clustering script')
            os.system("python fetch_price_info_and_fundamental_factors.py.py")

            print('running clustering script')
            os.system("python clustering.py")    

            print('running prediction script')
            os.system("python Price_prediction_xgboost.py")    


            print('create, saving ADS and save correlation for Tableau')
            os.system("python Create_tableau_required_trends_data.py")

            print('creating network data for Tableau') 
            os.system("python Network_builder.py")    

            print('creating post data for Tableau') 
            os.system("python create_post_text_tableau_data.py")


    
    
    
        
    
        
