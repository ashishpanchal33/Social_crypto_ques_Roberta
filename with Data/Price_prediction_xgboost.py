import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from ta import add_all_ta_features
from ta.utils import dropna
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import xgboost as xg
from sklearn.metrics import mean_squared_error as MSE
import plotly
import plotly.express as px
from pandas import read_csv
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import glob
import datetime
import pdrle
from joblib import parallel_backend


from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
import json

import numpy as np
from minepy import MINE
from scipy.signal import savgol_filter
import warnings

def get_pred_frame():
    path = './reddit_post_score_summary_month/'#'./reddit_post_summary/trend_data_' # use your path
    all_files = glob.glob(path + "*.csv")
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    frame['date'] = pd.to_datetime(frame.date)
    frame = frame.sort_values(by=['token','date'])
   
    return frame



def get_token_list():
    clusters=pd.read_csv('input/cluster_info.csv')
    tokens= [x.lower() for x in list(clusters['name'])+list(clusters['symbol'])]
    return tokens




def sentiment_factors():
    sentiment_metrics=[ 'weight','ups','downs','avg_upvote_ratio','num_comments','score_total','total_awards_received_total','date','daily_popularity','roberta_Negative_sum','roberta_Negative_wt_sum','roberta_Negative_mean','roberta_Neutral_sum','roberta_Neutral_wt_sum','roberta_Neutral_mean','roberta_Positive_sum','roberta_Positive_wt_sum','roberta_Positive_mean','vader_neg_sum','vader_neg_wt_sum','vader_neg_mean','vader_neu_sum','vader_neu_wt_sum','vader_neu_mean','vader_pos_sum','vader_pos_wt_sum','vader_pos_mean',
 'vader_compound_sum','vader_compound_wt_sum','vader_compound_mean','distil_sadness_sum','distil_sadness_wt_sum','distil_sadness_mean','distil_joy_sum','distil_joy_wt_sum','distil_joy_mean','distil_love_sum','distil_love_wt_sum','distil_love_mean','distil_anger_sum','distil_anger_wt_sum','distil_anger_mean','distil_fear_sum','distil_fear_wt_sum','distil_fear_mean','distil_surprise_sum','distil_surprise_wt_sum','distil_surprise_mean','daily_popularity_pop','roberta_Negative_sum_pop','roberta_Negative_wt_sum_pop','roberta_Negative_mean_pop','roberta_Positive_sum_pop','roberta_Positive_wt_sum_pop','roberta_Positive_mean_pop','vader_neg_sum_pop','vader_neg_wt_sum_pop',
 'vader_neg_mean_pop','vader_pos_sum_pop','vader_pos_wt_sum_pop','vader_pos_mean_pop','distil_sadness_sum_pop','distil_sadness_wt_sum_pop','distil_sadness_mean_pop','distil_joy_sum_pop','distil_joy_wt_sum_pop','distil_joy_mean_pop','distil_love_sum_pop','distil_love_wt_sum_pop','distil_love_mean_pop','distil_anger_sum_pop','distil_anger_wt_sum_pop','distil_anger_mean_pop','distil_fear_sum_pop','distil_fear_wt_sum_pop','distil_fear_mean_pop','distil_surprise_sum_pop','distil_surprise_wt_sum_pop','distil_surprise_mean_pop','daily_popularity_hot','roberta_Negative_sum_hot','roberta_Negative_wt_sum_hot','roberta_Negative_mean_hot','roberta_Positive_sum_hot','roberta_Positive_wt_sum_hot','roberta_Positive_mean_hot',
 'vader_neg_sum_hot','vader_neg_wt_sum_hot','vader_neg_mean_hot','vader_pos_sum_hot','vader_pos_wt_sum_hot','vader_pos_mean_hot','distil_sadness_sum_hot','distil_sadness_wt_sum_hot','distil_sadness_mean_hot','distil_joy_sum_hot','distil_joy_wt_sum_hot','distil_joy_mean_hot','distil_love_sum_hot','distil_love_wt_sum_hot','distil_love_mean_hot','distil_anger_sum_hot','distil_anger_wt_sum_hot','distil_anger_mean_hot','distil_fear_sum_hot','distil_fear_wt_sum_hot','distil_fear_mean_hot','distil_surprise_sum_hot','distil_surprise_wt_sum_hot','distil_surprise_mean_hot','symbol']
    return sentiment_metrics













def fetch_and_create_base_data():

    
    data=pd.read_csv('input/dataset.csv')
    data['date'] = pd.to_datetime(data['date'])
    data['day_of_month']=data['date'].dt.day
    data['weekday']=data['date'].dt.weekday
    
    pop_data=get_pred_frame()
    
    
    tokens= get_token_list()
    
    frame=pop_data[pop_data['token'].isin(tokens)]
    
    
    
    #read_symbol_name_mapping
    mapping=pd.read_csv('input/symbol_name_mapping.csv')
    
    
    
    
    #add symbol mapping and select token
    sym_pop_data=frame.merge(mapping, left_on='token', right_on='token', how='left')

    lookup_sym = sym_pop_data.groupby(["token","symbol"]).sum(["daily_popularity_pop"]).reset_index().sort_values("daily_popularity_pop", ascending = False).groupby(["symbol"]).head(1)[['symbol','token']]
    lookup_sym.columns = ['symbol','selecte_token']
    lookup_sym.reset_index(inplace=True)
    sym_pop_data= pd.merge(sym_pop_data,lookup_sym, how= 'left',on='symbol')
    sym_pop_data=sym_pop_data.loc[sym_pop_data['selecte_token']==sym_pop_data.token]
    
    sym_pop_data=sym_pop_data[sentiment_factors()]
    

    sym_pop_data['date']=pd.to_datetime(sym_pop_data['date'])
    sym_pop_data['symbol']=sym_pop_data['symbol'].str.upper()
    
    
    return sym_pop_data,data
    


    
#Make model
def make_model(df1,model,ratio,rp=1,param=None):
    df1.replace([np.inf, -np.inf], np.nan, inplace=True)
    df1=df1.fillna(0)
    x = df1[[i for i in df1.columns if i != 'close_y']]
    y = df1['close_y']
    
    num_test = int(len(df1)*ratio)
    x_train = pd.concat([x[:-num_test] for i in range(rp)])#,ignore_index=True)
    y_train = pd.concat([y[:-num_test] for i in range(rp)])#,ignore_index=True) 
    x_test = x[-num_test:]
    y_test = y[-num_test:]



    # create regressor object
    if (model=='random_forest'):
        regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0)
        
    elif(model=='xgboostWParam'):
        regressor = xg.XGBRegressor(objective =param['objective']
                                        ,n_estimators = 1000, random_state = 123, reg_alpha = param['l1'],
                                    reg_lambda = param['l2'],
                                   learning_rate=param['learning_rate'],
                                    max_depth = param['max_depth'],
                                    min_child_weight = param['min_child_weight'],
                                    eval_metric=param['eval_metric']
                                   )
    elif(model=='xgboostWParam_na'):
        regressor = xg.XGBRegressor(objective =param['objective']
                                        ,n_estimators = 1000, random_state = 123, reg_alpha = param['l1'],
                                    reg_lambda = param['l2'],
                                   max_depth = param['max_depth'],
                                    min_child_weight = param['min_child_weight'],
                                    eval_metric=param['eval_metric'])    
        
    elif(model=='linear'):
        regressor = xg.XGBRegressor(objective ='reg:linear',
                      n_estimators = 10, seed = 123)


        # fit the regressor with x and y data
    regressor.fit(x_train, y_train) 
    y_pred = regressor.predict(x_test)
    

    mean_absolute_error= metrics.mean_absolute_error(y_test, y_pred)
    mean_squared_error=metrics.mean_squared_error(y_test, y_pred)
    root_mean_squared_error=np.sqrt(metrics.mean_squared_error(y_test, y_pred))

    
    #Comparing actual and predicted value
    comp=y_test.reset_index()
    comp['pred']=y_pred
    comp.set_index('date')
    
    ret = (comp,mean_absolute_error,mean_squared_error,root_mean_squared_error,x_train,y_train,x_test,y_pred)  

    return  ret
    
    
    
    
def print_stats(mine,df,x,y):
    mine.compute_score(df[x],df[y] )
    dict_={ "x":x,
           "y":y,
    "MIC": mine.mic(),
    #,"MAS": mine.mas()
    #,"MEV": mine.mev()
    #,"MCN (eps=0)": mine.mcn(0)
    #,"MCN (eps=1-MIC)": mine.mcn_general()
    "GMIC": mine.gmic()
    ,"TIC": mine.tic(norm=True)
    #,'score':mine.get_score()
          }
    
    return dict_


def default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError('Not serializable')

    
    
    
    
def smooth_data (df1):
    df1.fillna(0,inplace =True) 
    non_num = [df1.columns[i]  for i,j in enumerate(df1.dtypes) if j not in (np.int64,np.float64) or df1.columns[i] in ['close','high','low','open','close_y' ] ]

    df_non_num = df1[non_num]
    df_num = df1[list(set(df1.columns) - set(non_num))]

    for i in list(set(df1.columns) - set(non_num)):
        df_num[i] = savgol_filter(df_num[i], 30, 3)

    return pd.concat([df_non_num,df_num],axis=1)



def filter_and_scale(df,sym,date,param, verbose=0):
    if verbose != 0:
        print("columns :",df.columns)
        print("params :",param)
    if 'sym' in df.columns:
        df=df.loc[df['sym']==sym]

    #Select columns and date range
    if 'date' in df.columns:
        
        df1=df[df['date']<=date]
        print ([i  for i,j in enumerate (df1.columns) if j == 'date'])
        df1.set_index('date' , inplace=True)
        df1 = df1[param]
    else:
        df1=df[param]
        
               
        
    #Scale parameters
    std_scaler = MinMaxScaler()#StandardScaler()
    df_scaled = std_scaler.fit_transform(df1[param].to_numpy())
    df_scaled = pd.DataFrame(df_scaled, columns=param)
    df_scaled.index=df1.index
    return(df_scaled,std_scaler)




def shift_by_n(df,n,delta = 1):
    #Shift dataframe by n
    if delta ==1:
        df['close_y'] = df['close'].shift(periods = -n,fill_value = 0) - df['close']
        
        #finding delta of each column
        for i in set(df.columns) - set(["close_y"]):
            df[i] =  df[i] - df[i].shift(periods = 1,fill_value = 0)            
        df=df[1:-n]
        
    elif delta ==2:
        df['close_y'] = df['close'].shift(periods = -n,fill_value = 0)# - df['close']
        
        #finding delta of each column
        for i in set(df.columns) - set(["close_y"]):
            df[i+"_change"] =  df[i] - df[i].shift(periods = 1,fill_value = 0)            
        df=df[1:-n]        
    else: 
        df['close_y'] = df['close'].shift(periods = -n,fill_value = 0)    
    #drop n rows
        #df=df[:-n]

    
    return(df)



#Include all technical indicators

def apply_ta_features(df_scaled):
    pd.set_option('display.max_columns', None)

#    df_scaled = dropna(df_scaled)
    df_scaled = add_all_ta_features(
        df_scaled, open="open", high="high", low="low", close="close", volume="volumefrom")
    df_scaled = df_scaled.fillna(0)
    df_scaled.tail()
    return(df_scaled)




def main_grid_search_and_prediction_process():
    Gri_best_ = []
    pre_best = []
    error = []
    mine = MINE(alpha=0.6, c=15, est="mic_approx")

    sym_pop_data,data = fetch_and_create_base_data()
    sentiment_metrics = sentiment_factors()
    
    
    for i in list(set(data['sym'].unique()) ):#['BTC','ETH','SHIB']:#

        if True:

            df = data[data['sym'] ==i].sort_values(by='date')




            sym=i

            df=df.merge(sym_pop_data, left_on=['sym','date'], right_on=['symbol','date'], how='left')

            param=['high', 'low', 'open', 'volumefrom',
                   'close', 
                   'SearchFrequency', 'amd_open', 'nvda_open', 'retail_sales', 'gspc_open', 'nya_open', 'gold', 'day_of_month', 'weekday']

            #df[sentiment_metrics]=df[sentiment_metrics].fillna(method='bfill')

            l=param+sentiment_metrics
            l.remove('symbol')
            l.remove('date')
            l.append('btc_open')

            if (sym!='BTC'):
                btc=data[data['sym']=='BTC']
                df=df.merge(btc[['date','open']].rename({'open': 'btc_open'}, axis=1), left_on='date', right_on='date', how='left')

            else:
                df['btc_open']=0


            df['roberta_pos_vs_neg_mean_pop'] = df.roberta_Positive_mean_pop - df.roberta_Negative_mean_pop #_
            df['joy_vs_fear_mean_pop'] = df.distil_joy_mean_pop - df.distil_fear_mean_pop #
            df['anger_vs_joy_mean_pop'] = df.distil_anger_mean_pop - df.distil_joy_mean_pop #
            df['anger_vs_sadness_mean_pop'] = df.distil_anger_mean_pop - df.distil_sadness_mean_pop #


            l += ['roberta_pos_vs_neg_mean_pop','joy_vs_fear_mean_pop','anger_vs_joy_mean_pop','anger_vs_sadness_mean_pop']

            for j in sentiment_metrics: # taking a sqrt of weighte_ sentiment social factors
                if "wt_sum_" in j:
                    df[j] = np.sqrt(df[j].to_list())





            df = smooth_data(df)

            t5q = df.close.mean()/2.3
            #t5q = df.close.min()#/2.3
            df = df[df.date > df[(df.close >= t5q)].sort_values(by='date').date.values[0]] 


            df['sym']=sym
            dfcsv=df[param]
            dfcsv['symbol']=sym
            #if (sym=='BTC'):
            #    dfcsv.to_csv('output/scale_ata/'+sym+'.csv')
            #else:
            dfcsv.reset_index().to_csv('output/unscale_ata/'+sym+'.csv')#, mode='a', index=True, header=False)
            df.reset_index().to_csv('output/scale_ata/'+sym+'.csv')#, mode='a', index=True, header=False)

            df2_base=apply_ta_features(df[l+['date','sym']])
            df2,std_scaler=filter_and_scale(df2_base
                                      ,sym,'2022-03-26',list (set( df2_base.columns )-{'date','sym'}), verbose =0)

            dfcsv.reset_index().to_csv('output/unscale_ata/'+sym+'.csv')#, mode='a', index=True, header=False)
            df3 = df2.copy()
            df3['sym']=sym
            df3.reset_index().to_csv('./output/scale_ata/'+sym+'.csv')#, mode='a', index=True, header=False)

            df2_extra=shift_by_n(df2,30,delta=3)  
            n = 30
            df2=df2_extra[:-n]

            gmic_thresh = 0.5 #75
            res = df2[list(set(df2.columns) - {'sym'})].apply(lambda x: print_stats(mine,df2,"close_y",x.name), axis =0)
            associate_columns = list(set( ['close']+[ i for i in list(set(pd.DataFrame.from_records(res.to_list()).query('GMIC >={} '.format(gmic_thresh)).y.to_list()))]))

            while len(associate_columns)<5:
                gmic_thresh -=0.05
                associate_columns =list(set( ['close']+[ i for i in list(set(pd.DataFrame.from_records(res.to_list()).query('GMIC >={} '.format(gmic_thresh)).y.to_list()))]))

            associat_f = pd.DataFrame.from_records(res.to_list())
            associat_f['sym'] = sym
            associat_f['thresh'] = gmic_thresh
            associat_f.to_csv("output/associate_factors_df/"+sym+'.csv')

            print(df2.index)
            asso_fig =px.line(df2.sort_values(by='date'), y = associate_columns,template = 'none')
            asso_fig.update_traces(mode="lines",hovertemplate=None)
            asso_fig.update_layout(hovermode="x")

            plotly.offline.plot(asso_fig, filename='output/associate_factors_html/'+sym+'.html',auto_open=False)

            ##grid search

            df2,std_scaler_2=filter_and_scale(df2_base
                                      ,sym,'2022-03-26', list( set(associate_columns) -{'date','sym','close_y'})   )



            #df2=shift_by_n(df2,30,delta=3)          

            df2_extra=shift_by_n(df2,30,delta=3)  

            df2=df2_extra[:-n]


            df1=df2.copy()          
            df_o = df1.copy()





            #associat_f.to_csv("output/associate_factors_df/"+sym+'.csv')





            ######################################
            ######**************##################
            ###### GRID SEARCH
            ######*************###################
            df_o_2=df_o[associate_columns].copy()
            rp=1
            
            
            
            num_test = int(len(df_o_2)*0.05)
            x_gri = df_o_2[[o for o in df_o_2.columns if o != 'close_y']]
            y_gri = df_o_2['close_y']


            x_train_gri = pd.concat([x_gri[:-num_test] for p in range(rp)],ignore_index=True)
            y_train_gri = pd.concat([y_gri[:-num_test] for p in range(rp)],ignore_index=True) 

            
            
            #defining grid param ranges
            
            if sym in ["BTC","ETH"]:
                learning_rate = [0.45,0.3,0.2,0.05]  
                reg_alpha_list = [0.005]+[i/1000 for i in range(50,500,60)]
                reg_lambda_list = [0.3,0.1,0.01, 0.001, 0.0001]
                max_depth_list = [6]
                min_child_weight_list = [2] 
            else:
                learning_rate = [0.45,0.2,0.03]  
                reg_alpha_list = [0.005]+[i/1000 for i in range(50,500,80)]
                reg_lambda_list = [0.1,0.01, 0.001, 0.0001]
                max_depth_list = [6]
                min_child_weight_list = [2]    


              
                 



            #creating param grid
            
            param_grid=dict(learning_rate=learning_rate, 
                            objective =['reg:squarederror'],
                            eval_metric = ['rmse'],


                            n_estimators = [1000],
                            random_state = [123],
                            tree_method = ['gpu_hist'],
                            predictor=['gpu_predictor'],
                            max_depth = max_depth_list,
                            min_child_weight = min_child_weight_list,
                            reg_alpha = reg_alpha_list,
                            reg_lambda = reg_lambda_list)

            #creating grid object
            
            gsc = GridSearchCV(
                        estimator=xg.XGBRegressor(),
                        param_grid=param_grid,
                        cv=5, scoring='neg_root_mean_squared_error'#quartic_error#'neg_mean_squared_error'
                    , verbose=1,n_jobs=-1)


            
            def fit_grid(): # for process parallalism with multithreading
                with parallel_backend('threading'):#
                    grid_result = gsc.fit(x_train_gri, y_train_gri)
                    return grid_result
            # run grid search with multithreading
            grid_result = fit_grid()

            
            with open("output/grid_results/"+sym+".json", 'w', encoding='utf-8') as f:
                json.dump(grid_result.cv_results_, f, ensure_ascii=False, indent=4,default=default)




            #saving best params
            if len(Gri_best_) == 0:
                Gri_best_ = pd.DataFrame.from_records([{**(grid_result.best_params_),"sym":sym}])

            else:
                Gri_best_ = pd.concat([Gri_best_,pd.DataFrame.from_records([{**(grid_result.best_params_),"sym":sym}])],ignore_index=True)

            Gri_best_.to_csv("output/grid_results_best_param/"+sym+".csv")


            print(sym,grid_result.best_params_)

            
            
            
            
            
            ####################################################
            ########### GRID SEARCH COMPLETE ###################
            ####################################################
            
            
            
            
            ####################################################
            ########## testing best params --------------------->
            ####################################################




            
            param_grid={ (l if 'reg' not in l else ('l1' if 'alpha' in l else 'l2') ):k   for l,k in grid_result.best_params_.items()}
            param_grid['tree_method'] ='gpu_exact'
            
            #train test split ratio
            ratio_val=0.05  
            
            #running ,model
            comp,mean_absolute_error,mean_squared_error,root_mean_squared_error,x_train,y_train,x_test,y_pred = make_model(df_o[list(set(df1.columns) - set(["sym"]))]
                                                                                             ,model='xgboostWParam',ratio=ratio_val,param=param_grid)


            #saving test scores
            if len(pre_best) == 0:
                pre_best = pd.DataFrame.from_records([{**(grid_result.best_params_),'mse':mean_squared_error
                                                       ,'mae':mean_absolute_error
                                                       ,'rmse':root_mean_squared_error
                                                       ,"sym":sym}])

            else:
                pre_best = pd.concat([pre_best,pd.DataFrame.from_records([{**(grid_result.best_params_),'mse':mean_squared_error
                                                       ,'mae':mean_absolute_error
                                                       ,'rmse':root_mean_squared_error
                                                       ,"sym":sym}])],ignore_index=True)

            pre_best.to_csv("output/pre_best/"+sym+".csv")




            ##### unscalling train and test data
            
            
            x_test['close']=y_pred

            associate_columns_no_cy = [i for i in associate_columns if i !="close_y"]

            testunscaled=std_scaler_2.inverse_transform(x_test[associate_columns_no_cy].to_numpy())
            testunscaled=pd.DataFrame(testunscaled, columns=associate_columns_no_cy)
            testunscaled.index=x_test.index
            #testunscaled.to_csv('output/test_unscaled_data.csv')

            x_train['close']=y_train
            trainunscaled=std_scaler_2.inverse_transform(x_train[associate_columns_no_cy].to_numpy())
            trainunscaled=pd.DataFrame(trainunscaled, columns=associate_columns_no_cy)
            trainunscaled.index=x_train.index
            x_train[[i for i in param if i in associate_columns ]]=trainunscaled[[i for i in param if i in associate_columns ]]
            #x_train.to_csv('output/train_unscaled_data.csv')

            testunscaled['symbol']=sym
            testunscaled['is_predicted']=1
            trainunscaled['symbol']=sym
            trainunscaled['is_predicted']=0

            results=pd.concat([trainunscaled, testunscaled])
            
            results.index=results.index+ datetime.timedelta(days=30)
            
            ## savint test results
            results.to_csv("output/model_predictions/"+sym+".csv")

            comp.to_csv("output/model_predictions_comparison/"+sym+".csv")


            #results['date']=results.index

            # creating test plot
            fig=px.line(comp, x='date', y=['close_y','pred'])
            fig.update_traces(mode="markers+lines",hovertemplate=None)
            fig.update_layout(hovermode="x")   

            #saving test plot
            plotly.offline.plot(fig, filename='output/prediction_html/'+sym+'.html',auto_open=False) 







            ########****************************#######
            ########----------------------------#######
            ####### ||---------Prediction-----||#######
            ########----------------------------#######
            #######*****************************#######

            # creating train and pred split
            ratio_val=30/len(df2_extra)
            #running model
            comp,mean_absolute_error,mean_squared_error,root_mean_squared_error,x_train,y_train,x_test,y_pred = make_model(df2_extra[list(set(df1.columns) - set(["sym"]))]#[top_param_list]
                                                                                             ,model='xgboostWParam',ratio=ratio_val,param=param_grid)


            
            if len(pre_best) == 0:
                pre_best = pd.DataFrame.from_records([{**(grid_result.best_params_),'mse':mean_squared_error
                                                       ,'mae':mean_absolute_error
                                                       ,'rmse':root_mean_squared_error
                                                       ,"sym":sym}])

            else:
                pre_best = pd.concat([pre_best,pd.DataFrame.from_records([{**(grid_result.best_params_),'mse':mean_squared_error
                                                       ,'mae':mean_absolute_error
                                                       ,'rmse':root_mean_squared_error
                                                       ,"sym":sym}])],ignore_index=True)

            pre_best.to_csv("output/pre_best_30_days/"+sym+".csv")





            x_test['close']=y_pred
            #x_test['close_y']=y_pred

            associate_columns_no_cy = [i for i in associate_columns if i !="close_y"]

            testunscaled=std_scaler_2.inverse_transform(x_test[associate_columns_no_cy].to_numpy())
            testunscaled=pd.DataFrame(testunscaled, columns=associate_columns_no_cy)
            testunscaled.index=x_test.index
            #testunscaled.to_csv('output/test_unscaled_data.csv')

            x_train['close']=y_train
            #x_train['close_y']=y_train
            trainunscaled=std_scaler_2.inverse_transform(x_train[associate_columns_no_cy].to_numpy())
            trainunscaled=pd.DataFrame(trainunscaled, columns=associate_columns_no_cy)
            trainunscaled.index=x_train.index
            x_train[[i for i in param if i in associate_columns ]]=trainunscaled[[i for i in param if i in associate_columns ]]
            #x_train.to_csv('output/train_unscaled_data.csv')

            testunscaled['symbol']=sym
            testunscaled['is_predicted']=1
            trainunscaled['symbol']=sym
            trainunscaled['is_predicted']=0

            results=pd.concat([trainunscaled, testunscaled])

            results.index=results.index+ datetime.timedelta(days=30)
            results.to_csv("output/model_predictions_30_days/"+sym+".csv")

            comp.to_csv("output/model_predictions_comparison_30_days/"+sym+".csv")


            #results['date']=results.index


            fig=px.line(comp, x='date', y=['close_y','pred'])
            fig.update_traces(mode="markers+lines",hovertemplate=None)
            fig.update_layout(hovermode="x")   


            plotly.offline.plot(fig, filename='output/prediction_html_30_days/'+sym+'.html',auto_open=False) 




if __name__=="__main__":
    
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    main_grid_search_and_prediction_process()
    
    
    


