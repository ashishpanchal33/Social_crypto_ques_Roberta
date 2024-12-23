import pandas as pd

column_list = ['daily_popularity', 'roberta_Negative_sum',
       'roberta_Negative_wt_sum', 'roberta_Negative_mean',
       'roberta_Positive_sum', 'roberta_Positive_wt_sum',
       'roberta_Positive_mean', 'vader_neg_sum', 'vader_neg_wt_sum',
       'vader_neg_mean',
       'vader_pos_sum', 'vader_pos_wt_sum', 'vader_pos_mean',
       'distil_sadness_sum', 'distil_sadness_wt_sum', 'distil_sadness_mean',
       'distil_joy_sum', 'distil_joy_wt_sum', 'distil_joy_mean',
       'distil_love_sum', 'distil_love_wt_sum', 'distil_love_mean',
       'distil_anger_sum', 'distil_anger_wt_sum', 'distil_anger_mean',
       'distil_fear_sum', 'distil_fear_wt_sum', 'distil_fear_mean',
       'distil_surprise_sum', 'distil_surprise_wt_sum',
       'distil_surprise_mean']




def pop_assist_2(data,en,cols=column_list):
    
    #en = data.date.max()
    #print(data.shape)
    if 'token' in list(data.columns) and any(en == data['date']) :
        #if(len(data)>1):
        #    display(data[['date','token']],en)#[data['date'].values == en])
        token_dict = data[data['date'].values == en].reset_index(drop=True).T.to_dict()[0]
        #display(token_dict)
        #print(a)
        #display(data)
        for i in  cols:
            #print(i)
            #display(data[i])
            token_dict[i+"_pop"] = data[i].values.sum()/30

        #hotness

        three_days_delta = en - pd.Timedelta("4 days")
        data_3_day = data[ (data['date'].values >three_days_delta) &  (data['date'].values <=en)]

        for i in  cols:
            token_dict[i+"_hot"] = data_3_day[i].values.sum()/(4*3)  

        #print(token_dict)
        return token_dict

    
    
    
    

def calculate_popularity_hotness_rising_(data,column_list =column_list):
    #month= data['month__'].head(1)
    data = data[data.token.isna()==False]
    _month_set = data.month__.unique()

    
    
    def run_for_one(i):
        day_list = []
        
        
        one_month_start = i - pd.Timedelta("30 days")
        #print(i)
        data_month = data[ (data['date'].values >one_month_start) &  (data['date'].values <=i)]
        
        #day_list = 
        
        return(pd.DataFrame.from_records([ j for j in  [pop_assist_2(group,i) for name,group in data_month.groupby('token')]
                                         if j!=None]))
                
        
    for month in _month_set:
        _ate_set = set(data[data.month__ == month].date.to_list())    
        data_month_results = pd.concat([run_for_one(i) for i in list(_ate_set)],ignore_index=True)


            #data_month_results.append(data_month.groupby('token').apply(
            #                                            pop_assist_2,i)
            #                                              )

        data_month_results.round(3).to_csv("./reddit_post_score_summary_month/all_summary_"+month+".csv")
    #return data_month_results #(pd.concat(data_month_results,ignore_index=True, axis =0)),
        
    
    
    
    
    