import pandas as pd
import glob
import warnings

#Read data
def get_frame(path = './reddit_topic_connections_day/',sort_by=['date']):
    all_files = glob.glob(path + "*.csv")

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    if 'date' in frame.columns:
        frame['date'] = pd.to_datetime(frame.date)
        frame = frame.sort_values(by=sort_by)
   
    return frame



def factors_lookup():
    sentiment_metrics=[ 'weight','ups','downs','avg_upvote_ratio','num_comments','score_total','total_awards_received_total','date','daily_popularity','roberta_Negative_sum','roberta_Negative_wt_sum','roberta_Negative_mean','roberta_Neutral_sum','roberta_Neutral_wt_sum','roberta_Neutral_mean','roberta_Positive_sum','roberta_Positive_wt_sum','roberta_Positive_mean','vader_neg_sum','vader_neg_wt_sum','vader_neg_mean','vader_neu_sum','vader_neu_wt_sum','vader_neu_mean','vader_pos_sum','vader_pos_wt_sum','vader_pos_mean',
 'vader_compound_sum','vader_compound_wt_sum','vader_compound_mean','distil_sadness_sum','distil_sadness_wt_sum','distil_sadness_mean','distil_joy_sum','distil_joy_wt_sum','distil_joy_mean','distil_love_sum','distil_love_wt_sum','distil_love_mean','distil_anger_sum','distil_anger_wt_sum','distil_anger_mean','distil_fear_sum','distil_fear_wt_sum','distil_fear_mean','distil_surprise_sum','distil_surprise_wt_sum','distil_surprise_mean','daily_popularity_pop','roberta_Negative_sum_pop','roberta_Negative_wt_sum_pop','roberta_Negative_mean_pop','roberta_Positive_sum_pop','roberta_Positive_wt_sum_pop','roberta_Positive_mean_pop','vader_neg_sum_pop','vader_neg_wt_sum_pop',
 'vader_neg_mean_pop','vader_pos_sum_pop','vader_pos_wt_sum_pop','vader_pos_mean_pop','distil_sadness_sum_pop','distil_sadness_wt_sum_pop','distil_sadness_mean_pop','distil_joy_sum_pop','distil_joy_wt_sum_pop','distil_joy_mean_pop','distil_love_sum_pop','distil_love_wt_sum_pop','distil_love_mean_pop','distil_anger_sum_pop','distil_anger_wt_sum_pop','distil_anger_mean_pop','distil_fear_sum_pop','distil_fear_wt_sum_pop','distil_fear_mean_pop','distil_surprise_sum_pop','distil_surprise_wt_sum_pop','distil_surprise_mean_pop','daily_popularity_hot','roberta_Negative_sum_hot','roberta_Negative_wt_sum_hot','roberta_Negative_mean_hot','roberta_Positive_sum_hot','roberta_Positive_wt_sum_hot','roberta_Positive_mean_hot',
 'vader_neg_sum_hot','vader_neg_wt_sum_hot','vader_neg_mean_hot','vader_pos_sum_hot','vader_pos_wt_sum_hot','vader_pos_mean_hot','distil_sadness_sum_hot','distil_sadness_wt_sum_hot','distil_sadness_mean_hot','distil_joy_sum_hot','distil_joy_wt_sum_hot','distil_joy_mean_hot','distil_love_sum_hot','distil_love_wt_sum_hot','distil_love_mean_hot','distil_anger_sum_hot','distil_anger_wt_sum_hot','distil_anger_mean_hot','distil_fear_sum_hot','distil_fear_wt_sum_hot','distil_fear_mean_hot','distil_surprise_sum_hot','distil_surprise_wt_sum_hot','distil_surprise_mean_hot','symbol',
                      'roberta_pos_vs_neg_mean_pop','joy_vs_fear_mean_pop','anger_vs_joy_mean_pop','anger_vs_sadness_mean_pop']
    
    fundamental = [ 'close','low','high','open','gspc_open','nvda_open','amd_open',
                   'SearchFrequency','nya_open','retail_sales','volumefrom','gold']
    return sentiment_metrics,fundamental



def create_combined_scaled_ta_data():
    all_scaled_data = get_frame(path = './output/scale_ata/',sort_by=['date'])
    sentiment_metrics,fundamental  = factors_lookup()
    all_scaled_data['symbol']  = all_scaled_data['sym']   
    ta_columns = (set(all_scaled_data.columns) - set(sentiment_metrics) ) | {'date','sym'}
    ta_data = all_scaled_data[ta_columns]
    ta_data.to_csv('./output/scaled_dataset.csv')
    
    all_scaled_data[sentiment_metrics+['sym']].to_csv('./input/reddit_pop_data.csv')
    
    
    
def create_combined_results_data():
    Results_data = get_frame(path = './output/model_predictions_30_days/',sort_by=['date'])
    Results_data[['date','close','symbol','is_predicted']].to_csv('./output/results.csv')
    
def create_correlation_data():
    sentiment_metrics,fundamental  = factors_lookup()
    MIC_data = get_frame(path = './output/associate_factors_df/',sort_by=['date'])
    MIC_data = MIC_data[['y','MIC','GMIC','TIC','sym','thresh']]
    MIC_data.columns = ['factors','MIC','GMIC','TIC','symbol','thresh']
    
    
    MIC_data['factor_type'] = MIC_data.factors.apply(lambda x: 'fundamental' if x in fundamental 
                                                     else ('sentiment' if x in sentiment_metrics else 'technical'))
    


    MIC_data.to_csv('./output/correlation.csv')
    

                   

        
        
if __name__=="__main__":
    
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    create_combined_scaled_ta_data()
    create_combined_results_data()
    create_correlation_data()
    
    
    