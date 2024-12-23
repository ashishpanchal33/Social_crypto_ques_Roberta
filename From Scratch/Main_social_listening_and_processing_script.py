


from Import_lib import *
from Get_post_data import *
from Post_Token_identifications import *
#import post_scoring_script as pss

import sentiment_analysis as sa
import generate_token_level_sentiment as gsa
import get_data_frames as gd
import calculate_pop_hot as pop_hot
import create_post_connections as post_con

subreddit_list = ['CryptoCurrency','CryptoMarkets']
duration = 31
post_limit =200
start = (2022, 3, 6)
end = (2022,4,22)


print('-> scrapping reddit_data, start:',start,'end : ',end)

####-------------------- 1.
#download reddit posts 
##### writing data at : "./redditdata/"+subreddit_+start_epoch_raw.strftime("%d_%m_%y")+".txt", sep="|",index = False)
get_post_data_and_summary(subreddit_list = subreddit_list,
                          duration = duration,
                          post_limit =post_limit, 
                          start = start, end = end)


print('-> scrapping completed')
print('-> identifying tokens')




####-------------------- 2.
############
#identify token per day and score per day
# creating data at './reddit_post_summary_month'/trend_data_'+month+'.csv')
post_frame = gd.get_post_data()
generate_Data(post_frame=post_frame)



print('-> token identification complete')
print('-> estimating post Sentiments')







####-------------------- 3.
#generate _ sentiments for the posts
######## needs : ./redditdata/*.txt
######## writing data at : "../redditdata_sentiment/"+j['date__'].iloc[0]+".txt", sep="|",index = False)
#[sa.get_emo_df(name,group) for name,group in post_frame.groupby('date__')]
sa.run_onlist_loop(post_frame)


del post_frame

print('-> Sentiments identified')
print('-> associating sentiments with tokens')





####-------------------- 4.
#get_sentiment_post_frame
token_frame = gd.get_post_summary_trend_data(path = './reddit_post_summary_month/trend_data_')
sentiment_frame = gd.get_sentiment_post_frame(path = './redditdata_sentiment/')

###############
############### 
##writes ./reddit_post_summary_day/trend_data_'+name+'.csv') 
# ----- dataframe with base metric and sentiments
gsa.run_sentiment_agg(token_frame,sentiment_frame)


del token_frame
del sentiment_frame


print('-> sentiments and tokens assocaition complete')
print('-> Calculating rolling popularity and hotness for requisite factors')

####-------------------- 5.
################
###############

token_sentiment_frame = gd.get_post_summary_trend_data(path = './reddit_post_summary_day/trend_data_')

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

##### needs : './reddit_post_summary_day/trend_data_'):
##### creating data at : "./reddit_post_score_summary_month/all_summary_"+month+".csv")

pop_hot.calculate_popularity_hotness_rising_(token_sentiment_frame,column_list =column_list)


del token_sentiment_frame

print('-> Calculation complete')
print('-> Creating connection nodes and edges and calculating connection strength')


###-----------------------6.
###########
#processing on data
# creates    ./reddit_topic_connections_day/ 
# creates    ./reddit_topic_weights_connections_day/

token_senti_summary_frame =  gd.get_frame_for_connections(path = './reddit_post_score_summary_month/all_summary_', n=300)

post_con.get_connections(token_senti_summary_frame)



del token_senti_summary_frame



print('-> Connections created')
print('#### Social listening chatter data Processing Completed #####')

