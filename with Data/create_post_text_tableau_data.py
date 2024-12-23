#Read post texts


import json
import pandas as pd
from pandas.io.json import json_normalize
import datetime

import numpy as np
import datetime as dt

import warnings
import glob





def get_post_text_data(path = './redditdata_sentiment/'):
     # use your path
    all_files = glob.glob(path + "*.txt")# "*01_01_22.txt")

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0, sep = "|")
        li.append(df)

    post_frame = pd.concat(li, axis=0, ignore_index=True)
    post_frame = post_frame.drop_duplicates()
    post_frame['date__'] = pd.to_datetime(post_frame['date'].apply(lambda x: x[:19])).apply(lambda x :x.strftime("%d_%m_%y"))
    #post_frame['date__']
    post_frame['month__'] = post_frame['date__'].apply( lambda x: x[-5:])
   
    return post_frame

def get_connection_frame(path = './reddit_topic_connections_day/',sort_by=['token_1','date']):
    #trend_data.to_csv('./reddit_post_summary/trend_data_'+start_epoch_raw.strftime("%d_%m_%y")+'.csv')
    #'./reddit_post_summary/trend_data_' # use your path
    all_files = glob.glob(path + "*.csv")

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    frame['date'] = pd.to_datetime(frame.date)
    frame = frame.sort_values(by=sort_by)
   
    return frame


def main_post_text_tableau_function(date = '2022-02-26'):

    frame=get_connection_frame()

    post_frame=get_post_text_data()
    post_frame['post_text']=post_frame['title']+(post_frame['selftext'].fillna(''))
    post_frame['post_url']="https://www.reddit.com/"+post_frame['reddit_id'].str[-6:]
    post_frame['post_id']=post_frame['reddit_id'].str[-6:]
    post_frame.loc[post_frame['date']>date][['post_id','post_text','post_url','ups','downs']]
    post_frame_reduced=post_frame.loc[post_frame['date']>date][['post_id','post_text','post_url','ups','downs','date']]
    frame['connections']=frame['connections'].str.replace('{', '').str.replace('}', '')
    frame['connections']=frame['connections'].apply(lambda x: x.split(","))
    frame_expanded=frame.explode(['connections'])
    frame_expanded=frame_expanded[frame_expanded['date']>=date]
    frame_expanded['connections']=frame_expanded['connections'].str.replace("'", "")


    post_text=post_frame_reduced.merge(frame_expanded, left_on='post_id',right_on='connections')
    post_text['post_text']=post_text['post_text'].str.replace("\n", "")
    #post_text.to_csv('output/post_text.csv')

    post_text_2 = post_text.copy()


    post_text_2.columns = [ i if "token" not in i else "token_2" if "_1" in i else "token_1" for i in post_text_2.columns]



    post_test_merge = pd.concat([post_text,post_text_2],ignore_index=True)

    post_test_merge['main_token']=post_test_merge['token_1']

    post_test_merge.to_csv('output/post_text.csv')
    
    
    
if __name__=="__main__":
    
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    main_post_text_tableau_function()     
    