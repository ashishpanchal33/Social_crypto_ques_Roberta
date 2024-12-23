#pandas and numpy
import pandas as pd
import numpy as np
import datetime as dt
#file address reader
import glob


def get_post_data(path = './redditdata/'):  # use your path

    all_files = glob.glob(path + "*.txt")

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0, sep = "|")
        li.append(df)
    #print(all_files)
    post_frame = pd.concat(li, axis=0, ignore_index=True)
    post_frame = post_frame.drop_duplicates()
    post_frame['date__'] = pd.to_datetime(post_frame['date'].apply(lambda x: x[:19])).apply(lambda x :x.strftime("%d_%m_%y"))
    #post_frame['date__']
    post_frame['month__'] = post_frame['date__'].apply( lambda x: x[-5:])
    
    return post_frame








def get_post_summary_trend_data(path = './reddit_post_summary_day/trend_data_'):
    #trend_data.to_csv('./reddit_post_summary/trend_data_'+start_epoch_raw.strftime("%d_%m_%y")+'.csv')
    #'./reddit_post_summary/trend_data_' # use your path
    all_files = glob.glob(path + "*.csv")

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    frame['date'] = pd.to_datetime(frame.date)
    frame = frame.sort_values(by=['token','date'])
    frame['month__'] = frame.date__.map(lambda x: x[-5:])
    
    return frame



def get_sentiment_post_frame(path = './redditdata_sentiment/'): # use your path
     
    all_files = glob.glob(path + "*.txt") #+glob.glob(path + "*02_01_22.txt")# "*01_01_22.txt")

    li = []
    print(all_files)
    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0, sep = "|")
        li.append(df)

    post_frame = pd.concat(li, axis=0, ignore_index=True)
    post_frame = post_frame.drop_duplicates()
    return post_frame


def get_frame_for_connections(path = './reddit_post_score_summary_month/all_summary_', n=300):
    #trend_data.to_csv('./reddit_post_summary/trend_data_'+start_epoch_raw.strftime("%d_%m_%y")+'.csv')
    #'./reddit_post_summary/trend_data_' # use your path
    all_files = glob.glob(path + "*.csv")

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        f_2 = df[df.noun].sort_values(by = 'daily_popularity_pop', ascending=False).groupby("date").head(n)
        
        li.append(f_2.reset_index(drop=True))

    frame = pd.concat(li, axis=0, ignore_index=True)
    frame['date'] = pd.to_datetime(frame.date)
    frame = frame.sort_values(by=['token','date'])
    frame['month__'] = frame.date__.map(lambda x: x[-5:])
    
    return frame










