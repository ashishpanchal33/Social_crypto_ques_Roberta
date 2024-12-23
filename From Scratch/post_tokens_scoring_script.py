import praw
from psaw import PushshiftAPI
import pandas as pd


#import nltk
#nltk.download('averaged_perceptron_tagger')

import nltk
from nltk import Text
from nltk.tokenize import regexp_tokenize
from nltk.tokenize import word_tokenize  
from nltk.tokenize import sent_tokenize 
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np



import datetime as dt
import plotly.express as px



import pandas as pd
import glob


from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)



def get_frame(path = './reddit_post_summary_month/trend_data_'): #'./reddit_post_summary/trend_data_' # use your path
    all_files = glob.glob(path + "*.csv")

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    frame['date'] = pd.to_datetime(frame.date)
    frame = frame.sort_values(by=['token','date'])
    
    return frame




def calculate_popularity_v2(date, data, pd=pd):
    one_month_start = date - pd.Timedelta("30 days")
    #print(one_month_start)
    data = data[ (data['date'].values >one_month_start) &  (data['date'].values <=date)]

    popularity = data.daily_popularity.values.sum()/30 


    return popularity  

def calculate_hotness_v2(date, data, pd=pd):
    one_start = date - pd.Timedelta("4 days")

    data = data[ (data['date'].values >one_start) &  (data['date'].values <=date)]

    hotness = data.daily_popularity.values.sum()/(4*3) #130


    return hotness    


def get_yesterdays_data_v2(x,data, pd=pd ):



    j= (data["hotness"][ (data['date'].values == (x.date - pd.Timedelta("1 days")) 
                                                                    )])

    if len(j)>0:
        return j.values[0]
    else:
        return 0   
                
def pop_assist(data,calculate_popularity_v2=calculate_popularity_v2,pd=pd):

    
  
    
    
    
    pop_results = data.apply(lambda x: calculate_popularity_v2(x.date, data), axis=1 )
    
    return pd.concat([data,pop_results] ,axis=1)


def hot_assist(data,calculate_hotness_v2=calculate_hotness_v2,pd=pd):

    

    pop_results = data.apply(lambda x: calculate_hotness_v2(x.date, data), axis=1 )
    
    return pd.concat([data,pop_results] ,axis=1)


def rise_assist(data,get_yesterdays_data_v2=get_yesterdays_data_v2,pd=pd):

    
 

    pop_results = data.apply(lambda x: (x.hotness - get_yesterdays_data_v2(x,data=data)), axis=1)

        
    return pd.concat([data,pop_results] ,axis=1)





def scoring_function(frame):


    list_popularity = frame[['date','token','daily_popularity']].groupby('token').parallel_apply(pop_assist)
    list_popularity.reset_index(drop=True, inplace =True)
    print('pop-done')
    list_popularity.columns = ['date','token','daily_popularity','popularity']

    list_hotness = list_popularity.groupby('token').parallel_apply(hot_assist)
    list_hotness.reset_index(drop=True, inplace =True)
    list_hotness.columns = ['date','token','daily_popularity','popularity','hotness']

    print('hot-done')
    list_rising = list_hotness.groupby('token').parallel_apply(rise_assist)
    list_rising.reset_index(drop=True, inplace =True)
    list_rising.columns = ['date','token','daily_popularity','popularity','hotness','rising']
    print('rising-done')



    pd.merge(frame,list_rising[['date','token','popularity','hotness','rising']],on=['token','date'],how='left')\
    .to_csv("all_Data_popularity__hotness_index_3_base.csv")

    print('written')

