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
import get_data_frames as gd


import datetime as dt
import plotly.express as px

import glob
import time





def wt_sum(x,sent):
    #print(x[[sent,'total_awards_received','num_comments','ups']])
    val = x[sent]*(x['total_awards_received']*10 +\
      x['num_comments'] +\
      x['ups']+10)
    #print(val.sum())
    return val







def sentiment_agg(name,token_ata= [],post_frame_2=[],sentiment_list=[]):
    #token_ata = frame[frame.date__.values == '01_01_22']


    if len(token_ata) == 0:
        token_ata = gd.get_post_summary_trend_data()
        
    if len(post_frame_2) == 0:
        post_frame_2 = gd.get_sentiment_post_frame().set_index('id')
    
    if len(sentiment_list) == 0:
        sentiment_list = ['roberta_Negative', 'roberta_Neutral',
                           'roberta_Positive', 'vader_neg', 'vader_neu', 'vader_pos',
                           'vader_compound', 'distil_sadness', 'distil_joy', 'distil_love',
                           'distil_anger', 'distil_fear', 'distil_surprise']
    
    
    post_frame_3 = post_frame_2[post_frame_2.date__.values == name]

    for sent in sentiment_list:
        token_ata[sent+"_sum"] = token_ata.post_id.apply(lambda x: sum([post_frame_3.loc[i][sent]
                                              for i in str(x).split(":") if i in post_frame_3.index ]) 

                                                        )
        token_ata[sent+"_wt_sum"] = token_ata.post_id.apply(lambda x: sum([   wt_sum(post_frame_3.loc[i], sent)

                                              for i in str(x).split(":") if i in post_frame_3.index ]))


        token_ata[sent+"_mean"] = token_ata.post_id.apply(lambda x: np.mean([post_frame_3.loc[i][sent]
                                          for i in str(x).split(":") if i in post_frame_3.index ]))
    token_ata.to_csv('./reddit_post_summary_day/trend_data_'+name+'.csv')
    
def run_sentiment_agg(frame,post_frame_2):
    [sentiment_agg(name,group,post_frame_2)  for name,group in  frame.groupby('date__')]

