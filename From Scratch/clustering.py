import requests
import json
import pandas as pd
from pandas.io.json import json_normalize
import datetime
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
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
#import seaborn as sns
#sns.set()
from kmodes.kmodes import KModes
import datadotworld as dw
import warnings

def dw_get(dataset_name, table_name):
    results = dw.query(dataset_name, "SELECT * FROM `{}`".format(table_name))
    df = results.dataframe
    return df


def create_record(text,master_list):
    text_dict = {i: 1 if i in text else 0  for i in master_list}
        
    return text_dict

def get_cleaned_dw_data():
    

    coin_info = dw_get('cnoza/cryptocurrencies', 'eur')

    df=coin_info[['symbol','tags']]
    df['clean_tags']=df['tags'].str.replace('"', '').str.replace('[', '').str.replace(']', '')
    df['clean_tags']=df['clean_tags'].apply(lambda x: x.split(","))
    df['clean_tags_str']  = df['clean_tags'].apply(lambda x:" ".join(x))

    df1=df
    master_list = np.unique((",".join(df1['clean_tags'].apply(lambda x: ','.join(map(str, x))).to_list())).split(","))

    
    #creating records
    count_vectors = pd.DataFrame.from_records(df1.clean_tags.apply(lambda x: create_record(x,master_list)).to_list())
    count_vectors=count_vectors[[i for i in (count_vectors.columns) if "portfolio" not in i]]
    
    return count_vectors,coin_info,df1


    
def clustering_elbow_curve(count_vectors, k_range =(10,30),mode='manual'):    
#Elbow Curve
    if mode == 'manual':
        cost = []
        K = range(k_range[0],k_range[1])
        for num_clusters in list(K):
            kmode = KModes(n_clusters=num_clusters, init = "random", n_init = 10, verbose=0)
            kmode.fit_predict(count_vectors)
            cost.append(kmode.cost_)

        #visualize and valiate
        plt.plot(K, cost, 'bx-')
        plt.xlabel('No. of clusters')
        plt.ylabel('Cost')
        plt.title('Elbow Method For Optimal k')
        plt.show()
        
        return None
    
    
    elif mode =='auto':
        from yellowbrick.cluster import KElbowVisualizer

        # Instantiate the clustering model and visualizer
        model = KModes(init = "random", n_init = 10, verbose=0)
        visualizer = KElbowVisualizer(model, k=k_range,method='silhouette',timings=False, locate_elbow=True)

        visualizer.fit(count_vectors)        # Fit the data to the visualizer
        
        return visualizer.elbow_value_       # Finalize and render the figure

def cluster_data(count_vectors,df1,n_clusters=20):
# Building the model with 20 clusters
    kmode = KModes(n_clusters=n_clusters, init = "random", n_init = 30, verbose=0)
    clusters = kmode.fit_predict(count_vectors)

    count_vectors.insert(0, "Cluster", clusters, True)
    count_vectors['symbol']=df1['symbol']

    similarities={}
    for i in count_vectors['Cluster'].unique():
        x=count_vectors.groupby(['Cluster']).mean().loc[i].sort_values(ascending= False)
        x=x[x>1/count_vectors.groupby(['Cluster'])['Cluster'].count()[i]]
        similarities[i]=x

    similarities=pd.DataFrame(similarities.items())
    count_vectors['groups']=count_vectors.groupby(['Cluster'])['symbol'].transform(lambda x : ','.join(x))
    count_vectors=count_vectors.merge(similarities, left_on='Cluster', right_on=0, how='left')

    return count_vectors



def  get_wikipedia_data(coin_info):
    
    info={}
    errors=[]



    import wikipedia
    for i in coin_info['name']:

        try:
            info[i]=wikipedia.summary(i+" crypto", sentences=3)
        except:
            try:
                info[i]=wikipedia.summary(i+" cryptocurrency", sentences=3)
            except:
                try:
                    info[i]=wikipedia.summary(i+" blockchain", sentences=3)
                except:
                    i = i.lower()
                    try:
                        info[i]=wikipedia.summary(i+" crypto", sentences=3)
                    except:
                        try:
                            info[i]=wikipedia.summary(i+" cryptocurrency", sentences=3)
                        except:
                            try:
                                info[i]=wikipedia.summary(i+" blockchain", sentences=3)
                            except:
                                errors.append(i)

    info=pd.DataFrame(info.items())

    info.columns=['name','info']
    
    return info


def main_coin_clustering_and_data_proc():
    print('fetched and cleaned coin data')
    count_vectors,coin_info,df1 = get_cleaned_dw_data()

    #clustering_elbow_curve(count_vectors)
    print('starting kmode cluster elbow curve')
    res_num = clustering_elbow_curve(count_vectors, k_range =(15,30),mode='manual')

    if res_num == None:
        res_num = 20
    print('starting kmode clustering')        
    count_vectors = cluster_data(count_vectors,df1,n_clusters=res_num)

    print('fetching wikipedia data') 
    wiki_info = get_wikipedia_data(coin_info)
    wiki_info.to_csv('input/wiki_info.csv')
    print('creating cluster detail ADS')

    count_vectors['name']=coin_info['name']

    count_vectors['main_token']=count_vectors['name'].str.lower()
    count_vectors = count_vectors.merge(wiki_info, on='name', how='left').reset_index()
    count_vectors.to_csv('input/cluster_info.csv')
    about_data=count_vectors.copy()#pd.read_csv('input/cluster_info.csv')
    about_data.rename({1: 'cluster_info'}, axis=1, inplace=True)
    about_data[['main_token','name','info','groups','Cluster']].to_csv('output/about_data.csv')
    about_data['cluster_info']=about_data['cluster_info'].apply(lambda x: x.to_dict())
    about_data[['main_token','groups','cluster_info']].to_csv('output/cluster_info.csv')
    
    
    
if __name__=="__main__":
    
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    main_coin_clustering_and_data_proc()      
    
    