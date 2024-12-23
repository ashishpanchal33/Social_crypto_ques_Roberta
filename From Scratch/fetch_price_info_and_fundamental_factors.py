import requests
import json
import pandas as pd
from pandas.io.json import json_normalize
import datetime
import datadotworld as dw
from pytrends.request import TrendReq
import pandas_datareader.data as web
import requests

import warnings


#Fetch price data

def get_hist_data(from_sym='BTC', to_sym='USD', timeframe = 'day', limit=2000, aggregation=1, exchange=''):
    
    url = 'https://min-api.cryptocompare.com/data/v2/histo'
    url += timeframe
    parameters = {'fsym': from_sym,
                  'tsym': to_sym,
                  'limit': limit,
                  'aggregate': aggregation}
    if exchange:
        parameters['e'] = exchange        
    # response comes as json
    response = requests.get(url, params=parameters)   
    data = response.json()['Data']['Data'] 
    
    return data      

def data_to_dataframe(data):
    #data from json is in array of dictionaries
    df = pd.DataFrame.from_dict(data)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    return df

def dw_get(dataset_name, table_name):
    results = dw.query(dataset_name, "SELECT * FROM `{}`".format(table_name))
    df = results.dataframe
    return df







#Login to Google for this

def scrap_crypto_price_data(coinlist,start_ =(2019, 6, 22),end_=(2022, 4,17)):
    #top 20 coins data


    #Timeperiod
    start = datetime.datetime(start_[0], start_[1], start_[2])
    end = datetime.datetime(end_[0], end_[1], end_[2])
    
    

    #baseurl='https://min-api.cryptocompare.com/data/v2/histoday'
    target_currency = 'USD'
    price_data = pd.DataFrame(columns=['high','low','open',
                                       'volumefrom','volumeto',
                                       'close','time','conversionType',
                                       'conversionSymbol','sym'])

    for cryptocurrency in coinlist:
        data = get_hist_data(cryptocurrency, target_currency, 'day', 1000)
        df = data_to_dataframe(data)
        df['sym']=cryptocurrency
        price_data=price_data.append(df)
        
    price_data['date']=price_data.index
    
    return price_data


def scrapping_search_data(coinlist,timeframe='2019-06-22 2022-04-17'):

    coin_info = dw_get('cnoza/cryptocurrencies', 'eur')
    coin_names=coin_info.loc[coin_info['symbol'].isin(coinlist)]
    coin_names= [x.lower() for x in list(coin_names['name'])]
    column_names = ["SearchFrequency", "sym"]
    searchTrends = pd.DataFrame(columns = column_names)

    # Login to Google. Only need to run this once
    pytrend = TrendReq()
    for i in coin_names:
        pytrend.build_payload(kw_list=[i], cat=16, timeframe=timeframe)  
        story_ggtrends = pytrend.interest_over_time()
        # Upsampling daily data to Date
        story_ggtrends = story_ggtrends.resample('D').pad().drop(['isPartial'], axis='columns')
        story_ggtrends.columns = ['SearchFrequency']
        story_ggtrends['sym']=i
        searchTrends=searchTrends.append(story_ggtrends)
        
    searchTrends['date']=searchTrends.index
    coin_info['name']=coin_info['name'].str.lower()
    search_df=searchTrends.merge(coin_info, left_on='sym', right_on='name', how='left')
        
    return search_df,coin_info




def scrap_external_factors(start_,end_):
    
        
    start = datetime.datetime(start_[0], start_[1], start_[2])
    end = datetime.datetime(end_[0], end_[1], end_[2])



        
        
        
    # Query stock data from Yahoo! Financial using pandas_datareader
    print('3.1 AMD and NVDA')
    amd = web.DataReader('AMD', 'yahoo', start, end)
    nvda = web.DataReader('NVDA', 'yahoo', start, end)
    
    amd['amd_open']=amd['Open']
    nvda['nvda_open']=nvda['Open']
    amd['date']=amd.index
    nvda['date']=nvda.index


    print('3.2 macroeconomic factors')
    
    url = 'https://www.alphavantage.co/query?function=RETAIL_SALES&apikey=demo'
    r = requests.get(url)
    retail_sales = pd.DataFrame(r.json()['data'])
    url = 'https://www.alphavantage.co/query?function=CONSUMER_SENTIMENT&apikey=demo'
    r = requests.get(url)
    consumer_sentiment = pd.DataFrame(r.json()['data'])
    url = 'https://www.alphavantage.co/query?function=UNEMPLOYMENT&apikey=demo'
    r = requests.get(url)
    umemployment = pd.DataFrame(r.json()['data'])
    
    
    #Add macroeconmic data
    retail_sales.rename(columns = {'value':'retail_sales'}, inplace = True)
    retail_sales['date']=pd.to_datetime(pd.Series(retail_sales['date']))
    
    umemployment.rename(columns = {'value':'umemployment'}, inplace = True)
    umemployment['date']=umemployment.index
    umemployment['date']=pd.to_datetime(pd.Series(umemployment['date']))

    
    
    
    print('3.3 Market indices')
    #Get indices info
    index_data = pd.DataFrame(columns=['High','Low','Open','Close','Volume','Adj Close','indexsym','date'])
    indices=['^DJI','^GSPC','^IXIC','^NYA']

    for i in indices:
        x=web.DataReader(i, 'yahoo', start, end)
        x['indexsym']=i
        x['date']=x.index
        x=x.reset_index()
        index_data=index_data.append(x)

    index_data.rename(columns = {'Open':'index_open'}, inplace = True)
    df1=index_data[['indexsym','index_open','date']].pivot(columns ='indexsym', values =['index_open','date'])
    df1.reset_index(inplace=True)

    df1.columns =[s1 + str(s2) for (s1,s2) in df1.columns.tolist()]
    df1.rename(columns = {'index_open^DJI':'dji_open'}, inplace = True)
    df1.rename(columns = {'index_open^GSPC':'gspc_open'}, inplace = True)
    df1.rename(columns = {'index_open^IXIC':'ixic_open'}, inplace = True)
    df1.rename(columns = {'index_open^NYA':'nya_open'}, inplace = True)


    df1.rename(columns = {'date^NYA':'date'}, inplace = True)
    df1=df1[['date','dji_open','gspc_open','ixic_open','nya_open']]
    
    print('3.4 Gold')
    #getting gold data
    gold_x=web.DataReader('GC=F', 'yahoo', start, end)
    gold_x.rename(columns = {'Open':'gold'}, inplace = True)
    gold_x['date']=gold_x.index
    gold=gold_x[['date','gold']].reset_index()
    
    
    
    
    
    
    
    
    

    return amd,nvda,retail_sales,consumer_sentiment,umemployment,df1,gold











def main_trends_data():
    coinlist=['BTC',
    'ETH',
    'USDT',
    'XRP',
    'LUNA',
    'SOL',
    'USDC',
    'AVAX',
    'ADA',
    'DOT','XLM','DOGE','LINK','MATIC','SHIB','ATOM','ICP','LTC','ALGO','MANA' ]
    
    
    start_ =(2019, 6, 22)
    end_ = (2022, 4,17)
    
    print('1. getting_crypto prices')
    price_data = scrap_crypto_price_data(coinlist=coinlist,start_ =start_,end_=end_)
    print('2. getting_crypto search trends')
    search_df,coin_info = scrapping_search_data(coinlist,timeframe= " ".join([ "-".join([str(i) for i in start_])    ,"-".join([str(i) for i in end_]) ]))
    print('3. getting_external factors')
    amd,nvda,retail_sales,consumer_sentiment,umemployment,df1,gold = scrap_external_factors(start_,end_)
    


    
    print('4 Creating ADS')
    
    #merge price data and search frquency
    datax=price_data.merge(search_df[['date','symbol','SearchFrequency']], left_on=['date','sym'], right_on=['date','symbol'], how='left')
    datax=datax.merge(coin_info[['symbol','tags']], left_on='sym', right_on='symbol', how='left')
    
    datax=datax.merge(amd[['date','amd_open']], on='date',how='left')
    datax=datax.merge(nvda[['date','nvda_open']], on='date',how='left')


    datax=datax.merge(retail_sales, on='date',how='left')

    datax=datax.merge(umemployment, on='date',how='left')
    datax=datax.merge(df1, on='date', how='left')
    datax=datax.merge(gold[['date','gold']], on='date', how='left')
    
    
    datax['retail_sales'].fillna(method='bfill', inplace=True)
    datax['umemployment'].fillna(method='bfill', inplace=True)    


    datax['amd_open'].fillna(method='bfill', inplace=True)
    datax['nvda_open'].fillna(method='bfill', inplace=True)
    datax['dji_open'].fillna(method='bfill', inplace=True)
    datax['gspc_open'].fillna(method='bfill', inplace=True)
    datax['ixic_open'].fillna(method='bfill', inplace=True)
    datax['nya_open'].fillna(method='bfill', inplace=True)
    datax['gold'].fillna(method='bfill', inplace=True)
    
    columns_needed="high,low,open,volumefrom,volumeto,close,conversionType,sym,date,SearchFrequency,tags,amd_open,nvda_open,retail_sales,dji_open,gspc_open,ixic_open,nya_open,gold"
    columns_needed=columns_needed.split(",")
    print('5 storing ADS')
    datax[columns_needed].to_csv('input/dataset.csv')    
    print(' Process complete')
    
    



if __name__=="__main__":
    
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    main_trends_data()






#datax.to_csv('input/dataset.csv')
#df1['dji_open']=df1['dji_open'].astype(float)
#df1['gspc_open']=df1['gspc_open'].astype(float)
#df1['ixic_open']=df1['ixic_open'].astype(float)
#df1['nya_open']=df1['nya_open'].astype(float)
#df1[['dji_open','gspc_open','ixic_open','nya_open']].corr(method='pearson')

#i='XAUUSD=X'




