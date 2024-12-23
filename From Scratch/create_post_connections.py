import pandas as pd
from Import_lib import *



def connection_per_day(data):
    
    date = pd.to_datetime(data.date.values[0])
    
    weight_list = []
    #connection_list = []
    connection_list =[]
    
    
    def find_top_6_connections(main_token, main_token_data,data=data ):
        data = data[data.token != main_token]
        if len(data)>0:
            weight_list.append(main_token_data[['token','date','daily_popularity_pop']].to_frame().T)
            #display(weight_list)
            token_list = main_token_data['post_id_broken']
            data['connections'] = data['post_id_broken'].apply(lambda x: set( x )& set(token_list ))
            data['connections_strength'] = data['connections'].apply(lambda x: int(len(x  )))
            #display(data)
            ret = data[data.connections_strength > 0].nlargest( columns = ['connections_strength'],n=5)
            #ret['main_token'] = main_token
            if len(ret)>0:
                ret['token_connection'] = ret['token'].apply(lambda x: {x,main_token})

                weight_list.append(ret[['token','date','daily_popularity_pop']])
                #display(weight_list)

                connection_list.append( ret[['token_connection','connections','connections_strength','date']])
            #print(l)
    
    
    
    
    #display(data)
    #display(data.sort_values(by='daily_popularity_pop',ascending =False).head(100)[['token','daily_popularity_pop']])
    data.sort_values(by='daily_popularity_pop',ascending =False).head(50).apply(lambda x: find_top_6_connections(x['token'], main_token_data=x),axis=1    
                                             )
    
    #display(data)
    if len(connection_list)>0:
    
        connection_list = pd.concat(connection_list,ignore_index=True)
        connection_list['dupl'] = connection_list.token_connection.apply(lambda x: ",".join(x))
        connection_list.drop_duplicates(subset=['dupl'], inplace=True)
        connection_list['token_connection'] = connection_list['token_connection'].map(list)
        #print(connection_list['token_connection'])
        connection_list['token_1'] = connection_list['token_connection'].apply(lambda x : x[0])
        connection_list['token_2'] = connection_list['token_connection'].apply(lambda x : x[1])        


        connection_list[['token_1','token_2','connections','connections_strength','date']].to_csv('./reddit_topic_connections_day/'+date.strftime("%d_%m_%y")+".csv")
    if len(weight_list)>0:
        weight_f = pd.concat(weight_list,ignore_index=True).drop_duplicates()
        #display(weight_f)
        weight_f.to_csv('./reddit_topic_weights_connections_day/'+date.strftime("%d_%m_%y")+".csv")
        





def get_connections(senti_path):
    senti_path['post_id_broken']= senti_path['post_id'].apply(lambda x:  set(x.split(":")))
    _ = [ connection_per_day(group) for name, group in senti_path[senti_path.noun].sort_values(by=['date','daily_popularity_pop'],ascending =[True,False]).groupby(by='date')]    