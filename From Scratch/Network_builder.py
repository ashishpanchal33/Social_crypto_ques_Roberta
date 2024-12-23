

#Network Builder

from pandas import read_csv
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import glob
import datetime

import networkx as nx
import numpy as np
import pandas as pd
from pandas import DataFrame
import pandas as pd
import warnings

#Read topic connections
def get_connection_frame(path = './reddit_topic_connections_day/',sort_by=['token_1','date']):
    all_files = glob.glob(path + "*.csv")

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=None, header=0)
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    frame['date'] = pd.to_datetime(frame.date)
    frame = frame.sort_values(by=sort_by)
   
    return frame



def create_nodes(date= '2022-02-26'):


    frame=get_connection_frame()
    filtered_tokens=frame[frame['date']>=date]
    filtered_tokens.to_csv('input/filtered_tokens.csv')
    frame.rename({'token_1': 'main_token'}, axis=1, inplace=True)

    filtered_node_file = get_connection_frame(path = './reddit_topic_weights_connections_day/',sort_by=['date'])
    filtered_node_file[filtered_node_file['date']>=date].to_csv('input/filtered_tokens_nodes.csv')

    return filtered_node_file





def create_clusters():

    filtered_node_file = create_nodes()


    InFile = 'input/filtered_tokens.csv'
    CodeType = 'latin1' # https://docs.python.org/3/library/codecs.html#standard-encodings
    Src_Column = 'token_1'
    Tgt_Column = 'token_2'
    CoordsFile = 'input/coordsfile.csv'
    BridgeFile = 'input/bridgefile.csv'

    # Read in Source file (add Index Column manually)...
    df_InputData_node = pd.read_csv("input/filtered_tokens_nodes.csv",sep=',',encoding=CodeType)
    df_InputData = pd.read_csv(InFile,sep=',',encoding=CodeType)
    main_combined = pd.DataFrame(columns = ['token_1','token_2','connections','connections_strength','date','X','Y'])




    for i in (df_InputData['date'].unique()):
        dfSub=df_InputData[df_InputData['date']==i]
        arr_SrcTgt = np.array(dfSub[[Src_Column,Tgt_Column]])
        # Create Network Graph Coordinates...
        Q = nx.Graph()
    #    Q.add_edges_from(arr_SrcTgt)
        Q.add_nodes_from(df_InputData_node.token.to_list())
        dict_Coords = nx.spring_layout(Q) 

        # Create Graph Coordinates File...
        df_Raw_Coords = DataFrame(dict_Coords)
        df_Raw_Coords = df_Raw_Coords.T
        df_Raw_Coords.columns = ['X','Y']

        # Create Bridge File... 
        # Tableau Code: IF [Src-Tgt]/2 = ROUND([Src-Tgt]/2) THEN 'Source' ELSE 'Target' END
        arr_SrcTgt2 = arr_SrcTgt.reshape(1,(len(arr_SrcTgt)*2)) 
        arr_SrcTgt2 = arr_SrcTgt2.reshape(-1) 
        df_SrcTgt = DataFrame(arr_SrcTgt2,columns=['NodeName']) 
        arr_Index = []
        for i in range(1,(len(arr_SrcTgt)+1)):
            arr_Index.append(i)
            arr_Index.append(i)
        df_SrcTgt['c_Index'] = arr_Index 


        print('Run Completed Successfully')

        df_Raw_Coords['tokenname']=df_Raw_Coords.index
        main1=dfSub.merge(df_Raw_Coords, left_on='token_1',right_on='tokenname')
        main2=dfSub.merge(df_Raw_Coords, left_on='token_2',right_on='tokenname')
        main=pd.concat([main1, main2],ignore_index=True)
        main_combined=main_combined.append(main)

    main_combined.to_csv('output/nw_main.csv')
    
    token_popularity=get_connection_frame(path = './reddit_topic_weights_connections_day/',sort_by=['token','date'])
    main_combined['date']=pd.to_datetime(main_combined['date'])
    
    main_combined_merged = main_combined.merge(token_popularity, left_on=['tokenname','date'], right_on=['token','date'], how='left')

    main_combined_merged_2 = main_combined_merged.copy()
    main_combined_merged_2.columns = ["token_2","token_1"] + list(main_combined_merged_2.columns[2:])
    main_combined_merged__merge = pd.concat([main_combined_merged_2,main_combined_merged],ignore_index=True)
    main_combined_merged__merge['main_token']=main_combined_merged__merge['token_1']
    main_combined_merged__merge.to_csv('output/nw_main.csv')  
    
    
    
    
    
    
    
    return main_combined


    







if __name__=="__main__":
    
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    create_clusters()    
    
    
