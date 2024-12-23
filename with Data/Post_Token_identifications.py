import get_data_frames as gd
from Import_lib import *


def create_summary_each_day(post_frame):

    processing_df = post_frame[[
                    'title', 'selftext'
                    ]].apply(lambda x: str(x.title)+" "+str(x.selftext),axis =1)


    # Instantiate the vectorizer
    word_vectorizer = CountVectorizer(
        stop_words='english',
        #sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        lowercase  = True,
        token_pattern=r'\w{2,}',  #vectorize 2-character words or more
        ngram_range=(1, 2),
        max_features=30000)

    # fit and transform on it the training features
    txt_fitted  = word_vectorizer.fit([" ".join(processing_df.to_list())])


    txt_fitted_2  = txt_fitted.transform(processing_df.to_list())
    txt_fitted_2_array = txt_fitted_2.toarray()
    doc_freq = [ np.count_nonzero([j[i] for j in txt_fitted_2_array if j[i]> 0 ])    for i in range(len(txt_fitted.get_feature_names())) ]



    ### edditiing here
    upvotes_total = [ np.sum([ post_frame.ups.iloc[j]  for j in range(len(txt_fitted_2_array)) if txt_fitted_2_array[j][i] > 0 ])    for i in range(len(txt_fitted.get_feature_names())) ]
    downs_total = [ np.sum([ post_frame.downs.iloc[j]  for j in range(len(txt_fitted_2_array)) if txt_fitted_2_array[j][i] > 0 ])    for i in range(len(txt_fitted.get_feature_names())) ]
    upvote_ratio_total = [ np.mean([ post_frame.upvote_ratio.iloc[j]  for j in range(len(txt_fitted_2_array)) if txt_fitted_2_array[j][i] > 0 ])    for i in range(len(txt_fitted.get_feature_names())) ]
    num_comments_total = [ np.sum([ post_frame.num_comments.iloc[j]  for j in range(len(txt_fitted_2_array)) if txt_fitted_2_array[j][i] > 0 ])    for i in range(len(txt_fitted.get_feature_names())) ]
    score_total = [ np.sum([ post_frame.score.iloc[j]  for j in range(len(txt_fitted_2_array)) if txt_fitted_2_array[j][i] > 0 ])    for i in range(len(txt_fitted.get_feature_names())) ]
    total_awards_received_total = [ np.sum([ post_frame.total_awards_received.iloc[j]  for j in range(len(txt_fitted_2_array)) if txt_fitted_2_array[j][i] > 0 ])    for i in range(len(txt_fitted.get_feature_names())) ]
    id_list_total = [ ":".join([ str(post_frame['id'].iloc[j])  for j in range(len(txt_fitted_2_array)) if txt_fitted_2_array[j][i] > 0 ]) for i in range(len(txt_fitted.get_feature_names())) ]
    Flair_list_total = [ ":".join([ str(post_frame['link_flair_text'].iloc[j])    for j in range(len(txt_fitted_2_array)) if txt_fitted_2_array[j][i] > 0 ])    for i in range(len(txt_fitted.get_feature_names())) ]

    #upvotes_total = [ np.sum([ Posts_df.ups.iloc[j]  for j in range(len(txt_fitted_2_array)) if txt_fitted_2_array[j][i] > 0 ])    for i in range(len(txt_fitted.get_feature_names())) ]
    #upvotes_total = [ np.sum([ Posts_df.ups.iloc[j]  for j in range(len(txt_fitted_2_array)) if txt_fitted_2_array[j][i] > 0 ])    for i in range(len(txt_fitted.get_feature_names())) ]


    ####



    rr_df = dict(zip(txt_fitted.get_feature_names(), doc_freq))
    token_weight_df = pd.DataFrame.from_dict(rr_df, orient='index').reset_index()
    token_weight_df.columns=('token','weight')

    token_weight_df['ups'] = upvotes_total
    token_weight_df['downs'] = downs_total
    token_weight_df['avg_upvote_ratio'] = upvote_ratio_total
    token_weight_df['num_comments'] = num_comments_total
    token_weight_df['score_total'] = score_total
    token_weight_df['total_awards_received_total'] = total_awards_received_total
    token_weight_df['post_id'] = id_list_total
    token_weight_df['Flair_list'] = Flair_list_total


    token_weight_df['noun'] = token_weight_df.token.apply( lambda x : nltk.pos_tag([x])[0][1][:2] == 'NN')
    token_weight_df['date'] = post_frame['date'].iloc[0][:10]
    token_weight_df['daily_popularity'] = token_weight_df.apply(lambda x : (x.weight*10 + (x.ups +
                                                             x.num_comments +  x.total_awards_received_total*10  #-
                                                            #(x.ups/x.avg_upvote_ratio)*(1-x.avg_upvote_ratio)
                                                            )) ,axis =1)
    
    #token_weight_df['date'] =  pd.to_datetime(start_epoch, utc=False, unit='s') #str(pd.to_datetime(start_epoch, utc=True, unit='s'))[:10]#
    token_weight_df = token_weight_df.sort_values(by='daily_popularity', ascending=False)





    token_weight_df_2 = token_weight_df.query('weight >1').copy()#.query('noun').query('weight >1').copy()

    return token_weight_df_2


def month_summary(data,summary_address = './reddit_post_summary_month'):
    #_ay_list = data['date__'].unique()
    month = data['month__'].iloc[0]
    #print(month)
    month_list = []
    for name,group in data.groupby('date__'):
        day_data  = create_summary_each_day(group)
        day_data['date__'] = name
        month_list.append(day_data)
    
    month_data = pd.concat(month_list,ignore_index = True)
    
    
    month_data.to_csv(summary_address+'/trend_data_'+month+'.csv')
    #return(month_data)



def generate_Data(post_frame = []):
    if len(post_frame) == 0:
        post_frame = gd.get_post_data()

    post_frame.groupby('month__').apply(month_summary)



