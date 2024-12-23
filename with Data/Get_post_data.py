#%%time
from Import_lib import *





###########################################
# initialize global variables #############
###########################################


start_epoch_raw=dt.datetime(2022, 3, 18)
end_epoch_raw= start_epoch_raw + dt.timedelta(1)

start_epoch = int(start_epoch_raw.timestamp())
end_epoch = int(end_epoch_raw.timestamp())


# Read-only instance
reddit_read_only = praw.Reddit(client_id="rsF4wE140Vk9jlNhJhjzmQ",         # your client id
                               client_secret="Li7yjYb_1Thog58-iWssIDz7k-s1dg",      # your client secret
                               user_agent="Cryptic-DVA-2022",
                               after=start_epoch, before = end_epoch,
                               filter=['url','author', 'title', 'subreddit'],
                               limit = 1000)        # your user agent
 
# Authorized instance
reddit_authorized = praw.Reddit(client_id="rsF4wE140Vk9jlNhJhjzmQ",         # your client id
                               client_secret="Li7yjYb_1Thog58-iWssIDz7k-s1dg",      # your client secret
                               user_agent="Cryptic-DVA-2022",        # your user agent
                                username="Cryptic-DVA-2022",        # your reddit username
                                password="Cryptic-DVA-2022") 


reddit_read_only_crypt = reddit_read_only#.subreddit("CryptoCurrency")

#r = praw.Reddit(...)
#api = PushshiftAPI(reddit_read_only_crypt)

# not using praw object
api = PushshiftAPI()








################################################################
###############################################################






def get_data(start_epoch_raw,end_epoch_raw, subreddit_ = 'CryptoCurrency',post_limit =200, given_data = None ):
    
    start_epoch = int(start_epoch_raw.timestamp())
    end_epoch = int(end_epoch_raw.timestamp())  
    
    if type(given_data) == type(None):



        #search posts with word
        #api_request_generator = api.search_submissions(q='Missy Elliott', score = ">2000")
        #only to be used without praw in psaw api
        while(True):
            try:
                posts = api.search_submissions(after=start_epoch,
                                             before = end_epoch,

                                    subreddit=subreddit_,
                                       sort = 'desc',
                                       sort_type =  'score',
                                    limit=post_limit#,
                                     #stop_condition=lambda x: x.is_crosspostable
                                      )
                Posts_df = pd.DataFrame([submission.d_ for submission in posts if submission.is_crosspostable])
                break
            except:
                print('Max Retries reached. Sleeping for 1 minute ',subreddit_," ",str(start_epoch_raw),flush=True)
                time.sleep(60)


    
    
    
    

        try:
            Posts_df['date'] = pd.to_datetime(Posts_df['created_utc'], utc=True, unit='s')
        except :
            return pd.DataFrame({'token': {},
                                 'weight': {},
                                 'ups': {},
                                 'downs': {},
                                 'avg_upvote_ratio': {},
                                 'num_comments': {},
                                 'score_total': {},
                                 'total_awards_received_total': {},
                                 'post_id': {},
                                 'Flair_list': {},
                                 'noun': {},
                                 'date': {},
                                 'subreddit': {}})

        Posts_df.drop_duplicates(subset ='id',inplace =True,ignore_index = True)
        Posts_df = Posts_df[Posts_df.id.apply(lambda x : x not in [np.nan,None,""])]


        Posts_df['reddit_id'] = Posts_df['id'].apply(lambda x :  x if x.startswith('t3_') else f't3_{x}')

        
        info_table = list(reddit_read_only.info( Posts_df['reddit_id'].to_list() ))  
        
        i_list = [i.id for i in info_table]
        if(len(i_list) != len(Posts_df['reddit_id'])):
            Posts_df = Posts_df[Posts_df['id'].apply(lambda x: x in i_list )]
        
        Posts_df['downs'] = [i.downs for i in info_table]
        Posts_df['ups'] = [i.ups for i in info_table]
        Posts_df['upvote_ratio'] = [i.upvote_ratio for i in info_table]
        Posts_df['num_comments'] = [i.num_comments for i in info_table]
        Posts_df[ 'num_crossposts'] = [ i.num_crossposts for i in info_table]
        Posts_df[ 'num_reports'] = [ i.num_reports for i in info_table]
        Posts_df[ 'over_18'] = [ i.over_18 for i in info_table]
        Posts_df['score'] = [ i.score for i in info_table]
        Posts_df['total_awards_received'] = [ i.total_awards_received  for i in info_table]
        Posts_df[ 'top_awarded_type'] = [ i.top_awarded_type for i in info_table]
        Posts_df[ 'link_flair_text'] = [ i.link_flair_text for i in info_table]





        #Posts_df.to_parquet("./redditdata/"+subreddit_+start_epoch_raw.strftime("%d_%m_%y")+".parquet")
        Posts_df.to_csv("./redditdata/"+subreddit_+start_epoch_raw.strftime("%d_%m_%y")+".txt", sep="|",index = False)

    else:
        #given_data[]
        Posts_df = given_data[given_data.apply(lambda x: x["date__"] == start_epoch_raw.strftime("%d_%m_%y") and x.subreddit == subreddit_ ,axis =1)]
        Posts_df.to_csv("./redditdata/"+subreddit_+start_epoch_raw.strftime("%d_%m_%y")+".txt", sep="|",index = False)
        
        
     
    





def bind_data_overdate(subreddit_ = 'CryptoCurrency' , start_epoch_raw = None, duration =31, post_limit =200, given_data = None):
    #time.sleep(300)
    if start_epoch_raw == None:
        start_epoch_raw=dt.datetime(2022, 2, 1)
    #end_epoch_raw= start_epoch_raw + dt.timedelta(1)
    
    
    data_list = []

    
    for i in range(duration):
        get_data(start_epoch_raw,start_epoch_raw + dt.timedelta(1),subreddit_ = subreddit_,post_limit=post_limit , given_data = given_data )
        
        start_epoch_raw += dt.timedelta(1)
        

    
    
    

def get_post_data_and_summary(subreddit_list = None,duration = 31,post_limit =200, start = (2022, 2, 1), end = (2022,3,1) ,given_data=None,summary_address = './reddit_post_summary' ):
    
    
    start_epoch_raw=dt.datetime(start[0], start[1], start[2])
    end_epoch_raw= dt.datetime(end[0], end[1], end[2])#start_epoch_raw + dt.timedelta(1)
    
    if subreddit_list == None:
        subreddit_list = ['CryptoCurrency','CryptoMarkets']#,'CryptoMoonShots']#'CryptoMoonShots']#,'CryptoTechnology']
        
    
    
    
    while(start_epoch_raw<end_epoch_raw):
        [bind_data_overdate(subreddit_ = i,start_epoch_raw = start_epoch_raw, duration = duration,post_limit =post_limit,given_data=given_data) for i in subreddit_list]

        start_epoch_raw = start_epoch_raw + dt.timedelta(duration)
        







##################################3
##################################
#################################














