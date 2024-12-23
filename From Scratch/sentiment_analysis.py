import multiprocessing as mp
import pandas as pd
import glob
import time
import pandas as pd









def roberta_init():
    from transformers import AutoModelForSequenceClassification
    from transformers import TFAutoModelForSequenceClassification
    from transformers import AutoTokenizer, AutoConfig
    import numpy as np
    from scipy.special import softmax
    # Preprocess text (username and link placeholders)
    def preprocess(text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    # PT
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)
    
    return {"model":model,"tokenizer":tokenizer,"config":config,"preprocess":preprocess,"softmax":softmax}
def vader_init():
    import nltk
    nltk.download('omw-1.4')
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


    sid_obj = SentimentIntensityAnalyzer()
    return sid_obj

def distilbert_emo_init():
    from transformers import pipeline

    classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True)
    #text = 'we will see'
    return classifier

def roberta_sent(text,params_dict):

    text = params_dict["preprocess"](text)
    encoded_input = params_dict["tokenizer"](text, return_tensors='pt')
    #output = 
    #print(output)
    scores = params_dict["model"](**encoded_input)[0][0].detach().numpy()
    scores = params_dict["softmax"](scores)
    
    return({"roberta_"+i:j for i,j in zip(params_dict["config"].id2label.values(),scores)})

def vader_sent(text,sid_obj):
    return {"vader_"+i:j for i,j in sid_obj.polarity_scores(text).items()}



def distilbert_emo(text,classifier):
    prediction = classifier(text, )
    return({ "distil_"+i['label']:i['score']   for i in  prediction[0]})


Sent_emo_model_params = dict({})
Sent_emo_model_params['roberta_params'] =roberta_init()
Sent_emo_model_params['vader_params'] =vader_init()
Sent_emo_model_params['distilbert_emo_params'] =distilbert_emo_init()

def get_emotions(text,Sent_emo_model_params = Sent_emo_model_params):
    text= text[0]
    #print(text)
    
    return  (roberta_sent(text, Sent_emo_model_params['roberta_params']) \
            | vader_sent(text,Sent_emo_model_params['vader_params'])\
            | distilbert_emo(text,Sent_emo_model_params['distilbert_emo_params']))


def get_emo_df(i,j):
    #i,j =dats
    
    
    j.reset_index(drop=True,inplace=True)
    fata = pd.DataFrame.from_records(
        j[['title','selftext']].apply(
            lambda x:get_emotions(x['title']+str(x['selftext']) )
            ,axis=1))
    #print(i)
    fata_2 = pd.concat([j,fata],axis=1)
    #fata_2.columns = list(j.columns) + list(fata.columns)
    fata_2.to_csv("./redditdata_sentiment/"+j['date__'].iloc[0]+".txt", sep="|",index = False)
    
    
    
    #return fata_2



def run_onlist_loop(data):

    [get_emo_df(name,group) for name,group in data.groupby('date__')]



def run_onlist(data):
    #return([get_emotions(i) for i in list_])



    pool = mp.Pool(5)
    #results = pool.starmap(sa.get_emotions,list_)
    results = pool.starmap(get_emo_df,data)
    #results = pool.starmap(sa.run_onlist,list_2)
    #results = pool.map(sa.run_onlist,list_2)
    pool.close()
    pool.join()
    return(results)






