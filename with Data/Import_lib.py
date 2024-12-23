
#reddit libs
import praw
from psaw import PushshiftAPI

#pandas and numpy
import pandas as pd
import numpy as np
import datetime as dt
#file address reader
import glob

#nltk
import nltk
nltk.download('averaged_perceptron_tagger')
import nltk
from nltk import Text
from nltk.tokenize import regexp_tokenize
from nltk.tokenize import word_tokenize  
from nltk.tokenize import sent_tokenize 
from nltk.corpus import stopwords

from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

# vectorizer
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer


# plotly charts
import plotly.express as px



from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)