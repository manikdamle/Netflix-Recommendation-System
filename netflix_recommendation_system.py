#Manik Damle
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Netflix Recommendation System",layout="wide")
st.title(" :clapper: Netflix Recommendation System")
st.divider()

df=pd.read_csv(r"C:\Users\HP\Downloads\archive(1)\netflixData.csv")
df.sample(5)

df.shape

df.info()

df.isnull().sum()

df=df[["Title","Description","Content Type","Genres"]]
df.isnull().sum()

import nltk
import re
nltk.download('stopwords')

stemmer=nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword=set(stopwords.words("english"))

def clean(text):
    text=re.sub('\[.*?]','',text)
    text = re.sub(r'\\http?://\S+|www\.\S+]', '', text)
    text=re.sub('<.*?>+','',text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text=re.sub('\n','',text)
    text=re.sub('\w*\d\w*','',text)
    text=[word for word in text.split(' ') if word not in stopword]
    text=' '.join(text)
    return text

df["Title"]=df["Title"].apply(clean)
df["Title"].sample(5)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

tfidf=TfidfVectorizer(stop_words='english')

feature=df['Genres'].tolist()
tfidf_matrix=tfidf.fit_transform(feature)

similarity=cosine_similarity(tfidf_matrix)

indices=pd.Series(df.index,index=df['Title']).drop_duplicates()

def netflix_recommendation(title,similarity=similarity):
    try:
        index=indices[title]
        similarity_scores=list(enumerate(similarity[index]))
        similarity_scores=sorted(similarity_scores,key=lambda x:x[1],reverse=True)
        similarity_scores=similarity_scores[0:10]
        movieindices=[i[0] for i in similarity_scores]
        return df['Title'].iloc[movieindices].tolist()
    except:
        print(title,' is not in database')

title = st.text_input('Movie title', '')
st.write('Recommendations: ', netflix_recommendation(title))
