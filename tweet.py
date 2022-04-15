
import nltk
nltk.download('stopwords')

nltk.download('wordnet')
pip install textblob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from textblob import TextBlob

data = pd.read_csv(r'D:\Data science\Text Mining\data_elonmusk.csv', encoding="latin-1")
data.head()


data['word_count'] = data['Tweet'].apply(lambda x: len(str(x).split(" ")))
data[['Tweet','word_count']].head()

data['char_count'] = data['Tweet'].str.len() 
data[['Tweet','char_count']].head()


stop = stopwords.words('english')

data['stopwords'] = data['Tweet'].apply(lambda x: len([x for x in x.split() if x in stop]))
data[['Tweet','stopwords']].head()

#######  Preprocessing

data['Tweet'] = data['Tweet'].apply(lambda x: " ".join(x.lower() for x in x.split()))
data['Tweet'].head()


data['Tweet'] = data['Tweet'].str.replace('[^\w\s]','')
data['Tweet'].head()


stop = stopwords.words('english')

data['Tweet'] = data['Tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
data['Tweet'].head()

freq = pd.Series(' '.join(data['Tweet']).split()).value_counts()[:10]
freq


freq = list(freq.index)
data['Text'] = data['Tweet'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
data['Tweet'].head()


data['Tweet'][:5].apply(lambda x: str(TextBlob(x).correct()))


nltk.download('punkt')


from nltk.stem import PorterStemmer

st = PorterStemmer()

data['Tweet'][:5].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))


from textblob import Word

data['Tweet'] = data['Text'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
data['Tweet'].head()


TextBlob(data['Tweet'][0]).ngrams(2)


tf1 = (data['Tweet'][1:2]).apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).reset_index()
tf1.columns = ['words','tf']
tf1

for i,word in enumerate(tf1['words']):
  tf1.loc[i, 'idf'] = np.log(data.shape[0]/(len(data[data['Tweet'].str.contains(word)])))

tf1

tf1['tfidf'] = tf1['tf'] * tf1['idf']
tf1


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(max_features=1000, lowercase=True, analyzer='word',
 stop_words= 'english',ngram_range=(1,1))
vect = tfidf.fit_transform(data['Tweet'])
vect


from sklearn.feature_extraction.text import CountVectorizer

bow = CountVectorizer(max_features=1000, lowercase=True, ngram_range=(1,1),analyzer = "word")
data_bow = bow.fit_transform(data['Tweet'])
data_bow


data['Tweet'][:5].apply(lambda x: TextBlob(x).sentiment)

data['sentiment'] = data['Tweet'].apply(lambda x: TextBlob(x).sentiment[0] )
data[['Tweet','sentiment']].head()


























