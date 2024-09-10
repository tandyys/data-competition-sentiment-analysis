import pandas as pd
import nltk #natural langauge toolkit
nltk.download('punkt')
from nltk.tokenize import word_tokenize as tokenizer
from nltk.stem import PorterStemmer as ps
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import spacy as spy
import gensim
from gensim.models import Word2Vec

#Read the data and clean it
data = pd.read_csv("mobil_listrik.csv")
data.shape
data = data.dropna()
data = data.drop_duplicates()
data.head()

#Sentiment labels mapping
sentimentMaps = {
    "positif": 0,
    "negatif": 1,
    "netral" : 2
}
data['sentimen'] = data['sentimen'].map(sentimentMaps)
print(data.head())

#Plotting total sentiments
sentimenPlot = data['sentimen'].value_counts().sort_index()
sentimenBarPlt = sentimenPlot.plot(kind='bar', title='Sentiment Distribution', figsize=(10,5))
sentimenBarPlt.set_xlabel('0: Positif, 1: Negatif, 2: Netral')

#Showing wordcloud -> to see the most frequent words
from collections import Counter
from wordcloud import WordCloud

frequentWords = Counter(" ".join(data['text_cleaning']).split())
wordCloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(frequentWords)
plt.figure(figsize=(15,15))
plt.imshow(wordCloud, interpolation='bilinear')
plt.axis('off')
plt.show()

#Normalization -> here, my normalization based on frequency of words in WordCloud
normWords = {
    "jg" : "juga",
    "cm" : "hanya",
    "yg" : "yang",
    "blm" : "belum",
    "dl"  : "dahulu",
    "ngaco" : "rusak",
    "jt" : "juta",
    "blom" : "belum",
    "tak" : "tidak",
    "jd" : "jadi",
}

def normalization(text):
    for word in normWords:
        text = text.replace(word, normWords[word])
    return text

data['text_cleaning'] = data['text_cleaning'].apply(lambda x: normalization(x))
data['text_cleaning'].head()
data.to_csv("normalizedData.csv", index=False)

#Tokenization

#Remove stopwords

#Stemming
