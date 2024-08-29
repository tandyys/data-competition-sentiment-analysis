import pandas as pd
import nltk #natural langauge toolkit
nltk.download('punkt')
from nltk.tokenize import word_tokenize as tokenizer
from nltk.stem import PorterStemmer as ps
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import spacy as spy

#Read the data
data = pd.read_csv("mobil_listrik.csv")
data.shape
data = data.dropna()
data = data.drop_duplicates()
data.head()

#Try to combine all the text data into one
words = data['text_cleaning'].tolist()
print(words)
joinedWords = " ".join(words).split()
setWords = set(joinedWords)
print(setWords)

totalSetWords = len(setWords)
print(totalSetWords)
print(len(joinedWords))

#I'm trying to get all the unique words from the data
for word in setWords:
    print(f"{word}")
#but it seems can't be done because the data is too big???

#Check missing and duplicate values
totalDuplicateData = data.duplicated().sum()
print(f"Total duplicated data : {totalDuplicateData}")
data.isnull().sum()
print(data.head())
data.shape
uniqueSymbols = data["text_cleaning"].unique()
print(uniqueSymbols)
uniqueSymbols.shape
uniqueSymbols2 = set(data["text_cleaning"])
print(uniqueSymbols2)

#find the data that contains the word "saran"
strToFind = "saran"
filteredData = data[data['text_cleaning'].str.contains(strToFind, case=False)]
print(filteredData)
filteredData["text_cleaning"].values[9]

#Coba bikin tokenizer
tokens = tokenizer(filteredData.iloc[0]['text_cleaning'])
print(tokens)
vectorizer = TfidfVectorizer()

#Bikin vectorizer buat cari frequency kata yang ada didalam data
X = vectorizer.fit_transform(tokens)
print(X.toarray())
data_575 = filteredData.iloc[7]
print(data_575)

firstData = data.iloc[0]
print(firstData['text_cleaning'])
print(firstData)

#Getting data information
print(data.shape)
data.info()

#Sentiment mapping to numeric
sentimentMapping = {
    "positif" : 0,
    "negatif" : 1,
    "netral" : 2
}
data["sentimen"] = data["sentimen"].map(sentimentMapping)
print(data.head())

sentimenPlot = data["sentimen"].value_counts().sort_index()
print(sentimenPlot)

sentimen_bar_plt = sentimenPlot.plot(kind='bar', title='Sentimen Distribution', figsize=(10, 5))
sentimen_bar_plt.set_xlabel('0: Positif, 1: Negatif, 2: Netral')

#Cobain Spacy
nlp_id = spy.load('id_core_news_sm') #Error id_core_news_sm not found
testData = firstData['text_cleaning']

#Try VADER Approach
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
from tqdm.notebook import tqdm
sia = SentimentIntensityAnalyzer()
sia.polarity_scores(testData)
#Kesimpulan saat ini, VADER gabisa untuk bahas indonesia.
#Test bhs inggris dibawah
sia.polarity_scores("i don't like this music!")
sia.polarity_scores("for me, electric vehicle is just not convenient because of the price.")

#For the second test using english, the result is not good.
#I think VADER just not the right approach to do sentiment analysis, the result's accuracy isn't consistent

from collections import Counter
from wordcloud import WordCloud

commonWords = Counter(" ".join(data["text_cleaning"]).split())
wc = WordCloud(width=800, height=400, background_color='black').generate_from_frequencies(dict(commonWords))
plt.figure(figsize=(15, 15))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

uniqueWords = set(" ".join(data["text_cleaning"]).split())
for i in uniqueWords:
    print(i)

#Find the data that contains the word "sawit"
wordToFind_5 = "sawit"
filteredData_6 = data[data['text_cleaning'].str.contains(wordToFind_5, case=False)]
print(filteredData_6)
print(filteredData_6["text_cleaning"].values[0])
print(filteredData_6["text_cleaning"].values[1])
print(filteredData_6["text_cleaning"].values[2])
#Find the data that contains the word "beli"
wordToFind = "beli"
filteredData_2 = data[data['text_cleaning'].str.contains(wordToFind, case=False)]
print(filteredData_2)
#Find the data that contains the word "harga"
wordToFind_2 = "harga"
filteredData_3 = data[data['text_cleaning'].str.contains(wordToFind_2, case=False)]
print(filteredData_3)
#Find the word that contains the word "mobil"
wordToFind_3 = "mobil"
filteredData_4 = data[data['text_cleaning'].str.contains(wordToFind_3, case=False)]
print(filteredData_4)
#Find the data that contains the word "beli", "harga", and "mobil"
wordToFind_4 = ["beli", "harga", "mobil"]
pattern = r'\b' + r'\b|\b'.join(wordToFind_4) + r'\b'
pattern = pattern.replace(r'\b|\b', r'\b|\b')
filteredData_5 = data[data['text_cleaning'].str.contains(pattern, case=False, regex=True)]
print(filteredData_5)
print(filteredData_5["text_cleaning"].values[483])

#Preprocessing on 3rd cycle
#Normalisasi

#Stopword removal -> remove words that doesn't have any meaning ex: "dan", "atau", "yang"
#Indo using Sastrawi & English using NLTK.corpus import stopwords

#Translate the data

#Tokenization -> split the sentence into words

#Stemming

print("Hello world...")