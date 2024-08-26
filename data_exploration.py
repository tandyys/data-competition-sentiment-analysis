import pandas as pd
import nltk #natural langauge toolkit
nltk.download('punkt')
from nltk.tokenize import word_tokenize as tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
data = pd.read_csv("mobil_listrik.csv")
data = data.dropna()
print(data)

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

firstData = filteredData.iloc[0]
print(firstData['text_cleaning'])

#Getting data information
print(data.shape)
data.info()

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
