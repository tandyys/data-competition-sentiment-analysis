import pandas as pd
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize as tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
data = pd.read_csv("mobil_listrik.csv")
cleanedData = data.dropna()
print(cleanedData)

strToFind = "saran"
filteredData = cleanedData[cleanedData['text_cleaning'].str.contains(strToFind, case=False)]
print(filteredData)

tokens = tokenizer(filteredData.iloc[0]['text_cleaning'])
print(tokens)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(tokens)
print(X.toarray())
data_575 = filteredData.iloc[7]
print(data_575)

firstData = filteredData.iloc[0]
print(firstData['text_cleaning'])
