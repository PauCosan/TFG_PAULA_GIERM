import codecs
import csv
import re

import unidecode
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Cargar los datos
with codecs.open('1-classification-trainset.tsv', 'r', encoding='utf-8') as tsvfile:
    lector_tsv = csv.reader(tsvfile, delimiter='\t')
    tweets = [fila for fila in lector_tsv]

# Convertir a DataFrame
tweets = pd.DataFrame(tweets[1:], columns=tweets[0])

# Obtener tweets y etiquetas
tweets_text = tweets['text'].values
labels = tweets['label'].values

stop_words = stopwords.words('spanish')
stemmer = SnowballStemmer('spanish')

def clean_tweet(tweet):
    tweet = tweet.lower()  # Convertir minus
    tweet = re.sub(r"http\S+", "", tweet)  # Eliminar URLs
    tweet = re.sub(r"[^a-zA-Záéíóúñ]", " ", tweet)  # Eliminar caracteres no alfabéticos
    tweet = tweet.split()  # Tokenizar
    tweet = [word if word in ["siniestro","incidente","accidente","autopista","autovia"] else stemmer.stem(word) for word in tweet if
             not word in stop_words]  # Stemming con SnowballStemmer
    tweet = " ".join(tweet)  # Unir tokens
    return tweet


cleaned_tweets = [clean_tweet(tweet) for tweet in tweets_text]
#print(cleaned_tweets)
#print(labels)


# Crear matriz TF-IDF
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned_tweets)

# Entrenar el modelo Naive Bayes
nb_model = MultinomialNB().fit(tfidf_matrix, labels)

# Clasificar un tweet aleatorio
tweet = "Hoy hubo un accidente en la autopista"
tweet_cleaned = clean_tweet(tweet)
print(tweet_cleaned)
tweet_tfidf = tfidf_vectorizer.transform([tweet_cleaned])
prediction = nb_model.predict(tweet_tfidf)
print(prediction)