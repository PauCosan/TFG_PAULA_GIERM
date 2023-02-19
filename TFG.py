# Importar las librerías necesarias
import csv
import codecs
import pandas as pd
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

# Cargar los datos

with codecs.open('AccidentsTweets.tsv', 'r', encoding='utf-8') as tsvfile:
    lector_tsv = csv.reader(tsvfile, delimiter='\t')
    tweets = [fila for fila in lector_tsv]
    #text_column = [row[13] for row in tweets]
    #print(text_column)

# Convertir a DataFrame
tweets = pd.DataFrame(tweets[1:], columns=tweets[0])

# Seleccionar la columna 13 y guardarla en una variable
text_column = tweets.iloc[:, 13:14]

# Preprocesar los datos
stopwords = nltk.corpus.stopwords.words('spanish')
tweets['cleaned_text'] = text_column['text'].apply(lambda x: ' '.join([word for word in x.split()]))

# Mostrar la columna 13
print(tweets['text'])
""""
# Extraer características
vectorizer = CountVectorizer()
features = vectorizer.fit_transform(tweets['cleaned_text'])

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features, tweets['label'], test_size=0.2)

# Entrenar el modelo
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Evaluar el modelo
y_pred = lr.predict(X_test)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print('Precision: {:.2f}'.format(precision))
print('Recall: {:.2f}'.format(recall))
"""