import codecs
import csv
import pickle
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from tensorflow import keras
from keras.layers import Dense, Embedding, Conv1D, GlobalMaxPooling1D, Dropout
from keras.models import Sequential

# Cargar los datos para train
with codecs.open('1-classification-trainset.tsv', 'r', encoding='utf-8') as tsvfile:
    lector_tsv = csv.reader(tsvfile, delimiter='\t')
    tweets_train = [fila for fila in lector_tsv]
# Cargar los datos para test
with codecs.open('1-classification-testset.tsv', 'r', encoding='utf-8') as tsvfile:
    lector_tsv = csv.reader(tsvfile, delimiter='\t')
    tweets_test = [fila for fila in lector_tsv]


# Convertir a DataFrame de la librería Pandas de Python. Pasamos a una estructura de datos bidimensional para poder
# almacenar en filas y columnas
tweets_train = pd.DataFrame(tweets_train[1:], columns=tweets_train[0])
tweets_test = pd.DataFrame(tweets_test[1:], columns=tweets_test[0])

# Obtener tweets y etiquetas. Separo las etiquetas de los textos
all_tweets_text = tweets_train['text'].values
labels = tweets_train['label'].values
all_tweets_text_test = tweets_test['text'].values
labels_test = tweets_test['label'].values
print(all_tweets_text_test[1])
print(labels_test[1])
# Convertir las etiquetas a un arreglo numpy-Permite realizar cálculos matemáticos y estadísticos de manera más
# eficiente
labels = np.array(labels)
labels_test = np.array(labels_test)

def clean_tweet(tweet):
    # Eliminar URLs
    tweet = re.sub(r'http\S+|www.\S+', '', tweet)

    # Eliminar caracteres no alfanuméricos excepto interrogaciones y exclamaciones
    tweet = re.sub(r'[^a-zA-Z0-9\s¿?¡!áéíóúÁÉÍÓÚ]', '', tweet)

    # Eliminar espacios en blanco adicionales
    tweet = tweet.strip()

    return tweet

# Limpiar cada tweet de train y test
clean_tweet_train = [clean_tweet(tweet) for tweet in all_tweets_text]
clean_tweet_test = [clean_tweet(tweet) for tweet in all_tweets_text_test]
print(clean_tweet_test[1])

"""
# Imprimir las frases limpias
for tweet_limpio in tweets_limpios:
    print(tweet_limpio)
"""

# Tokenizar los tweets y convertirlos en secuencias
tokenizer = Tokenizer(num_words=5000) #Tokenizer- para tokenizar el texto y convertirlo en secuencia de numeros
                                      #nº maximo que se utiliza en el tokenizador (solo 5000 palanras + frecuentes)

tokenizer.fit_on_texts(clean_tweet_train) #ajustar el tokenizer al conjunto de datos de entrenamiento
                                          #asigna los índices a cada palabra en función de su vocabulario

seq_train = tokenizer.texts_to_sequences(clean_tweet_train) #convertir el texto en secuencia de numeros
seq_test = tokenizer.texts_to_sequences(clean_tweet_test)

seq_train = keras.preprocessing.sequence.pad_sequences(seq_train, maxlen=100) #rellena secuencia con ceros para que
seq_test = keras.preprocessing.sequence.pad_sequences(seq_test, maxlen=100)#todas tengan igual longitud

print(seq_test[1])
"""
# Guardar el tokenizer en un archivo
with open('tokenizerNeuralNetworkCNN.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)
"""
# Dividir los datos en conjunto de entrenamiento y conjunto de validacion
X_train, X_val, y_train, y_val = train_test_split(seq_train, labels, test_size=0.2)
X_test = seq_test
y_test = labels_test

#Convertir los valores de etiquetas en numéricos. Al ser etiquetas categóricas se deben codificar numéricamente
#para que el modelo pueda procesarlas. Se ha usado codificación ordinal
le = LabelEncoder()
le.fit(y_train)                     #ajusta a las etiquetas de entrenamiento
y_train = le.transform(y_train)
y_val = le.transform(y_val)
y_test = le.transform(y_test)

top_words = 5000
max_review_length = 100

# create the model RNN
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Conv1D(64, 5, activation='relu'))
model.add(Dropout(0.5)) #p2
model.add(Conv1D(128, 3, activation='relu')) #p2
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3, batch_size=64)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

model.save('ModelNeuralNetworkCNN.h5')


