import codecs
import csv
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, SimpleRNN, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import Adam

# Cargar los datos
with codecs.open('1-classification-trainset.tsv', 'r', encoding='utf-8') as tsvfile:
    lector_tsv = csv.reader(tsvfile, delimiter='\t')
    tweets = [fila for fila in lector_tsv]

# Convertir a DataFrame
tweets = pd.DataFrame(tweets[1:], columns=tweets[0])

# Obtener tweets y etiquetas
tweets_text = tweets['text'].values
labels = tweets['label'].values

# Convertir las etiquetas a un arreglo numpy
labels = np.array(labels)

# Tokenizar los tweets y convertirlos en secuencias
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(tweets_text)
tweets_text = tokenizer.texts_to_sequences(tweets_text)
tweets_text = keras.preprocessing.sequence.pad_sequences(tweets_text, maxlen=100)

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(tweets_text, labels, test_size=0.2)

#Convertir los valores de eqtiquetas en numéricos
le = LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)
y_test = le.transform(y_test)

# Crear la arquitectura del modelo RNN
embedding_dim = 100
vocab_size = len(tokenizer.word_index) + 1

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=100))
model.add(SimpleRNN(64))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compilar el modelo
optimizer = Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=10, batch_size=64)

# Evaluar el modelo en el conjunto de prueba
score = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (score[1]*100))