import codecs
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Dense, Dropout
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

# Construir el modelo
model = Sequential()
model.add(Dense(64, input_dim=tweets_text.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compilar el modelo
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluar el modelo en el conjunto de prueba
score = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (score[1]*100))