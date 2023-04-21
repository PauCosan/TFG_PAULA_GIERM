import codecs
import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Flatten
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

# Dividir el conjunto de train en train y validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#Convertir los valores de eqtiquetas en numéricos
le = LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)
y_val = le.transform(y_val)
y_test = le.transform(y_test)

# Construir el modelo con una capa de Embedding
embedding_dim = 100
vocab_size = len(tokenizer.word_index) + 1

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=100))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compilar el modelo
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Evaluar el modelo en los conjuntos de entrenamiento, validación y prueba
train_loss, train_acc = model.evaluate(X_train, y_train)
val_loss, val_acc = model.evaluate(X_val, y_val)
test_loss, test_acc = model.evaluate(X_test, y_test)

# Imprimir los resultados
print("Entrenamiento - Loss: {:.2f} - Accuracy: {:.2f}%".format(train_loss, train_acc * 100))
print("Validación - Loss: {:.2f} - Accuracy: {:.2f}%".format(val_loss, val_acc * 100))
print("Prueba - Loss: {:.2f} - Accuracy: {:.2f}%".format(test_loss, test_acc * 100))
