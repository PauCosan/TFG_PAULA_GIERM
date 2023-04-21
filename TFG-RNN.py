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

# Dividir el conjunto de train en train y validation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

#Convertir los valores de eqtiquetas en numéricos
le = LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)
y_val = le.transform(y_val)
y_test = le.transform(y_test)

# Crear la arquitectura del modelo RNN
embedding_dim = 100
vocab_size = len(tokenizer.word_index) + 1

model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=100))
model.add(SimpleRNN(64, return_sequences=True))
model.add(SimpleRNN(32))
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

"""
#COMPROBACIÓN CON TWEET
# Preprocesar el tweet
tweet = "Accidente de coche en la autopista dirección a miami"
tweet = tokenizer.texts_to_sequences([tweet])
tweet = keras.preprocessing.sequence.pad_sequences(tweet, maxlen=100)

# Hacer la predicción
prediction = model.predict(tweet)

# Comparar las probabilidades y tomar una decisión en función del umbral
if prediction[0][0] >= 0.5:
    print("El tweet es un accidente de tráfico")
else:
    print("El tweet no es un accidente de tráfico")
"""