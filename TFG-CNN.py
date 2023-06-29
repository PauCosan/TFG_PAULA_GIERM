import codecs
import csv
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from transformers import AutoTokenizer
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Reshape, Conv2D, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import Adam

# Cargar los datos
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

# Convertir las etiquetas a un arreglo numpy-Permite realizar cálculos matemáticos y estadísticos de manera más
# eficiente
labels = np.array(labels)
labels_test = np.array(labels_test)

# Tokenizar los tweets y convertirlos en secuencias
tokenizer = Tokenizer(num_words=5000) #Tokenizer- para tokenizar el texto y convertirlo en secuencia de numeros
                                      #nº maximo que se utiliza en el tokenizador
tokenizer.fit_on_texts(all_tweets_text) #ajustar el tokenizer al conjunto de datos de entrenamiento
seq_train = tokenizer.texts_to_sequences(all_tweets_text) #convertir el texto en secuencia de numeros
seq_test = tokenizer.texts_to_sequences(all_tweets_text_test)
seq_train = keras.preprocessing.sequence.pad_sequences(seq_train, maxlen=100) #rellena secuencia con ceros para que
seq_test = keras.preprocessing.sequence.pad_sequences(seq_test, maxlen=100)#todas tengan igual longitud

# Cargar el tokenizer pre-entrenado de BERT
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#Guardar el tokenizer
tokenizer.save_pretrained('TokenizerCNN.h5')

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, y_train, y_test = train_test_split(seq_train, labels, test_size=0.2)
X_val = seq_test
y_val = labels_test

#Convertir los valores de etiquetas en numéricos. Al ser etiquetas categóricas se deben codificar numéricamente
#para que el modelo pueda procesarlas. Se ha usado codificación ordinal
le = LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)
y_val = le.transform(y_val)
y_test = le.transform(y_test)


# Crear la arquitectura del modelo CNN

embedding_dim = 100
vocab_size = len(tokenizer)
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=100))
model.add(Conv1D(100, 5, activation='relu'))
model.add(MaxPooling1D(pool_size=4))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compilar el modelo
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))
train_loss = history.history['loss']
train_acc = history.history['accuracy']
val_loss = history.history['val_loss']
val_acc = history.history['val_accuracy']
print("Entrenamiento - Loss: {:.2f} - Accuracy: {:.2f}%".format(train_loss[-1], train_acc[-1] * 100))
print("Validación - Loss: {:.2f} - Accuracy: {:.2f}%".format(val_loss[-1], val_acc[-1] * 100))

test_loss, test_acc = model.evaluate(X_test, y_test)

print("Prueba - Loss: {:.2f} - Accuracy: {:.2f}%".format(test_loss, test_acc * 100))

model.save('ModeloCNNv2.h5')
