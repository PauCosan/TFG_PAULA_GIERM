from tensorflow import keras
from keras.models import load_model
from keras.datasets import imdb
from keras.preprocessing import sequence

try:
    modelo_RNN = load_model('ModeloDePrueba.h5')
except OSError:
    print("Error: No se pudo cargar el modelo RNN")
    exit()

random_tweet = "a todos les gusto la peli pero a mi no"
print(random_tweet)

# Cargar el tokenizer IMDb y configurarlo para mantener las 5000 palabras principales
top_words = 5000
(X_train, _), (_, _) = imdb.load_data(num_words=top_words)
tokenizer = imdb.get_word_index()

# Tokenizar el tweet y convertirlo en secuencia
tweet_tokens = []
for word in random_tweet.split():
    index = tokenizer.get(word.lower(), 0) + 3
    if index > top_words:
        index = 2  # Palabra desconocida
    tweet_tokens.append(index)
tweet_tokens = keras.preprocessing.sequence.pad_sequences([tweet_tokens], maxlen=500)
print(tweet_tokens)

# Hacer la predicción
prediction = modelo_RNN.predict(tweet_tokens)
print(prediction)

# Comparar las probabilidades y tomar una decisión en función del umbral
if prediction[0][0] >= 0.5:
    print("sentimiento positivo")
else:
    print("sentimiento negativo")
