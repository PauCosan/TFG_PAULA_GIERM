from tensorflow import keras
from keras.models import load_model
from transformers import AutoTokenizer

try:
    modelo_MLP = load_model('ModeloMLPv5.h5')
except OSError:
    print("Error: No se pudo cargar el modelo MLP")
    exit()

random_tweet ="choque coche y moto"
print(random_tweet)

# Cargar el tokenizer desde el archivo guardado
tokenizer = AutoTokenizer.from_pretrained('Tokenizer.h5')


# Tokenizar los tweets y convertirlos en secuencias
tweet_tokens = tokenizer.encode(random_tweet, add_special_tokens=True, max_length=100, truncation=True)
tweet_tokens = keras.preprocessing.sequence.pad_sequences([tweet_tokens], maxlen=100)

# Hacer la predicción
prediction = modelo_MLP.predict(tweet_tokens)
print(prediction)

"""
# Tokenizar los tweets y convertirlos en secuencias
tweet_text = tokenizer.texts_to_sequences(random_tweet)
tweet_text = keras.preprocessing.sequence.pad_sequences(tweet_text, maxlen=100)

# Hacer la predicción
prediction = modelo_MLP.predict(tweet_text)
print(prediction)
"""
# Comparar las probabilidades y tomar una decisión en función del umbral
if prediction[0][0] >= 0.5:
    print("El tweet es un accidente de tráfico")
else:
    print("El tweet no es un accidente de tráfico")