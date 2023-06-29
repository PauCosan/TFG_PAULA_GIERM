from tensorflow import keras
from keras.models import load_model
from transformers import AutoTokenizer

try:
    modelo_RNN = load_model('ModeloRNNv7.h5')
except OSError:
    print("Error: No se pudo cargar el modelo RNN")
    exit()

random_tweet ="choque coche y moto"
print(random_tweet)

# Cargar el tokenizer desde el archivo guardado
tokenizer = AutoTokenizer.from_pretrained('TokenizerRNN.h5')

# Tokenizar los tweets y convertirlos en secuencias
tweet_tokens = tokenizer.encode(random_tweet, add_special_tokens=True, max_length=100, truncation=True)
print(tweet_tokens)
tweet_tokens = keras.preprocessing.sequence.pad_sequences([tweet_tokens], maxlen=100)
print(tweet_tokens)
# Hacer la predicción
prediction = modelo_RNN.predict(tweet_tokens)
print(prediction)

# Comparar las probabilidades y tomar una decisión en función del umbral
if prediction[0][0] >= 0.5:
    print("El tweet es un accidente de tráfico")
else:
    print("El tweet no es un accidente de tráfico")


