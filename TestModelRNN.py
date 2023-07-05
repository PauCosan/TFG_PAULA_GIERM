import pickle
from tensorflow import keras
from keras.models import load_model

try:
    modelo_prueba = load_model('ModeloDePruebaCocheValidat.h5')
except OSError:
    print("Error: No se pudo cargar el modelo MLP")
    exit()
print(modelo_prueba.summary())

random_tweet ="incidente vial en la avenida por culpa de un coche"
print(random_tweet)

# Cargar el tokenizer desde el archivo
with open('tokenizerP2.pkl', 'rb') as file:
    tokenizer = pickle.load(file)


clean_random_tweet = tokenizer.texts_to_sequences([random_tweet])
clean_random_tweet = keras.preprocessing.sequence.pad_sequences(clean_random_tweet, maxlen=100)
print(clean_random_tweet)

# Hacer la predicción
prediction = modelo_prueba.predict(clean_random_tweet)
print(prediction)

# Comparar las probabilidades y tomar una decisión en función del umbral
if prediction[0][0] >= 0.5:
    print("El tweet es un accidente de tráfico")
else:
    print("El tweet no es un accidente de tráfico")

