import tensorflow as tf
import numpy as np

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype = float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype = float)

oculta1 = tf.keras.layers.Dense(units = 3, input_shape = [1])
oculta2 = tf.keras.layers.Dense(units = 3)
salida = tf.keras.layers.Dense(units = 1)
modelo = tf.keras.Sequential([oculta1, oculta2, salida])
modelo.compile( optimizer = tf.keras.optimizers.Adam(0.1), loss = 'mean_squared_error')
print("Entrenando   ...   ")
historial = modelo.fit(celsius, fahrenheit, epochs = 1000, verbose = False)
print("Modelo entrenado!")
import matplotlib.pyplot as plt
plt.xlabel("Tiempo")
plt.ylabel("Perdida")
plt.plot(historial.history["loss"])
print("Conversión")
resultado = modelo.predict([30.0])
print("El resultado es " + str(resultado) + " fahrenheit!")
print("Variables del modelo")
print(oculta1.get_weights())
print(oculta2.get_weights())
print(salida.get_weights())
