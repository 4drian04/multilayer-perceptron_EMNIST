import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.metrics import Accuracy
from keras.models import Model, load_model
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from sklearn.metrics import ConfusionMatrixDisplay
import seaborn as sns
import plotly.graph_objects as go
import pandas as pd
from scipy.interpolate import griddata
from funciones_auxiliares_Adrian_Garcia_Garcia import transpose_and_flatten


# Los datos se procesan por lotes, y lo que nos permite esta constante es obtener antes de que se soliciten y tenerlos preparados,
# un número de datos que a TensorFlow le parezca correcto (normalmente igual o superior al número de datos anterior procesados)
AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 256 # Vamos a indicar que el número de lotes sea 256

# Escribimos las distitnas clases que le corresponden a EMNIST/bymerge
EMNIST_LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']

# # Cargar EMNIST/bymerge, separando los datos en entrenamiento y test
# El parámetro "as_supervised" indica que también se obtiene del dataset las etiquetas de cada imagen (si es un 7, una "S"...)
ds_train, ds_test = tfds.load(
    'emnist/bymerge',
    split=['train', 'test'],
    as_supervised=True
)

# Se mezclan los datos de entrenamiento con shuffle y se aplica las distintas transformaciones para poder entrenar el modelo correctamente
# el parámetro "num_parallel_calls" hace que se ejecute la función en paralelo usando los recursos del ordenador que esten disponibles,
# por otro lado, la función "batch", lo que hace es agrupar los datos por lotes que se le indiquen, en este caso 256,
# por lo que en lugar de pasarle un ejemplo al modelo, se le pasan lotes de 256 imagenes. Esto mejora la eficiencia.
# Por último, prefetch va preparando los datos siguientes, según analice TensorFlow (ya que es automático),
# pero mínimo debe ser la cantidad de los datos anteriores, es decir, mínimo 256

ds_train = ds_train.shuffle(1024).map(transpose_and_flatten, num_parallel_calls=AUTO).batch(BATCH_SIZE).prefetch(AUTO)

# Hacemos lo mismo con los datos de test, pero sin mezclarlos
ds_test = ds_test.map(transpose_and_flatten, num_parallel_calls=AUTO).batch(BATCH_SIZE).prefetch(AUTO)
VAL_BATCHES = int(0.1 * 697932) // BATCH_SIZE # Cogemos el 10% de los datos para obtener datos de validación para el entrenamiento
ds_val = ds_train.take(VAL_BATCHES) # Obtenemos el 10% de los datos de entrenamiento
ds_train_final = ds_train.skip(VAL_BATCHES) # Luego, el 90% restantes, serán los datos de entrenamientos finales que se utilizarán para los modelos

# Esto hace que, si cuando se entrena el modelo, en dos épocas (epoch, en este caso se indica con 'patience') no disminuye el valor del error
# se para el entrenamiento, ya que se da por hecho que se ha llegado a una estabilidad
# de esta manera, se evita el overfitting

early_stopper = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=2, verbose=0, mode='auto', restore_best_weights=True
)

# Entrenamos el mejor modelo, que como he explicado en el comentario anterior, es el de dos capas con 256 neuronas, Adam, una tasa de 0.001 y la función ReLU
inputs = Input(shape=(784,))
x = Dense(256,activation='relu')(inputs)
x = Dense(256, activation='relu')(x)
outputs = Dense(len(EMNIST_LABELS), activation='softmax')(x)
bestModel = Model(inputs=inputs, outputs=outputs)
optimizer = Adam(learning_rate=0.001)
bestModel.compile(optimizer, 'categorical_crossentropy', metrics=['acc'])
history = bestModel.fit(ds_train, epochs=5, validation_data=ds_val, callbacks=early_stopper)

# Guardamos el modelo entrenado con el formato ".h5" como se indica en el enunciado, para ello hacemos uso del método ".save"
# y le indicamos el nombre del archivo, siendo AGG mis iniciales
bestModel.save("Analizador_Alfanumerico_AGG.h5")

print("Predicciendo imágenes...")
# Creamos diferentes listas para ir guardando los resultados de la predicciçon
y_test = [] # Guarda las etiquetas o label de los test
y_preds = [] # Guarda las etiquetas de las imagenes predicha por el modelo
misclassified_images = [] # Guarda las imágenes que ha predicho erroneamente para mostrarlas posteriormente
misclassified_labels = [] # Se guarda el label real de la imagen
misclassified_preds = [] # Se guarda el label que ha predicho el modelo
for imgs, labels in ds_test: # En cada bucle, se procesa 256 imagenes, que es el batch definido al principio
    # Recordemos que los labels estaban en one hot, por lo que haciendo el "argmax" obtenemos el índice y lo pasaríamos a
    # la clase normal, por ejemplo el caso del 3 sería [0,0,0,1,0,0...] en este caso "argmax" devolvería 3, que es donde se encuentra el 1
    y_labels_batch = np.argmax(labels, axis=1)
    y_pred = bestModel.predict(imgs) # Hacemos la predicción
    y_predict_batch = np.argmax(y_pred, axis=1)
    for i in range(len(imgs)): # Recorremos las imagenes procesadas, esto recorre fila por fila del batch y cada fila es un vector de la imagen en cuestión
        if y_labels_batch[i] != y_predict_batch[i]: # Comprobamos si el modelo ha predecido mal
            # Guardamos la imagen original de vuelta a 28x28
            image = tf.reshape(imgs[i], (28,28)).numpy() # Convertimos la imagen de nuevo a una dimension 28x28 (como era originalmente)
            misclassified_images.append(image)
            misclassified_labels.append(y_labels_batch[i]) # Agregamos el valor real
            misclassified_preds.append(y_predict_batch[i]) # Añadimos el valor predicho
    # Añade elemento por elemento a la lista, es decir, supongamos que el batch que tenemos es este: [2,6,4,3], pues con extend añadiría uno por uno [1,6,3, 2,6,4,3]
    # en el caso del append, se añadiría como lista dentro de una lista ([[2,6,4,3]])
    y_test.extend(y_labels_batch)
    y_preds.extend(y_predict_batch)

# Calculamos el accuraccy de los valores predecidos
acc = Accuracy()
acc.update_state(y_test, y_preds)

print(f"La precición final del mejor modelo es de {acc.result().numpy()}")

# Calculamos la matriz de confusión normalizada (valores entre 0 y 1)
cm = confusion_matrix(y_true = y_test, y_pred = y_preds)
cm_normalize = normalize(cm, axis=1)

# Calculamos la matriz con valores numéricos
cmd = ConfusionMatrixDisplay(cm, display_labels=EMNIST_LABELS)

# Mostramos la matriz normalizada
sns.heatmap(cm_normalize, annot=False, xticklabels=EMNIST_LABELS, yticklabels=EMNIST_LABELS, linewidths=.1)
cmd.plot() # Se muestra la matriz de confusión numérica

plt.figure(figsize=(10,10))

# Mostramos cuatro diferentes casos de error
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(misclassified_images[i], cmap='gray')
    plt.title(f"Real: {EMNIST_LABELS[misclassified_labels[i]]} | Pred: {EMNIST_LABELS[misclassified_preds[i]]}")
    plt.axis('off')
plt.show()


# --------------------------------------

# Predicción imagen externa

# Con la función load_model de Keras, podemos cargar un modelo que tengamos guardado, en este caso, se carga un modelo que hemos guardado anteriormente
#bestModel = load_model('Analizador_Alfanumerico_AGG.h5')
# Aqui se indica la ruta de la imagen, en este caso, utilizaré ruta relativa
img_path = "q.png"
# Se carga la imagen con la librería Pillow
img = Image.open(img_path)

# Lo convertimos a escala de grises como las imagenes de emnist
img = img.convert("L")

# Redimensionamos la imagen a 28x28, como las imágenes de emnist
img = img.resize((28, 28))

# Lo convertimos a array numpy
img_array = np.array(img)

# Normalizar (igual que anteriormente)
img_array = img_array / 255.0

# Redimensionamos a 784 elementos
img_array = img_array.reshape(784,)

# Añadimos dimensión batch → (1, 784)
img_array = np.expand_dims(img_array, axis=0)

# Hacemos la predicción de nuestra imagen
pred = bestModel.predict(img_array)
pred_class = np.argmax(pred)

# Lo mostramos en un gráfico
plt.figure(figsize=(4, 4))
plt.imshow(img_array.reshape((28,28)), cmap="gray")
plt.title(f"Predicción del modelo: {EMNIST_LABELS[pred_class]}")
plt.axis("off")
plt.show()

df = pd.read_csv('mejores_resultados.csv')
df = (
    df.loc[
        df.groupby(['Capas', 'Unidades'])['ValAccuracy'].idxmax()
    ]
    .reset_index(drop=True)
)
x = [] # Esta lista va a contener las neuronas de la capa uno
y = [] # Esta lista va a contener las neuronas de la capa dos, si solo tiene una capa, se añadirá un 0
neuronas = df.Unidades
for neurona in neuronas: # Recorremos las neuronas
    neuronas_totales = neurona.split(",") # Si tiene dos capas, hacemos un split por la coma
    if(len(neuronas_totales)>1): # Si la longitud de lo devuelto es mayor que 1, quiere decir que tiene dos capas
        y.append(int(neuronas_totales[1])) # Por lo que se añade las neuronas a la lista de la capa dos
    else:
        y.append(0) # Si la longitud no es mayor que uno, quiere decir que solo hay una capa, por lo que añadimos 0 a la lista de la segunda capa
    x.append(int(neuronas_totales[0])) # Independientemente si hay o no segunda capa, añadimos en la lista de la primera capa las nerunoas de dicha capa
    # ya que siempre habrá neuronas en esa capa

# Convertimos los arrays en arrays de Numpy
x = np.array(x)
y = np.array(y)
z = np.array(df.ValAccuracy)
xi = np.linspace(x.min(), x.max(), 100) # Crea 100 puntos equispaciados entre el mínimo y máximo del array x (Neuronas primera capa)
yi = np.linspace(y.min(), y.max(), 100) # Crea 100 puntos equispaciados entre el mínimo y máximo del array y (Neuronas segunda capa)

X,Y = np.meshgrid(xi,yi) # Convierte xi e yi en matrices 2D

Z = griddata((x,y),z,(X,Y), method='cubic')
fig = go.Figure(go.Surface(x=xi,y=yi,z=Z)) # Crea la superficie 3D
fig.update_layout(
    title='Superficie de Val Accuracy según el número de neuronas',
    scene=dict(
        xaxis_title='Neuronas en la primera capa',
        yaxis_title='Neuronas en la segunda capa',
        zaxis_title='Val Accuracy'
    )
)
fig.show()