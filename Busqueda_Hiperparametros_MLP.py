import tensorflow as tf
import tensorflow_datasets as tfds
from keras.layers import Input, Dense
from keras.optimizers import Adam, SGD, RMSprop
from keras.models import Model
import time
from funciones_auxiliares_Adrian_Garcia_Garcia import transpose_and_flatten

# Los datos se procesan por lotes, y lo que nos permite esta constante es obtener antes de que se soliciten y tenerlos preparados,
# un número de datos que a TensorFlow le parezca correcto (normalmente igual o superior al número de datos anterior procesados)
AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 256 # Vamos a indicar que el número de lotes sea 256

# Escribimos las distitnas clases que le corresponden a EMNIST/bymerge
EMNIST_LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']

# Creamos los distintos parámetros a probar
nums_layers = [(128,), (256,), (256, 128), (256, 256)] # Guardamos las distintas capas y neuronas
activation_functions = ['relu', 'tanh', 'sigmoid'] # Guardamos las distintas funciones de activación
learning_rates = [0.001, 0.01] # Guardamos las distintas tasas de aprendizaje
optimizers = [Adam, SGD, RMSprop] # Guardamos los distintos optimizadores

# Construye el modelo con las caacterísticas pasadas como parámetro
def build_model(num_layers=(256,128), activation_function='relu'):
    inputs = Input(shape=(784,)) # Se crea la capa de entrada de un matriz de dimensión 784
    x = inputs
    for neurons in num_layers: # En este bucle se crean tantas capas ocultas como se le pase por parámetro
        x = Dense(neurons, activation=activation_function)(x) # Se crea la capa oculta con las neuronas correspondientes
    outputs = Dense(len(EMNIST_LABELS), activation='softmax')(x) # Se crea la capa de salida según el número de clases que devuelve emnist
    model = Model(inputs=inputs, outputs=outputs) # Se crea el modelo con las capas creadas anteriormente
    return model

# Cargar EMNIST/bymerge, separando los datos en entrenamiento y test
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
results = []

# Recorremos los diferentes parámetros para ver cual es el mejor modelo
for num_layer in nums_layers:
    for activation_function in activation_functions:
        for learning_rate in learning_rates:
            for optimizerClass in optimizers:
                # Indicamos que parámetros se probarán
                print("PROBANDO CON PARÁMETROS: ")
                print(f"Numero de capas y neuronas: {num_layer}")
                print(f"FUNCIÓN DE ACTIVACIÓN: {activation_function}")
                print(f"TASA DE APRENDIZAJE: {learning_rate}")
                print(f"OPTIMIZADOR: {optimizerClass.__name__}")
                model = build_model(num_layer, activation_function) # Construimos el modelo
                optimizer = optimizerClass(learning_rate=learning_rate) # Creamos el optimizador
                # COnfiguramos el modelo con el optimizador e indicamos que queremos obtener el accuracy y la precisión
                model.compile(optimizer, 'categorical_crossentropy', metrics=['acc', tf.keras.metrics.Precision(name='precision')])
                # Calculamos el tiempo que tarda para indicarlo en la tabla del pdf
                start = time.time()
                # Entrenamos el modelo con 3 iteraciones, y le pasamos los datos de validación y el callback para que pare cuando el modelo se estabilice
                history = model.fit(ds_train_final, epochs=3, validation_data=ds_val, callbacks=early_stopper)
                end = time.time()
                # Guardamos las métricas en la lista de resultados
                val_acc = history.history['val_acc'][-1]
                val_loss = history.history['val_loss'][-1]
                val_precision = history.history['val_precision'][-1]
                epochs_to_converge = len(history.history['loss'])
                # Calculamos el tiempo que ha tardado en entrenar el modelo
                training_time = end - start
                results.append({
                    'layers': len(num_layer),
                    'units': num_layer,
                    'activation': activation_function,
                    'val_acc': val_acc,
                    'val_prec': val_precision,
                    'val_loss': val_loss,
                    'lr': learning_rate,
                    'optimizer': optimizerClass.__name__,
                    'epoch_to_converge': epochs_to_converge,
                    'training_time': training_time
                })

print("A continuación se muestran todos los resultados: ")
print(results)
best = max(results, key=lambda x: x['val_acc'])
print("Mejor configuración:")
print(best)

# Aqui se muestra cual es la mejor configuración, de 5 veces que lo he entrenado 3 ha sido mejor el de dos capas de 256 neuronas cada una
# con el optimizador Adam, una tasa de aprendizaje de 0.001, y la función ReLU, las otras 2 es lo mismo que el anterior pero una capa era de 256 neurona y otra de 128
# Por lo que es posible que, si lo ejecutas, salga una u otra, pero voy a elegir el de 256 neuronas cada capa

print(f"Podemos ver que la configuración que devuelve mejor accuracy es la que tiene {best['layers']} capas ocultas de {best['units'][0]} neuronas")
print(f"Con la función {best['activation']} con una tasa de aprendizaje de {best['lr']} y el optimizador es {best['optimizer']}")
print(f"Con una precision de {best['val_prec']} y un error de {best['val_loss']}")