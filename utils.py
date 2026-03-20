import tensorflow as tf

# Escribimos las distitnas clases que le corresponden a EMNIST/bymerge
EMNIST_LABELS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
        'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']

# Esta función nos va a permitir normalizar los píxeles de la imagen entre 0 y 1, transponer la imagen,
# ya que estan invertidas horizontalmente y redimensionarlo a una dimensión, ya que el modelo necesita vectores de números
def transpose_and_flatten(image, label):
        image = tf.image.convert_image_dtype(image, dtype=tf.float32) # Se escala los píxeles de la imagen entre 0 y 1
        image = tf.transpose(image, [1,0,2]) # Rotamos la imagen para que tenga una orientación legible para el humano
        image = tf.reshape(image, shape=(784,)) # Redimensionamos la imagen a un vector de 784 elementos (28x28=784)

        label = tf.one_hot(label, depth=len(EMNIST_LABELS)) # Se hace one hot encoder al label

        return image, label