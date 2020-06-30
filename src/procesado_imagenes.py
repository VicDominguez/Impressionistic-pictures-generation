import base64
import numpy as np
import tensorflow as tf
import io
from PIL import Image


def imagen_a_base64_string(imagen, formato="JPEG"):
    """Convierte un objeto de tipo PIL Image a un string base64"""
    buffer = io.BytesIO()
    imagen.save(buffer, format=formato)
    return base64.b64encode(buffer.getvalue())


def numpy_array_a_imagen(numpy_array):
    numpy_array = np.squeeze(numpy_array, axis=0)
    numpy_array *= 255.0
    numpy_array = (numpy_array.clip(0, 255) + 0.5).astype(np.uint8)
    return Image.fromarray(numpy_array)


def leer_y_normalizar_imagen(ruta, ancho, alto, canales, train=False):
    """Primitiva que lee y prepocesa una imagen según si es para entrenar o no"""
    imagen = tf.io.read_file(ruta)  # Abrimos el archivo
    # Convertir el string a un tensor 3D uint8
    imagen = tf.image.decode_jpeg(imagen, channels=canales)
    imagen = tf.image.resize(imagen, size=(ancho, alto), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    if train:  # si no es test espejamos aleatoriamente
        imagen = tf.image.random_flip_left_right(imagen)
    imagen = tf.image.convert_image_dtype(imagen, tf.float32)
    return imagen


def preprocesar_imagen(ruta, ancho, alto, canales):
    """Lee y prepocesa una imagen a partir de su ruta"""
    imagen = leer_y_normalizar_imagen(ruta, ancho, alto, canales, train=False)
    return tf.expand_dims(imagen, 0)
