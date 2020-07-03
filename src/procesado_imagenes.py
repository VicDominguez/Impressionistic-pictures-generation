import base64
import imghdr

import numpy as np
import tensorflow as tf
import io
from PIL import Image


def comprueba_imagen(string_base64):
    """Comprueba si la imagen es válida y si lo es devuelve el tipo correspondiente.

    Parámetros:
        string_base64 : imagen en forma de string base64."""
    try:
        resultado = imghdr.what("ac", h=base64.b64decode(str(string_base64)))
    except:
        resultado = None
    return resultado


def imagen_a_base64_string(imagen, formato="JPEG"):
    """Convierte una imagen a un string_base64

    Parámetros:
        imagen: imagen en formato PIL Image.
        formato: parámetro opcional que indica la extensión de la imagen deseada."""
    buffer = io.BytesIO()
    imagen.save(buffer, format=formato)
    return base64.b64encode(buffer.getvalue())


def numpy_array_normalizado_a_imagen(numpy_array):
    """Convierte un array de valores entre 0 y 1 a una imagen en forma de objeto PIL.

    Parámetros:
        numpy_array: imagen en forma de numpy array. Se espera que los valores estén entre 0 y 1."""
    return Image.fromarray(float_array_a_int_array(numpy_array))


def float_array_a_int_array(numpy_array):
    """Convierte un array de valores en el rango 0-1 a un array de enteros en el rango 0-255

    Parámetros:
        numpy_array: el propio array a convertir"""
    resultado = np.copy(numpy_array)
    if len(resultado.shape) == 4:
        resultado = np.squeeze(resultado, axis=0)
    resultado *= 255.0
    return (resultado.clip(0, 255) + 0.5).astype(np.uint8)


def numpy_array_a_imagen_PIL_bytes(imagen, extension):
    """Convierte un array en una imagen guardada como array de bytes.

    Parámetros:
        imagen: el propio array que contiene la imagen. Puede venir como un tensor en 3 o 4 dimensiones.
        extension: la extensión deseada en forma de string."""

    if len(tf.shape(imagen)) == 4:
        imagen = imagen.squeeze(axis=0)  # Quitamos el primer eje para poder operar con ella
    imagen = (imagen.clip(0, 255)).astype(np.uint8)
    imagen_PIL = Image.fromarray(imagen)
    imagen_bytes = io.BytesIO()
    imagen_PIL.save(imagen_bytes, format=extension.upper())
    return imagen_bytes.getvalue()


def aumentar_resolucion(imagen, factor=4):
    """Reescala una imagen. Comparativa de métodos en
    https://www.tensorflow.org/api_docs/python/tf/image/resize

    Parámetros:
        imagen: el tensor que contiene la imagen. Puede venir en 3 o 4 dimensiones.
        factor: proporción en la que se amplia la imagen.
    """
    if len(tf.shape(imagen)) == 4:
        ancho = imagen.shape[1]
        alto = imagen.shape[2]
    else:
        ancho = imagen.shape[0]
        alto = imagen.shape[1]
    return tf.image.resize(imagen, [factor * ancho, factor * alto], method=tf.image.ResizeMethod.BICUBIC).numpy()


def preprocesar_imagen(ruta, dimensiones):
    """Preprocesa una imagen y la devuelve como un tensor.

    Parámetros:
        ruta: ruta en disco de la imagen.
        dimensiones: dimensiones deseadas para la imagen."""
    imagen = tf.io.read_file(ruta)  # Abrimos el archivo
    # Convertir el string a un tensor 3D uint8
    imagen = tf.image.decode_jpeg(imagen, channels=dimensiones[2])
    # Tenemos que cortar la imagen. Desgraciadamente no se puede mantener el ratio del aspecto
    imagen = tf.image.resize(imagen, size=dimensiones[0:2], method=tf.image.ResizeMethod.AREA)
    return tf.image.convert_image_dtype(imagen / 255, tf.float32)


def preprocesar_imagen_individual(ruta, dimensiones):
    """Lee y prepocesa una imagen a partir de su ruta pero apto para usrse en producción.
    Parámetros:
        ruta: ruta en disco de la imagen.
        dimensiones: dimensiones deseadas para la imagen."""
    imagen = preprocesar_imagen(ruta, dimensiones)
    return tf.expand_dims(imagen, 0)  # Añadimos la dimension batch
