import base64
import imghdr

import requests
import tensorflow as tf
from PIL import Image
import io
import numpy as np
import procesado_imagenes  # TODO quitar esto


def comprueba_imagen(string_base64):
    """Comprueba si la imagen es válida y devuelve el tipo correspondiente"""
    try:
        resultado = imghdr.what("ac", h=base64.b64decode(str(string_base64)))
    except:
        resultado = None
    return resultado

def ampliar(imagen,extension, url_api):
    # preprocesamos la imagen
    imagen_formato_PIL = procesado_imagenes.numpy_array_a_imagen(imagen)

    imagen_base64 = procesado_imagenes.imagen_a_base64_string(imagen_formato_PIL)
    imagen_base64 = str(imagen_base64, 'utf-8')
    # creamos la peticion
    payload = {"imagen": imagen_base64}
    try:
        resultado_peticion = requests.post(url_api, json=payload)
        # if resultado_peticion.ok:
        resultado_peticion = resultado_peticion.json()
        imagen_redimensionada_bytes = base64.b64decode(resultado_peticion["imagen_ampliada"])
    except:
        # TODO comprobar si es un tensor o un array de numpy lo que entra y variar en consecuencia shape
        # TODO ver si la api devuelve los bytes de una PIL imagen o no
        imagen = imagen.squeeze(axis=0)
        imagen_redimensionada = tf.image.resize(imagen, [4 * imagen.shape[0], 4 * imagen.shape[1]],
                                                method=tf.image.ResizeMethod.BICUBIC)
        imagen_redimensionada = imagen_redimensionada.numpy()
        imagen_redimensionada = (imagen_redimensionada.clip(0, 255)).astype(np.uint8)
        imagen_redimensionada_PIL = Image.fromarray(imagen_redimensionada)
        imagen_redimensionada_bytes = io.BytesIO()
        imagen_redimensionada_PIL.save(imagen_redimensionada_bytes, format=extension.upper())  # TODO meter aqui la extension de la imagen
        imagen_redimensionada_bytes = imagen_redimensionada_bytes.getvalue()
    return imagen_redimensionada_bytes
