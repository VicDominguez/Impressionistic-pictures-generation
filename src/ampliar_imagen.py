import base64

import requests
import tensorflow as tf

import src.procesado_imagenes  # TODO quitar esto


def ampliar(imagen, url_api):
    # preprocesamos la imagen
    imagen_formato_PIL = src.procesado_imagenes.numpy_array_a_imagen(imagen)
    imagen_base64 = src.procesado_imagenes.imagen_a_base64_string(imagen_formato_PIL)
    imagen_base64 = str(imagen_base64, 'utf-8')
    # creamos la peticion
    payload = {"imagen": imagen_base64}
    resultado_peticion = requests.post(url_api, json=payload)
    if resultado_peticion.ok:
        resultado_peticion = resultado_peticion.json()
        imagen_redimensionada_bytes = base64.b64decode(resultado_peticion["imagen_ampliada"])
        return imagen_redimensionada_bytes
    else:
        # TODO comprobar si es un tensor o un array de numpy lo que entra y variar en consecuencia shape
        # TODO ver si la api devuelve los bytes de una PIL imagen o no
        imagen_redimensionada = tf.image.resize_images(imagen, [4 * imagen.shape[0], 4 * imagen.shape[1]],
                                                       method=tf.image.ResizeMethod.BICUBIC)
        return None
