import base64

import procesado_imagenes
import requests


def _ampliar_local(imagen, extension, factor_aumento): 
    imagen_int = procesado_imagenes.float_array_a_int_array(imagen)
    return procesado_imagenes.numpy_array_a_imagen_PIL_bytes(
        procesado_imagenes.aumentar_resolucion(imagen_int, factor=factor_aumento), extension)


def ampliar(imagen, extension, factor_aumento, url_api=None):
    """Amplia una imagen. Según el valor de url_api, la amplia con la API de EnhanceNet o devuelve la imagen ampliada
    con el algoritmo de ampliación bicúbica. También se amplia con el algoritmo bicúbico si no es posible acceder a la
    API o se presenta alguna excepción en el intento.

    Parámetros:
        Imagen: imagen en formato array de numpy con valores entre 0 y 1.

        Extensión: la extensión de la imagen anterior.

        Factor aumento: La proporción para aumentar la resolución de la imagen. Si se llama a la API de EnhanceNet,
        este factor es ignorado.

        Url_api: a qué url tenemos que realizar la petición de usarse la API. Si es None, no se utiliza la API.
        """
    if url_api is None:
        return _ampliar_local(imagen, extension, factor_aumento)
    else:
        # preprocesamos la imagen
        imagen_formato_PIL = procesado_imagenes.numpy_array_normalizado_a_imagen(imagen)

        imagen_base64 = procesado_imagenes.imagen_a_base64_string(imagen_formato_PIL)
        imagen_base64 = str(imagen_base64, 'utf-8')
        # creamos la peticion
        payload = {"imagen": imagen_base64}

        try:
            # intentamos redimensionar con la api y si hay errores en el intento realizamos el metodo bicubico
            resultado_peticion = requests.post(url_api, json=payload)
            resultado_peticion = resultado_peticion.json()
            imagen_redimensionada_bytes = base64.b64decode(resultado_peticion["imagen_ampliada"])
        except:
            return _ampliar_local(imagen, extension, factor_aumento)

        return imagen_redimensionada_bytes
