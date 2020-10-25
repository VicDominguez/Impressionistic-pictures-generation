import tensorflow as tf
import src.utilidades


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
    return tf.image.convert_image_dtype(imagen / 255.0, tf.float32)


def preprocesar_imagen_individual(ruta, dimensiones):
    """Lee y prepocesa una imagen a partir de su ruta pero apto para usrse en producción.
    Parámetros:
        ruta: ruta en disco de la imagen.
        dimensiones: dimensiones deseadas para la imagen."""
    imagen = preprocesar_imagen(ruta, dimensiones)
    return tf.expand_dims(imagen, 0)  # Añadimos la dimension batch


utils = src.utilidades.Utilidades("test_x", "monet2photo", "configuracion_256.json")

imagen_muestra_pintor = preprocesar_imagen_individual(utils.obtener_archivo_muestra_pintor(),
                                                      (256, 256, 3)).numpy()
imagen_muestra_foto = preprocesar_imagen_individual(utils.obtener_archivo_muestra_foto(),
                                                    (256, 256, 3)).numpy()
print(imagen_muestra_pintor)
print(imagen_muestra_foto)
