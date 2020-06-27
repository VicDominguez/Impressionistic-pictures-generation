import tensorflow as tf

from parametros_modelo import alto, ancho, canales, tamanio_batch, tamanio_buffer
from utilidades import *


def _leer_y_normalizar_imagen(ruta, train=False):
    """Primitiva que lee y prepocesa una imagen según si es para entrenar o no"""
    img = tf.io.read_file(ruta)  # Abrimos el archivo
    img = tf.image.decode_jpeg(img, channels=canales)  # Convertir el string a un tensor 3D uint8
    img = tf.image.resize(img, size=(ancho, alto), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    if train:  # si no es test espejamos aleatoriamente
        img = tf.image.random_flip_left_right(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


def preprocesar_imagen(ruta):
    """Lee y prepocesa una imagen a partir de su ruta"""
    img = _leer_y_normalizar_imagen(ruta, train=False)
    return tf.expand_dims(img, 0)


def _preprocesar_imagen_train(ruta):
    return _leer_y_normalizar_imagen(ruta, train=True)


def _preprocesar_imagen_test(ruta):
    return _leer_y_normalizar_imagen(ruta, train=False)


def preparar_dataset(ds, cache=True):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size=tamanio_buffer)
    ds = ds.repeat()  # repeat forever
    ds = ds.batch(tamanio_batch)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


class LectorImagenes(metaclass=Singleton):
    # TODO actualizar slots
    __slots__ = ["utils", "imagenes_train_pintor", "imagenes_train_foto", "imagenes_test_pintor",
                 "imagenes_test_foto", "dataset_train_pintor", "dataset_train_foto", "dataset_test_pintor",
                 "dataset_test_foto", "logger"]

    def __init__(self):

        self.utils = Utilidades()
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.imagenes_train_pintor = self.utils.obtener_rutas_imagenes_train_pintor()
        self.imagenes_train_foto = self.utils.obtener_rutas_imagenes_train_foto()
        self.imagenes_test_pintor = self.utils.obtener_rutas_imagenes_test_pintor()
        self.imagenes_test_foto = self.utils.obtener_rutas_imagenes_test_foto()

        self.logger = self.utils.obtener_logger("lector_imagenes")

        start = timestamp()
        listado_dataset_train_pintor = tf.data.Dataset.list_files(self.imagenes_train_pintor)
        self.logger.info("listado_dataset_train_pintor " + str(timestamp() - start))

        start = timestamp()
        listado_dataset_train_foto = tf.data.Dataset.list_files(self.imagenes_train_foto)
        self.logger.info("listado_dataset_train_foto " + str(timestamp() - start))

        start = timestamp()
        listado_dataset_test_pintor = tf.data.Dataset.list_files(self.imagenes_test_pintor)
        self.logger.info("listado_dataset_test_pintor " + str(timestamp() - start))

        start = timestamp()
        listado_dataset_test_foto = tf.data.Dataset.list_files(self.imagenes_test_foto)
        self.logger.info("listado_dataset_test_foto " + str(timestamp() - start))

        start = timestamp()
        listado_train_pintor_cargado = listado_dataset_train_pintor.map(_preprocesar_imagen_train,
                                                                        num_parallel_calls=AUTOTUNE)
        listado_train_foto_cargado = listado_dataset_train_foto.map(_preprocesar_imagen_train,
                                                                    num_parallel_calls=AUTOTUNE)
        listado_test_pintor_cargado = listado_dataset_test_pintor.map(_preprocesar_imagen_test,
                                                                      num_parallel_calls=AUTOTUNE)
        listado_test_foto_cargado = listado_dataset_test_foto.map(_preprocesar_imagen_test, num_parallel_calls=AUTOTUNE)
        self.logger.info("Mapeado datasets " + str(timestamp() - start))

        start = timestamp()
        self.dataset_train_pintor = preparar_dataset(listado_train_pintor_cargado,
                                                     cache=self.utils.obtener_archivo_cache("dataset_train_pintor"))
        self.dataset_train_foto = preparar_dataset(listado_train_foto_cargado,
                                                   cache=self.utils.obtener_archivo_cache("dataset_train_foto"))
        self.dataset_test_pintor = preparar_dataset(listado_test_pintor_cargado,
                                                    cache=self.utils.obtener_archivo_cache("dataset_test_pintor"))
        self.dataset_test_foto = preparar_dataset(listado_test_foto_cargado,
                                                  cache=self.utils.obtener_archivo_cache("dataset_test_foto"))
        self.logger.info("Preparación datasets " + str(timestamp() - start))

    def train_pintor(self):
        return self.dataset_train_pintor

    def train_foto(self):
        return self.dataset_train_foto

    def test_pintor(self):
        return self.dataset_test_pintor

    def test_foto(self):
        return self.dataset_test_foto

    def calcular_n_batches(self, is_training=True):
        if is_training:
            imagenes_pintor = len(self.imagenes_train_pintor)
            imagenes_foto = len(self.imagenes_train_foto)
        else:
            imagenes_pintor = len(self.imagenes_test_pintor)
            imagenes_foto = len(self.imagenes_test_foto)

        instancias = min(imagenes_pintor, imagenes_foto)

        return int(instancias / tamanio_batch)

    def cargar_batch(self, is_training=True):

        if is_training:
            dataset_pintor = self.dataset_train_pintor
            dataset_foto = self.dataset_train_foto
        else:
            dataset_pintor = self.dataset_test_pintor
            dataset_foto = self.dataset_test_foto

        n_batches = self.calcular_n_batches(is_training)

        iter_dataset_pintor = iter(dataset_pintor)
        iter_dataset_foto = iter(dataset_foto)

        for i in range(n_batches):
            yield next(iter_dataset_pintor).numpy(), next(iter_dataset_foto).numpy()

    def obtener_imagen_muestra_pintor(self):
        return preprocesar_imagen(self.utils.obtener_archivo_muestra_pintor()).numpy()

    def obtener_imagen_muestra_foto(self):
        return preprocesar_imagen(self.utils.obtener_archivo_muestra_foto()).numpy()
