import tensorflow as tf

from src.parametros_modelo import alto, ancho, batch_size_test, batch_size_train, canales
from src.utilidades import *


@tf.function
def _load_img(ruta, flip=True):
    # abrimos el archivo
    img = tf.io.read_file(ruta)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=canales)

    # # si no es test espejamos aproximadamente la mitad de las mismas
    if flip and tf.random.uniform(shape=[1])[0] > 0.5:
        tf.image.flip_left_right(img)

    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, size=(ancho, alto))


def preparar_dataset(ds, cache=True, shuffle_buffer_size=1000, batch_size=1):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.repeat()  # repeat forever
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def load_single_img(path):
    img = _load_img(path, )
    return tf.expand_dims(img, 0)


class LectorImagenes(metaclass=Singleton):
    __slots__ = ["utils", "imagenes_train_pintor", "imagenes_train_real", "imagenes_test_pintor",
                 "imagenes_test_real", "dataset_train_pintor", "dataset_train_real", "dataset_test_pintor",
                 "dataset_test_real", "logger"]

    def __init__(self):
        self.utils = Utilidades()
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.imagenes_train_pintor = self.utils.obtener_rutas_imagenes_train_pintor()
        self.imagenes_train_real = self.utils.obtener_rutas_imagenes_train_real()
        self.imagenes_test_pintor = self.utils.obtener_rutas_imagenes_test_pintor()
        self.imagenes_test_real = self.utils.obtener_rutas_imagenes_test_real()

        self.logger = self.utils.obtener_logger("lector_imagenes")

        start = timestamp()
        listado_dataset_train_pintor = tf.data.Dataset.list_files(self.imagenes_train_pintor)
        self.logger.info("listado_dataset_train_pintor " + str(timestamp() - start))

        start = timestamp()
        listado_dataset_train_real = tf.data.Dataset.list_files(self.imagenes_train_real)
        self.logger.info("listado_dataset_train_real " + str(timestamp() - start))

        start = timestamp()
        listado_dataset_test_pintor = tf.data.Dataset.list_files(self.imagenes_test_pintor)
        self.logger.info("listado_dataset_test_pintor " + str(timestamp() - start))

        start = timestamp()
        listado_dataset_test_real = tf.data.Dataset.list_files(self.imagenes_test_real)
        self.logger.info("listado_dataset_test_real " + str(timestamp() - start))

        start = timestamp()
        listado_train_pintor_cargado = listado_dataset_train_pintor.map(_load_img, num_parallel_calls=AUTOTUNE)
        listado_train_real_cargado = listado_dataset_train_real.map(_load_img, num_parallel_calls=AUTOTUNE)
        listado_test_pintor_cargado = listado_dataset_test_pintor.map(_load_img, num_parallel_calls=AUTOTUNE)
        listado_test_real_cargado = listado_dataset_test_real.map(_load_img, num_parallel_calls=AUTOTUNE)
        self.logger.info("Mapeado datasets " + str(timestamp() - start))

        start = timestamp()
        self.dataset_train_pintor = preparar_dataset(listado_train_pintor_cargado, batch_size=batch_size_train,
                                                     cache=self.utils.obtener_archivo_cache("dataset_train_pintor"))
        self.dataset_train_real = preparar_dataset(listado_train_real_cargado, batch_size=batch_size_train,
                                                   cache=self.utils.obtener_archivo_cache("dataset_train_real"))
        self.dataset_test_pintor = preparar_dataset(listado_test_pintor_cargado, batch_size=batch_size_test,
                                                    cache=self.utils.obtener_archivo_cache("dataset_test_pintor"))
        self.dataset_test_real = preparar_dataset(listado_test_real_cargado, batch_size=batch_size_test,
                                                  cache=self.utils.obtener_archivo_cache("dataset_test_real"))
        self.logger.info("Preparación datasets " + str(timestamp() - start))

    def get_n_batches(self, batch_size=1, is_training=True, maximo_instancias=None):
        if is_training:
            imagenes_pintor = len(self.imagenes_train_pintor)
            imagenes_real = len(self.imagenes_train_real)
        else:
            imagenes_pintor = len(self.imagenes_test_pintor)
            imagenes_real = len(self.imagenes_test_real)

        if maximo_instancias is None:
            instancias = min(imagenes_pintor, imagenes_real)
        else:
            instancias = min(imagenes_pintor, imagenes_real, maximo_instancias)

        return int(instancias / batch_size)

    def load_batch(self, is_training=True, maximo_instancias=None):

        if is_training:
            dataset_pintor = self.dataset_train_pintor
            dataset_real = self.dataset_train_real
            batch_size = batch_size_train
        else:
            dataset_pintor = self.dataset_test_pintor
            dataset_real = self.dataset_test_real
            batch_size = batch_size_test

        n_batches = self.get_n_batches(batch_size, is_training, maximo_instancias)

        iter_dataset_pintor = iter(dataset_pintor)
        iter_dataset_real = iter(dataset_real)

        for i in range(n_batches):
            yield next(iter_dataset_pintor), next(iter_dataset_real)

    def obtener_imagen_muestra_pintor(self):
        return load_single_img(self.utils.obtener_archivo_muestra_pintor())

    def obtener_imagen_muestra_real(self):
        return load_single_img(self.utils.obtener_archivo_muestra_real())
