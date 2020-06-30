import tensorflow as tf

from src.utilidades import *  # TODO quitar esto
from src.procesado_imagenes import leer_y_normalizar_imagen, preprocesar_imagen


# TODO documentacion

class CargadorImagenes(metaclass=Singleton):
    # TODO actualizar slots
    # __slots__ = ["utils", "AUTOTUNE", "imagenes_train_pintor", "imagenes_train_foto", "imagenes_test_pintor",
    #             "imagenes_test_foto", "dataset_train_pintor", "dataset_train_foto", "dataset_test_pintor",
    #             "dataset_test_foto", "logger"]

    def __init__(self, train=True):

        self.utils = Utilidades()
        if train:
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
            listado_train_pintor_cargado = listado_dataset_train_pintor.map(self._preprocesar_imagen_train,
                                                                            num_parallel_calls=AUTOTUNE)
            listado_train_foto_cargado = listado_dataset_train_foto.map(self._preprocesar_imagen_train,
                                                                        num_parallel_calls=AUTOTUNE)
            listado_test_pintor_cargado = listado_dataset_test_pintor.map(self._preprocesar_imagen_test,
                                                                          num_parallel_calls=AUTOTUNE)
            listado_test_foto_cargado = listado_dataset_test_foto.map(self._preprocesar_imagen_test,
                                                                      num_parallel_calls=AUTOTUNE)
            self.logger.info("Mapeado datasets " + str(timestamp() - start))

            start = timestamp()
            self.dataset_train_pintor = self.preparar_dataset(listado_train_pintor_cargado,
                                                              cache=self.utils.obtener_archivo_cache(
                                                                  "dataset_train_pintor"))
            self.dataset_train_foto = self.preparar_dataset(listado_train_foto_cargado,
                                                            cache=self.utils.obtener_archivo_cache(
                                                                "dataset_train_foto"))
            self.dataset_test_pintor = self.preparar_dataset(listado_test_pintor_cargado,
                                                             cache=self.utils.obtener_archivo_cache(
                                                                 "dataset_test_pintor"))
            self.dataset_test_foto = self.preparar_dataset(listado_test_foto_cargado,
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

        return int(instancias / self.utils.obtener_tamanio_batch())

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

        self.logger.info("Vamos a sacar batches nano")
        for i in range(n_batches):
            yield next(iter_dataset_pintor).numpy(), next(iter_dataset_foto).numpy()

    def obtener_imagen_muestra_pintor(self):
        return preprocesar_imagen(self.utils.obtener_archivo_muestra_pintor(), self.utils.obtener_anchura(),
                                  self.utils.obtener_altura(), self.utils.obtener_canales()).numpy()

    def obtener_imagen_muestra_foto(self):
        return preprocesar_imagen(self.utils.obtener_archivo_muestra_foto(), self.utils.obtener_anchura(),
                                  self.utils.obtener_altura(), self.utils.obtener_canales()).numpy()

    def preparar_dataset(self, ds, cache=True):  # Capamos a x valores?
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()
        ds = ds.shuffle(buffer_size=self.utils.obtener_tamanio_buffer())
        ds = ds.repeat()  # repeat forever
        ds = ds.batch(self.utils.obtener_tamanio_batch())
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds

    def _preprocesar_imagen_train(self, ruta):
        return leer_y_normalizar_imagen(ruta, self.utils.obtener_anchura(), self.utils.obtener_altura(),
                                        self.utils.obtener_canales(), train=True)

    def _preprocesar_imagen_test(self, ruta):
        return leer_y_normalizar_imagen(ruta, self.utils.obtener_anchura(), self.utils.obtener_altura(),
                                        self.utils.obtener_canales(), train=False)
