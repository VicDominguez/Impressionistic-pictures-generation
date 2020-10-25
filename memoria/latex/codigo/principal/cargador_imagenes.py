import tensorflow as tf

from utilidades import *
import procesado_imagenes


class CargadorImagenes(metaclass=Singleton):
    """Clase que sirve para alimentar de imágenes a la red cuando entrena"""

    __slots__ = ["tamanio_batch", "tamanio_buffer", "dimensiones",
                 "logger", "numero_imagenes_entreno_pintor",
                 "numero_imagenes_entreno_foto", "numero_imagenes_test_pintor",
                 "numero_imagenes_test_foto", "dataset_entreno_pintor",
                 "dataset_entreno_foto", "dataset_test_pintor",
                 "dataset_test_foto"]

    def __init__(self):

        utils = Utilidades()

        self.tamanio_batch = utils.obtener_tamanio_batch()
        self.tamanio_buffer = utils.obtener_tamanio_buffer()
        self.dimensiones = utils.obtener_dimensiones()

        rutas_imagenes_entreno_pintor = \
            utils.obtener_rutas_imagenes_entreno_pintor()
        rutas_imagenes_entreno_foto = \
            utils.obtener_rutas_imagenes_entreno_foto()
        rutas_imagenes_test_pintor = utils.obtener_rutas_imagenes_test_pintor()
        rutas_imagenes_test_foto = utils.obtener_rutas_imagenes_test_foto()

        # Atributos para calcular n_batches de forma eficiente
        self.numero_imagenes_entreno_pintor = len(rutas_imagenes_entreno_pintor)
        self.numero_imagenes_entreno_foto = len(rutas_imagenes_entreno_foto)
        self.numero_imagenes_test_pintor = len(rutas_imagenes_test_pintor)
        self.numero_imagenes_test_foto = len(rutas_imagenes_test_foto)

        self.logger = utils.obtener_logger("lector_imagenes")

        inicio = timestamp()
        listado_dataset_entreno_pintor = \
            tf.data.Dataset.list_files(rutas_imagenes_entreno_pintor)
        self.logger.info("listado_dataset_entreno_pintor " +
                         str(timestamp() - inicio))

        inicio = timestamp()
        listado_dataset_entreno_foto = \
            tf.data.Dataset.list_files(rutas_imagenes_entreno_foto)
        self.logger.info("listado_dataset_entreno_foto " +
                         str(timestamp() - inicio))

        inicio = timestamp()
        listado_dataset_test_pintor = \
            tf.data.Dataset.list_files(rutas_imagenes_test_pintor)
        self.logger.info("listado_dataset_test_pintor " +
                         str(timestamp() - inicio))

        inicio = timestamp()
        listado_dataset_test_foto = \
            tf.data.Dataset.list_files(rutas_imagenes_test_foto)
        self.logger.info("listado_dataset_test_foto "
                         + str(timestamp() - inicio))

        AUTOTUNE = tf.data.experimental.AUTOTUNE

        inicio = timestamp()
        imagenes_entreno_pintor = listado_dataset_entreno_pintor.map(
            self._preprocesar_imagen_dataset, num_parallel_calls=AUTOTUNE)
        imagenes_entreno_foto = listado_dataset_entreno_foto.map(
            self._preprocesar_imagen_dataset, num_parallel_calls=AUTOTUNE)
        imagenes_test_pintor = listado_dataset_test_pintor.map(
            self._preprocesar_imagen_dataset, num_parallel_calls=AUTOTUNE)
        imagenes_test_foto = listado_dataset_test_foto.map(
            self._preprocesar_imagen_dataset, num_parallel_calls=AUTOTUNE)
        self.logger.info("Mapeado datasets " + str(timestamp() - inicio))

        inicio = timestamp()
        self.dataset_entreno_pintor = self.preparar_dataset(
            imagenes_entreno_pintor, cache=True,
            ruta_cache=utils.obtener_archivo_cache("dataset_entreno_pintor"))
        self.dataset_entreno_foto = self.preparar_dataset(
            imagenes_entreno_foto, cache=True,
            ruta_cache=utils.obtener_archivo_cache("dataset_entreno_foto"))
        self.dataset_test_pintor = self.preparar_dataset(
            imagenes_test_pintor, cache=True,
            ruta_cache=utils.obtener_archivo_cache("dataset_test_pintor"))
        self.dataset_test_foto = self.preparar_dataset(
            imagenes_test_foto, cache=True,
            ruta_cache=utils.obtener_archivo_cache("dataset_test_foto"))
        self.logger.info("Preparación datasets " + str(timestamp() - inicio))

    def calcular_n_batches(self, entreno=True):
        """Calcula los n_batches del proceso a realizar.
        
        Parámetros:
            entreno: indicación si de el proceso es entrenamiento o no."""

        if entreno:
            imagenes_pintor = self.numero_imagenes_entreno_pintor
            imagenes_foto = self.numero_imagenes_entreno_foto
        else:
            imagenes_pintor = self.numero_imagenes_test_pintor
            imagenes_foto = self.numero_imagenes_test_foto

        return int(min(imagenes_pintor, imagenes_foto) / self.tamanio_batch)

    def cargar_batch(self, entreno=True):
        """Generador que devuelve imágenes en formato numpy
        para realizar un proceso batch.

            Parámetros:
                entreno: indicación de si el proceso es entrenamiento o no."""

        if entreno:
            dataset_pintor = self.dataset_entreno_pintor
            dataset_foto = self.dataset_entreno_foto
        else:
            dataset_pintor = self.dataset_test_pintor
            dataset_foto = self.dataset_test_foto

        n_batches = self.calcular_n_batches(entreno)

        iter_dataset_pintor = iter(dataset_pintor)
        iter_dataset_foto = iter(dataset_foto)

        for i in range(n_batches):
            yield next(iter_dataset_pintor).numpy(), \
                  next(iter_dataset_foto).numpy()

    def preparar_dataset(self, dataset, cache, ruta_cache=None):
        """Prepara el dataset para ser consumido.
        
        Parámetros:
            dataset: el dataset a preparar (tf.Data.Dataset)
            
            cache: si queremos usar o no caché.
            
            ruta_cache: si hemos indicado que queremos usar caché,
             parámetro indica la ruta a guardar
            la caché si no encaja en memoria."""
        AUTOTUNE = tf.data.experimental.AUTOTUNE
        if cache:
            # verdadero si el parámetro es un string, un número distinto de 0..
            if isinstance(ruta_cache, str):
                dataset = dataset.cache(ruta_cache)
            else:
                dataset = dataset.cache()
        dataset = dataset.shuffle(buffer_size=self.tamanio_buffer)
        dataset = dataset.repeat()  # repetimos continuamente
        dataset = dataset.batch(self.tamanio_batch)
        dataset = dataset.prefetch(buffer_size=AUTOTUNE)
        return dataset

    def _preprocesar_imagen_dataset(self, ruta):
        """Recubridor para llamar a la función procesar
        imagen desde la función map.

        Parámetros:
            ruta: ruta en disco de la imagen"""
        return procesado_imagenes.preprocesar_imagen(ruta, self.dimensiones)
