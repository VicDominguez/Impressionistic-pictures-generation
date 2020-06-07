from glob import glob

import constantes as ctes
import tensorflow as tf


class LectorDataset:
    def __init__(self):
        self.AUTOTUNE = tf.data.experimental.AUTOTUNE

        directorio_datos = ctes.ruta_dataset
        ruta_pintor = directorio_datos + "\\trainA\\*"
        ruta_fotos = directorio_datos + "\\trainB\\*"

        self.tamanio_dataset_pintor = len(glob(ruta_pintor))
        self.tamanio_dataset_fotos = len(glob(ruta_fotos))

        # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
        self.dataset_procesado_a = self.preparar_dataset(
            tf.data.Dataset.list_files(ruta_pintor).map(
                self.leer_imagen, num_parallel_calls=self.AUTOTUNE))

        self.dataset_procesado_b = self.preparar_dataset(
            tf.data.Dataset.list_files(ruta_fotos).map(
                self.leer_imagen, num_parallel_calls=self.AUTOTUNE))

    @staticmethod
    def leer_imagen(ruta, test=False):
        # abrimos el archivo
        img = tf.io.read_file(ruta)
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)

        # si no es test espejamos aproximadamente la mitad de las mismas
        if not test:
            if tf.random.uniform(shape=[1])[0] > 0.5:
                tf.image.flip_left_right(img)

        # Use `convert_image_dtype` to convert to floats in the [0,1] range.
        img = tf.image.convert_image_dtype(img, tf.float32)
        # resize the image to the desired size.
        return tf.image.resize(img, ctes.dimensiones)

    def preparar_dataset(self, ds, shuffle_buffer_size=1000):

        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

        # Repeat forever
        ds = ds.repeat()

        ds = ds.batch(1)

        # # `prefetch` lets the dataset fetch batches in the background while the model
        # # is training.
        ds = ds.prefetch(self.AUTOTUNE)

        return ds

    def obtener_imagenes_entreno(self, batch_size=1):
        n_batches = int(min(self.tamanio_dataset_pintor, self.tamanio_dataset_pintor) / batch_size)
        for i in range(n_batches):
            yield next(iter(self.dataset_procesado_a)), next(iter(self.dataset_procesado_b))

    def obtener_imagenes_muestra(self):
        return self.leer_imagen(ruta=ctes.imagen_muestra_monet, test=True), \
               self.leer_imagen(ruta=ctes.imagen_muestra_real, test=True)

