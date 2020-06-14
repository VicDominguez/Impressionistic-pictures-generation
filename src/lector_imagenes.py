import imageio
import numpy as np
import skimage.transform

from src.parametros_modelo import alto, ancho, canales
from src.utilidades import *


def _load_img(path, flip=False):
    img = imageio.imread(path, pilmode='RGB').astype(np.float)
    img = skimage.transform.resize(img, (ancho, alto))
    if flip and np.random.random() > 0.5:
        img = np.fliplr(img)
    img = img / 127.5 - 1.
    return img


def load_single_img(path):
    img = _load_img(path, flip=False)
    return img[np.newaxis, :, :, :]


class LectorImagenes:
    __slots__ = ["utils", "imagenes_train_pintor", "imagenes_train_real", "imagenes_test_pintor", "imagenes_test_real"]

    def __init__(self):
        self.utils = Utilidades()
        self.imagenes_train_pintor = self.utils.obtener_rutas_imagenes_train_pintor()
        self.imagenes_train_real = self.utils.obtener_rutas_imagenes_train_real()
        self.imagenes_test_pintor = self.utils.obtener_rutas_imagenes_test_pintor()
        self.imagenes_test_real = self.utils.obtener_rutas_imagenes_test_real()

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

    def load_batch(self, batch_size=1, is_training=True, maximo_instancias=None):
        n_batches = self.get_n_batches(batch_size, is_training, maximo_instancias)
        total_samples = n_batches * batch_size

        if is_training:
            imagenes_pintor = self.imagenes_train_pintor
            imagenes_real = self.imagenes_train_real
        else:
            imagenes_pintor = self.imagenes_test_pintor
            imagenes_real = self.imagenes_test_real

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        elecciones_pintor = np.random.choice(imagenes_pintor, total_samples, replace=False)
        elecciones_real = np.random.choice(imagenes_real, total_samples, replace=False)

        for i in range(n_batches - 1):
            batch_A = elecciones_pintor[i * batch_size:(i + 1) * batch_size]
            batch_B = elecciones_real[i * batch_size:(i + 1) * batch_size]

            imgs_A = np.empty([batch_size, ancho, alto, canales])
            imgs_B = np.empty([batch_size, ancho, alto, canales])

            for indice, (img_A, img_B) in enumerate(zip(batch_A, batch_B)):
                imgs_A[indice] = _load_img(img_A, flip=True)
                imgs_B[indice] = _load_img(img_B, flip=True)
            yield imgs_A, imgs_B

    def obtener_imagen_muestra_pintor(self):
        return load_single_img(self.utils.obtener_ruta_imagen_muestra_pintor())

    def obtener_imagen_muestra_real(self):
        return load_single_img(self.utils.obtener_ruta_imagen_muestra_real())
