import imageio
from glob import glob
import numpy as np
import skimage.transform

from constantes import Constantes


class LectorImagenes:
    __instance = None
    constantes = Constantes()

    def __new__(cls):
        if LectorImagenes.__instance is None:
            LectorImagenes.__instance = object.__new__(cls)
        return LectorImagenes.__instance

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        path_A = glob(self.constantes.ruta_dataset + '/%sA/*' % data_type)
        path_B = glob(self.constantes.ruta_dataset + '/%sB/*' % data_type)
        n_batches = int(min(len(path_A), len(path_B)) / batch_size)
        total_samples = n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)

        for i in range(n_batches - 1):
            batch_A = path_A[i * batch_size:(i + 1) * batch_size]
            batch_B = path_B[i * batch_size:(i + 1) * batch_size]

            imgs_A = np.empty([batch_size, self.constantes.ancho, self.constantes.alto, self.constantes.canales])
            imgs_B = np.empty([batch_size, self.constantes.ancho, self.constantes.alto, self.constantes.canales])

            for indice, (img_A, img_B) in enumerate(zip(batch_A, batch_B)):
                imgs_A[indice] = self.load_img(img_A, flip=True)
                imgs_B[indice] = self.load_img(img_B, flip=True)
            yield imgs_A, imgs_B

    def load_img(self, path, flip=False):
        img = imageio.imread(path, pilmode='RGB').astype(np.float)
        img = skimage.transform.resize(img, self.constantes.dimensiones)
        if flip and np.random.random() > 0.5:
            img = np.fliplr(img)
        img = img / 127.5 - 1.
        return img

    def load_single_img(self, path):
        img = self.load_img(path, flip=False)
        return img[np.newaxis, :, :, :]

