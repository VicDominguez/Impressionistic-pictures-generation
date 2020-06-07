import imageio
from glob import glob
import numpy as np
from skimage.transform import resize
import src.utils as cte


class DataLoader:
    
    def load_data(self, domain, batch_size=1, is_testing=False):
        data_type = "train%s" % domain if not is_testing else "test%s" % domain
        path = glob(cte.ruta_dataset + '/%s/*' % data_type)

        batch_images = np.random.choice(path, size=batch_size)

        imgs = []
        for img_path in batch_images:
            img = self.imread(img_path)
            if not is_testing:
                img = resize(img, cte.dimensiones)

                if np.random.random() > 0.5:
                    img = np.fliplr(img)
            else:
                img = resize(img, cte.dimensiones)
            imgs.append(img)

        imgs = np.array(imgs) / 127.5 - 1.

        return imgs

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        path_A = glob(cte.ruta_dataset + '/%sA/*' % data_type)
        path_B = glob(cte.ruta_dataset + '/%sB/*' % data_type)
        n_batches = int(min(len(path_A), len(path_B)) / batch_size)
        total_samples = n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)

        for i in range(n_batches - 1):
            batch_A = path_A[i * batch_size:(i + 1) * batch_size]
            batch_B = path_B[i * batch_size:(i + 1) * batch_size]
            imgs_A, imgs_B = [], []
            for img_A, img_B in zip(batch_A, batch_B):
                img_A = self.imread(img_A)
                img_B = self.imread(img_B)

                img_A = resize(img_A, cte.dimensiones)
                img_B = resize(img_B, cte.dimensiones)

                if not is_testing and np.random.random() > 0.5:
                    img_A = np.fliplr(img_A)
                    img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A) / 127.5 - 1.
            imgs_B = np.array(imgs_B) / 127.5 - 1.

            yield imgs_A, imgs_B

    def load_img(self, path):
        img = self.imread(path)
        img = imageio.imresize(img, cte.dimensiones)
        img = img / 127.5 - 1.
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        return imageio.imread(path, pilmode='RGB').astype(np.float)
