import tensorflow as tf
import os
import src.constantes as ctes
import matplotlib.pyplot  as plt


def show_batch(image_batch):
    plt.figure(figsize=(10, 10))
    # for n in range(25):
    #     ax = plt.subplot(5,5,n+1)
    #     plt.imshow(image_batch[n])
    #     plt.axis('off')

    plt.imshow(image_batch[0])
    plt.show()


AUTOTUNE = tf.data.experimental.AUTOTUNE


class LectorDatos:
    def __init__(self, directorio_datos):
        self.directorio_datos = ctes.ruta_dataset

    def listar(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU

        # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
        dataset_a = preparar_dataset(tf.data.Dataset.list_files(str(self.directorio_datos + "\\trainA\\*")).map(leer_imagen, num_parallel_calls=AUTOTUNE))
        dataset_b = preparar_dataset(tf.data.Dataset.list_files(str(self.directorio_datos + "\\trainB\\*")).map(leer_imagen, num_parallel_calls=AUTOTUNE))

        for a, b in zip(dataset_a, dataset_b):
            # image_batch = next(iter(train_ds))

            show_batch(a)
            show_batch(b)


def leer_imagen(ruta):
    return decodificar_imagen(tf.io.read_file(ruta))


def decodificar_imagen(img):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # Use `convert_image_dtype` to convert to floats in the [0,1] range.
    img = tf.image.convert_image_dtype(img, tf.float32)
    # resize the image to the desired size.
    return tf.image.resize(img, ctes.dimensiones)


def preparar_dataset(ds, cache=True, shuffle_buffer_size=1000):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()

    ds = ds.shuffle(buffer_size=shuffle_buffer_size)

    # Repeat forever
    ds = ds.repeat()

    ds = ds.batch(1)

    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=1072)

    return ds


if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')
    ld = LectorDatos(ctes.ruta_dataset)
    ld.listar()

# def process_path(file_path):
#   label = get_label(file_path)
#   # load the raw data from the file as a string
#   img = tf.io.read_file(file_path)
#   img = decode_img(img)
#   return img, label
#
# # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
# labeled_ds = list_ds.map(process_path, num_parallel_calls=AUTOTUNE)
#
# for image, label in labeled_ds.take(1):
#   print("Image shape: ", image.numpy().shape)
#   print("Label: ", label.numpy())
#
# def prepare_for_training(ds, cache=True, shuffle_buffer_size=1000):
#   # This is a small dataset, only load it once, and keep it in memory.
#   # use `.cache(filename)` to cache preprocessing work for datasets that don't
#   # fit in memory.
#   if cache:
#     if isinstance(cache, str):
#       ds = ds.cache(cache)
#     else:
#       ds = ds.cache()
#
#   ds = ds.shuffle(buffer_size=shuffle_buffer_size)
#
#   # Repeat forever
#   ds = ds.repeat()
#
#   ds = ds.batch(BATCH_SIZE)
#
#   # `prefetch` lets the dataset fetch batches in the background while the model
#   # is training.
#   ds = ds.prefetch(buffer_size=AUTOTUNE)
#
#   return ds
