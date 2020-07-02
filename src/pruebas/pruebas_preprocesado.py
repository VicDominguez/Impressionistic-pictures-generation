from PIL import Image
import numpy as np
import tensorflow as tf

nombres_guardar = ["area", "bicubic", "bilinear", "gaussian", "lanczos3", "lanczos5", "mitchell", "nearest"]
metodos = [tf.image.ResizeMethod.AREA, tf.image.ResizeMethod.BICUBIC, tf.image.ResizeMethod.BILINEAR,
           tf.image.ResizeMethod.GAUSSIAN, tf.image.ResizeMethod.LANCZOS3, tf.image.ResizeMethod.LANCZOS5,
           tf.image.ResizeMethod.MITCHELLCUBIC, tf.image.ResizeMethod.NEAREST_NEIGHBOR]
for nombre,metodo in zip(nombres_guardar,metodos):
    b = tf.io.read_file("1.jpg")  # Abrimos el archivo
    # Convertir el string a un tensor 3D uint8
    b = tf.image.decode_jpeg(b, channels=3)
    # Tenemos que cortar la b. Desgraciadamente no se puede mantener el ratio del aspecto
    b = tf.image.resize(b, size=(256, 256), method=metodo).numpy().astype(np.uint8)
    Image.fromarray(b, "RGB").save(nombre + ".jpeg")
