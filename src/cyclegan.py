from __future__ import division, print_function

import io

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras.initializers import RandomNormal
from keras.layers import Activation, Concatenate, Dropout, Input, Layer, InputSpec
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.merge import add
from keras.models import Model
from keras.optimizers import Adam
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras import backend as K
from keras.utils import plot_model
import pickle as pkl

from src.parametros_modelo import *
from src.utilidades import *


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3]

    def call(self, y, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(y, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')


class CycleGAN:
    # TODO realizar slots
    # __slots__ = ['modelo_entrenado', 'disc_patch', 'discriminador_pintor', 'discriminador_foto', 'generador_foto',
    # 'generador_pintor', 'modelo','utils', 'logger']

    # TODO A es pintor, B es foto

    def __init__(self, tipo_generador):

        self.utils = Utilidades()
        self.logger = self.utils.obtener_logger("cyclegan")

        self.dimensiones_entrada = (ancho, alto, canales)
        self.tasa_aprendizaje = tasa_aprendizaje
        self.lambda_validacion = lambda_validacion
        self.lambda_reconstruccion = lambda_reconstruccion
        self.lambda_identidad = lambda_identidad
        self.tipo_generador = tipo_generador
        self.filtros_generador = filtros_generador
        self.filtros_discriminador = filtros_discriminador

        # Input shape
        self.alto = alto
        self.ancho = ancho
        self.canales = canales
        self.forma_imagen = (ancho, alto, canales)

        self.epoch = 0

        # Calculate salida shape of D (PatchGAN)
        patch = int(self.alto / 2 ** 3)
        self.disc_patch = (patch, patch, 1)

        self.inicializacion_pesos = RandomNormal(mean=0., stddev=0.02)

        self.optimizador = Adam(self.tasa_aprendizaje, 0.5)

        self.compilar_modelo()

        # self.ckpt = tf.train.Checkpoint(generador_foto=self.generador_foto,
        #                                 generador_pintor=self.generador_pintor,
        #                                 discriminador_pintor=self.discriminador_pintor,
        #                                 discriminador_foto=self.discriminador_foto)

        # self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, self.utils.obtener_ruta_checkpoints(), max_to_keep=5)
        #
        # if self.ckpt_manager.latest_checkpoint:
        #     self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        #     self.logger("Punto de control restaurado")

    def compilar_modelo(self):

        # Build and compile the discriminators
        self.discriminador_pintor = self.crear_discriminador()
        self.discriminador_foto = self.crear_discriminador()

        self.discriminador_pintor.compile(loss='mse',
                                          # optimizer=self.optimizador,
                                          optimizer=Adam(self.tasa_aprendizaje, 0.5),
                                          metrics=['accuracy'])
        self.discriminador_foto.compile(loss='mse',
                                        # optimizer=self.optimizador,
                                        optimizer=Adam(self.tasa_aprendizaje, 0.5),
                                        metrics=['accuracy'])

        # Build the generators
        if self.tipo_generador == 'unet':
            self.generador_foto = self.crear_generador_unet()
            self.generador_pintor = self.crear_generador_unet()
        else:
            self.generador_foto = self.crear_generador_resnet()
            self.generador_pintor = self.crear_generador_resnet()

        # For the combined model we will only train the generators
        self.discriminador_pintor.trainable = False
        self.discriminador_foto.trainable = False

        # Input images from both domains
        imagen_pintor = Input(shape=self.forma_imagen)
        imagen_foto = Input(shape=self.forma_imagen)

        # Translate images to the other domain
        falsificacion_foto = self.generador_foto(imagen_pintor)
        falsificacion_pintor = self.generador_pintor(imagen_foto)
        # Translate images back to original domain
        reconstruccion_pintor = self.generador_pintor(falsificacion_foto)
        reconstruccion_foto = self.generador_foto(falsificacion_pintor)
        # Identity mapping of images
        imagen_pintor_identidad = self.generador_pintor(imagen_pintor)
        imagen_foto_identidad = self.generador_foto(imagen_foto)

        # Discriminators determines validity of translated images
        validador_pintor = self.discriminador_pintor(falsificacion_pintor)
        validador_foto = self.discriminador_foto(falsificacion_foto)

        # Combined model trains generators to fool discriminators
        self.modelo_combinado = Model(inputs=[imagen_pintor, imagen_foto],
                                      outputs=[validador_pintor, validador_foto,
                                               reconstruccion_pintor, reconstruccion_foto,
                                               imagen_pintor_identidad, imagen_foto_identidad])
        self.modelo_combinado.compile(loss=['mse', 'mse',
                                            'mae', 'mae',
                                            'mae', 'mae'],
                                      loss_weights=[self.lambda_validacion, self.lambda_validacion,
                                                    self.lambda_reconstruccion, self.lambda_reconstruccion,
                                                    self.lambda_identidad, self.lambda_identidad],
                                      # optimizer=self.optimizador,
                                      optimizer=Adam(self.tasa_aprendizaje, 0.5))

        self.discriminador_pintor.trainable = True
        self.discriminador_foto.trainable = True

    def crear_generador_unet(self):

        def downsample(capa_entrada, filtros, tamanio_filtro=4):
            d = Conv2D(filtros, kernel_size=tamanio_filtro, strides=2, padding='same')(capa_entrada)
            d = InstanceNormalization(axis=-1, center=False, scale=False)(d)
            d = Activation('relu')(d)

            return d

        def upsample(capa_entrada, saltar_entrada, filtros, tamanio_filtro=4, ratio_dropout=0):
            u = UpSampling2D(size=2)(capa_entrada)
            u = Conv2D(filtros, kernel_size=tamanio_filtro, strides=1, padding='same')(u)
            u = InstanceNormalization(axis=-1, center=False, scale=False)(u)
            u = Activation('relu')(u)
            if ratio_dropout:
                u = Dropout(ratio_dropout)(u)

            u = Concatenate()([u, saltar_entrada])
            return u

        # Image input
        img = Input(shape=self.forma_imagen)

        # Downsampling
        d1 = downsample(img, self.filtros_generador)
        d2 = downsample(d1, self.filtros_generador * 2)
        d3 = downsample(d2, self.filtros_generador * 4)
        d4 = downsample(d3, self.filtros_generador * 8)

        # Upsampling
        u1 = upsample(d4, d3, self.filtros_generador * 4)
        u2 = upsample(u1, d2, self.filtros_generador * 2)
        u3 = upsample(u2, d1, self.filtros_generador)

        u4 = UpSampling2D(size=2)(u3)
        salida_img = Conv2D(self.canales, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(img, salida_img)

    def crear_generador_resnet(self):

        def conv7s1(capa_entrada, filtros, final):
            y = ReflectionPadding2D(padding=(3, 3))(capa_entrada)
            y = Conv2D(filtros, kernel_size=(7, 7), strides=1, padding='valid',
                       kernel_initializer=self.inicializacion_pesos)(y)
            if final:
                y = Activation('tanh')(y)
            else:
                y = InstanceNormalization(axis=-1, center=False, scale=False)(y)
                y = Activation('relu')(y)
            return y

        def downsample(capa_entrada, filtros):
            y = Conv2D(filtros, kernel_size=(3, 3), strides=2, padding='same',
                       kernel_initializer=self.inicializacion_pesos)(
                capa_entrada)
            y = InstanceNormalization(axis=-1, center=False, scale=False)(y)
            y = Activation('relu')(y)
            return y

        def residual(capa_entrada, filtros):
            atajo = capa_entrada
            y = ReflectionPadding2D(padding=(1, 1))(capa_entrada)
            y = Conv2D(filtros, kernel_size=(3, 3), strides=1, padding='valid',
                       kernel_initializer=self.inicializacion_pesos)(y)
            y = InstanceNormalization(axis=-1, center=False, scale=False)(y)
            y = Activation('relu')(y)

            y = ReflectionPadding2D(padding=(1, 1))(y)
            y = Conv2D(filtros, kernel_size=(3, 3), strides=1, padding='valid',
                       kernel_initializer=self.inicializacion_pesos)(y)
            y = InstanceNormalization(axis=-1, center=False, scale=False)(y)

            return add([atajo, y])

        def upsample(capa_entrada, filtros):
            y = Conv2DTranspose(filtros, kernel_size=(3, 3), strides=2, padding='same',
                                kernel_initializer=self.inicializacion_pesos)(capa_entrada)
            y = InstanceNormalization(axis=-1, center=False, scale=False)(y)
            y = Activation('relu')(y)

            return y

        # Image input
        img = Input(shape=self.forma_imagen)

        x = img

        x = conv7s1(x, self.filtros_generador, False)
        x = downsample(x, self.filtros_generador * 2)
        x = downsample(x, self.filtros_generador * 4)
        x = residual(x, self.filtros_generador * 4)
        x = residual(x, self.filtros_generador * 4)
        x = residual(x, self.filtros_generador * 4)
        x = residual(x, self.filtros_generador * 4)
        x = residual(x, self.filtros_generador * 4)
        x = residual(x, self.filtros_generador * 4)
        x = residual(x, self.filtros_generador * 4)
        x = residual(x, self.filtros_generador * 4)
        x = residual(x, self.filtros_generador * 4)
        x = upsample(x, self.filtros_generador * 2)
        x = upsample(x, self.filtros_generador)
        x = conv7s1(x, 3, True)
        salida = x

        return Model(img, salida)

    def crear_discriminador(self):

        def conv4(capa_entrada, filtros, stride=2, norm=True):
            y = Conv2D(filtros, kernel_size=(4, 4), strides=stride, padding='same',
                       kernel_initializer=self.inicializacion_pesos)(capa_entrada)

            if norm:
                y = InstanceNormalization(axis=-1, center=False, scale=False)(y)

            y = LeakyReLU(0.2)(y)

            return y

        img = Input(shape=self.forma_imagen)

        x = conv4(img, self.filtros_discriminador, stride=2, norm=False)
        x = conv4(x, self.filtros_discriminador * 2, stride=2)
        x = conv4(x, self.filtros_discriminador * 4, stride=2)
        x = conv4(x, self.filtros_discriminador * 8, stride=1)

        salida = Conv2D(1, kernel_size=4, strides=1, padding='same', kernel_initializer=self.inicializacion_pesos)(x)

        return Model(img, salida)

    def train(self, lector_imagenes):

        writer_train = tf.summary.create_file_writer(self.utils.obtener_ruta_logs_train())

        imagen_muestra_pintor = lector_imagenes.obtener_imagen_muestra_pintor()
        imagen_muestra_foto = lector_imagenes.obtener_imagen_muestra_foto()

        # Adversarial loss ground truths
        umbral_verdad = np.ones((1,) + self.disc_patch)
        umbral_falso = np.zeros((1,) + self.disc_patch)

        numero_batches = lector_imagenes.calcular_n_batches()

        error_discriminadores = np.ones(numero_batches)
        precision_discriminadores = np.ones(numero_batches)
        error_generadores = np.ones(numero_batches)
        error_validez_generadores = np.ones(numero_batches)
        error_reconstruccion_generadores = np.ones(numero_batches)
        error_identidad_generadores = np.ones(numero_batches)

        comienzo_entrenamiento = timestamp()

        for epoch in range(self.epoch, epochs):
            if gsutil_disponible():
                self.utils.copiar_logs_gcp()

            comienzo_epoch = timestamp()
            self.logger.info("epoch " + str(epoch))

            for indice, (imagenes_pintor, imagenes_foto) in enumerate(lector_imagenes.cargar_batch()):
                error_discriminadores_actual = self.entrenar_discriminadores(imagenes_pintor, imagenes_foto,
                                                                             umbral_verdad, umbral_falso)
                error_generadores_actual = self.entrenar_generadores(imagenes_pintor, imagenes_foto, umbral_verdad)

                error_discriminadores[indice] = error_discriminadores_actual[0]
                precision_discriminadores[indice] = 100 * error_discriminadores_actual[1]
                error_generadores[indice] = error_generadores_actual[0]
                error_validez_generadores[indice] = np.mean(error_generadores_actual[1:3])
                error_reconstruccion_generadores[indice] = np.mean(error_generadores_actual[3:5])
                error_identidad_generadores[indice] = np.mean(error_generadores_actual[5:6])

            # TODO test
            self.escribir_metricas_perdidas(error_discriminadores, precision_discriminadores, error_generadores,
                                            error_validez_generadores, error_reconstruccion_generadores,
                                            error_identidad_generadores, writer_train, epoch)
            self._imagen_muestra(imagen_muestra_pintor, imagen_muestra_foto, epoch)
            self.guardar_progreso(epoch)
            fin_epoch = timestamp()

            self.logger.info("epoch " + str(epoch) + " completado en " + str(fin_epoch - comienzo_epoch))

            self.epoch += 1

            # if (epoch + 1) % 5 == 0:
            #     ckpt_save_path = self.ckpt_manager.save()
            #     self.logger.info('Guardando checkpoint para el epoch {} en {}'.format(epoch + 1, ckpt_save_path))

        fin_entrenamiento = timestamp()
        self.logger.info("Entrenamiento completado en " + str(fin_entrenamiento - comienzo_entrenamiento))
        self._guardar_modelo("prueba")

    # @tf.function
    def entrenar_discriminadores(self, imagenes_pintor, imagenes_foto, umbral_verdad, umbral_falso):

        # Translate images to opposite domain
        falsificacion_foto = self.generador_foto.predict(imagenes_pintor)
        falsificacion_pintor = self.generador_pintor.predict(imagenes_foto)

        # Train the discriminators (original images = real / translated = Fake)
        error_discriminador_pintor_real = self.discriminador_pintor.train_on_batch(imagenes_pintor, umbral_verdad)
        error_discriminador_pintor_falsificacion = self.discriminador_pintor.train_on_batch(
            falsificacion_pintor, umbral_falso)

        error_discriminador_pintor = 0.5 * np.add(error_discriminador_pintor_real,
                                                  error_discriminador_pintor_falsificacion)

        error_discriminador_foto_real = self.discriminador_foto.train_on_batch(imagenes_foto, umbral_verdad)
        error_discriminador_foto_falsificacion = self.discriminador_foto.train_on_batch(falsificacion_foto,
                                                                                        umbral_falso)

        error_discriminador_foto = 0.5 * np.add(error_discriminador_foto_real,
                                                error_discriminador_foto_falsificacion)

        # Total disciminator loss
        return 0.5 * np.add(error_discriminador_pintor, error_discriminador_foto)

    # @tf.function
    def entrenar_generadores(self, imagenes_pintor, imagenes_foto, umbral_verdad):
        return self.modelo_combinado.train_on_batch([imagenes_pintor, imagenes_foto],
                                                    [umbral_verdad, umbral_verdad,
                                                     imagenes_pintor, imagenes_foto,
                                                     imagenes_pintor, imagenes_foto])

    def _imagen_muestra(self, imagen_pintor, imagen_foto, epoch):
        file_writer = tf.summary.create_file_writer(self.utils.obtener_ruta_logs())

        filas, columnas = 2, 3

        estimacion_foto = self.generador_foto.predict(imagen_pintor)
        estimacion_pintor = self.generador_pintor.predict(imagen_foto)
        # Translate back to original domain
        pintor_reconstruido = self.generador_pintor.predict(estimacion_foto)
        foto_reconstruida = self.generador_foto.predict(estimacion_pintor)

        gen_imgs = np.concatenate(
            [imagen_pintor, estimacion_foto, pintor_reconstruido, imagen_foto, estimacion_pintor, foto_reconstruida])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Traducida', 'Reconstruida']
        fig, axs = plt.subplots(filas, columnas)
        cnt = 0
        for i in range(filas):
            for j in range(columnas):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i, j].axis('off')
                cnt += 1

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=400)
        buf.seek(0)
        plt.close()

        imagen_tf = tf.image.decode_png(buf.getvalue(), channels=4)
        imagen_tf = tf.expand_dims(imagen_tf, 0)

        with file_writer.as_default():
            tf.summary.image("Imagen resumen", imagen_tf, step=epoch)

        return fig

    def _guardar_modelo(self, nombre):
        self.modelo_combinado.save(self.utils.obtener_archivo_modelo_a_guardar(nombre))

    @staticmethod
    def escribir_metricas_perdidas(error_discriminadores, precision_discriminadores, error_generadores,
                                   error_validez_generadores, error_reconstruccion_generadores,
                                   error_identidad_generadores, writer, step):

        with writer.as_default():
            tf.summary.scalar("Media del error de los discriminadores", np.mean(error_discriminadores), step=step)
            tf.summary.scalar("Media de la precision de los discriminadores", np.mean(precision_discriminadores),
                              step=step)
            tf.summary.scalar("Media del error de los generadores", np.mean(error_generadores), step=step)
            tf.summary.scalar("Media del error de validez de los generadores", np.mean(error_validez_generadores),
                              step=step)
            tf.summary.scalar("Media del error de reconstruccion de los generadores",
                              np.mean(error_reconstruccion_generadores), step=step)
            tf.summary.scalar("Media del error de identidad de los generadores", np.mean(error_identidad_generadores),
                              step=step)
            tf.summary.scalar("Desviación estándar del error de los discriminadores", np.std(error_discriminadores),
                              step=step)
            tf.summary.scalar("Desviación estándar de la precision de los discriminadores",
                              np.std(precision_discriminadores), step=step)
            tf.summary.scalar("Desviación estándar del error de los generadores", np.std(error_generadores), step=step)
            tf.summary.scalar("Desviación estándar del error de validez de los generadores",
                              np.std(error_validez_generadores), step=step)
            tf.summary.scalar("Desviación estándar del error de reconstruccion de los generadores",
                              np.std(error_reconstruccion_generadores), step=step)
            tf.summary.scalar("Desviación estándar del error de identidad de los generadores",
                              np.std(error_identidad_generadores), step=step)

    def pintar_modelo(self):
        plot_model(self.modelo_combinado, to_file=self.utils.obtener_ruta_archivo_modelo_esquema(),
                   show_shapes=True, show_layer_names=True)
        plot_model(self.discriminador_pintor,
                   to_file=self.utils.obtener_ruta_archivo_discriminador_pintor_esquema(),
                   show_shapes=True, show_layer_names=True)
        plot_model(self.discriminador_foto, to_file=self.utils.obtener_ruta_archivo_discriminador_foto_esquema(),
                   show_shapes=True, show_layer_names=True)
        plot_model(self.generador_pintor, to_file=self.utils.obtener_ruta_archivo_generador_pintor_esquema(),
                   show_shapes=True, show_layer_names=True)
        plot_model(self.generador_foto, to_file=self.utils.obtener_ruta_archivo_generador_foto_esquema(),
                   show_shapes=True, show_layer_names=True)

    def serializar_red(self):  # TODO poner en lanzadera entreno

        with open(self.utils.obtener_ruta_archivo_modelo_parametros(), 'wb') as archivo:
            pkl.dump([
                tasa_aprendizaje,
                lambda_reconstruccion,
                lambda_validacion,
                lambda_identidad,
                ancho,
                alto,
                canales,
                epochs,
                tamanio_buffer,
                tamanio_batch,
                filtros_generador,
                filtros_discriminador], archivo)

        self.pintar_modelo()

    def guardar_modelo(self):
        self.modelo_combinado.save(self.utils.obtener_ruta_fichero_modelo())
        self.discriminador_pintor.save(self.utils.obtener_ruta_fichero_discriminador_pintor())
        self.discriminador_foto.save(self.utils.obtener_ruta_fichero_discriminador_foto())
        self.generador_pintor.save(self.utils.obtener_ruta_fichero_generador_pintor())
        self.generador_foto.save(self.utils.obtener_ruta_fichero_generador_foto())

        with open(self.utils.obtener_ruta_archivo_modelo_objeto(), "wb") as archivo:
            pkl.dump(self, archivo)

    def cargar_pesos(self):
        self.modelo_combinado.load_weights(self.utils.obtener_ruta_fichero_pesos_modelo())

    def guardar_progreso(self, epoch):
        self.modelo_combinado.save_weights(self.utils.obtener_ruta_fichero_pesos_modelo_epoch(epoch))
        self.modelo_combinado.save_weights(self.utils.obtener_ruta_fichero_pesos_modelo())
        self.guardar_modelo()
