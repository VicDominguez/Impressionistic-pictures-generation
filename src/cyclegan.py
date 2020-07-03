from __future__ import division, print_function

import pickle as pkl

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.initializers import RandomNormal
from keras.layers import Activation, Concatenate, Dropout, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers.merge import add
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import numpy as np
import io

from ampliar_imagen import ampliar
from procesado_imagenes import preprocesar_imagen_individual
from utilidades import *
from capas_extra import ReflectionPadding2D


class CycleGAN:
    __slots__ = ['utils', 'logger', "tipo_generador", "tasa_aprendizaje", "lambda_validacion", "lambda_reconstruccion",
                 "lambda_identidad", "filtros_generador", "filtros_discriminador", "dimensiones", "epoch_actual",
                 "disc_patch", "inicializacion_pesos", "discriminador_pintor", "discriminador_foto", "generador_foto",
                 "generador_pintor", "modelo_combinado"]

    def __init__(self, restaurar=False, tipo_generador="resnet"):

        def crear_generador_unet():

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
            img = Input(shape=self.dimensiones)

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

        def crear_generador_resnet():

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
            img = Input(shape=self.dimensiones)

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

        def crear_discriminador():

            def conv4(capa_entrada, filtros, stride=2, norm=True):
                y = Conv2D(filtros, kernel_size=(4, 4), strides=stride, padding='same',
                           kernel_initializer=self.inicializacion_pesos)(capa_entrada)

                if norm:
                    y = InstanceNormalization(axis=-1, center=False, scale=False)(y)

                y = LeakyReLU(0.2)(y)

                return y

            img = Input(shape=self.dimensiones)

            x = conv4(img, self.filtros_discriminador, stride=2, norm=False)
            x = conv4(x, self.filtros_discriminador * 2, stride=2)
            x = conv4(x, self.filtros_discriminador * 4, stride=2)
            x = conv4(x, self.filtros_discriminador * 8, stride=1)

            salida = Conv2D(1, kernel_size=4, strides=1, padding='same', kernel_initializer=self.inicializacion_pesos)(
                x)

            return Model(img, salida)

        self.utils = Utilidades()
        self.logger = self.utils.obtener_logger("cyclegan")
        self.dimensiones = self.utils.obtener_dimensiones()

        if restaurar and self.utils.existen_ficheros_modelo():
            self.logger.info("Cargando modelo")
            self.discriminador_pintor = load_model(self.utils.obtener_ruta_fichero_discriminador_pintor(),
                                                   custom_objects={'InstanceNormalization': InstanceNormalization})
            self.discriminador_foto = load_model(self.utils.obtener_ruta_fichero_discriminador_foto(),
                                                 custom_objects={'InstanceNormalization': InstanceNormalization})
            self.generador_pintor = load_model(self.utils.obtener_ruta_fichero_generador_pintor(),
                                               custom_objects={'ReflectionPadding2D': ReflectionPadding2D,
                                                               'InstanceNormalization': InstanceNormalization})
            self.generador_foto = load_model(self.utils.obtener_ruta_fichero_generador_foto(),
                                             custom_objects={'ReflectionPadding2D': ReflectionPadding2D,
                                                             'InstanceNormalization': InstanceNormalization})
            self.modelo_combinado = load_model(self.utils.obtener_ruta_fichero_modelo(),
                                               custom_objects={'ReflectionPadding2D': ReflectionPadding2D,
                                                               'InstanceNormalization': InstanceNormalization})
            self.logger.info("Modelo cargado")
        else:
            self.logger.info("Creando modelo")
            self.tipo_generador = tipo_generador

            # obtenemos parámetros del modelo
            self.tasa_aprendizaje = self.utils.obtener_tasa_aprendizaje()
            self.lambda_validacion = self.utils.obtener_lambda_validacion()
            self.lambda_reconstruccion = self.utils.obtener_lambda_reconstruccion()
            self.lambda_identidad = self.utils.obtener_lambda_identidad()
            self.filtros_generador = self.utils.obtener_filtros_generador()
            self.filtros_discriminador = self.utils.obtener_filtros_discriminador()
            self.epoch_actual = 0

            # Calculate salida shape of D (PatchGAN) #TODO revisar esto
            patch = int(self.dimensiones[1] / 2 ** 3)
            self.disc_patch = (patch, patch, 1)

            self.inicializacion_pesos = RandomNormal(mean=0., stddev=0.02)

            # Creamos los discriminadores
            self.discriminador_pintor = crear_discriminador()
            self.discriminador_foto = crear_discriminador()

            self.discriminador_pintor.compile(loss='mse',
                                              optimizer=Adam(self.tasa_aprendizaje, 0.5),
                                              metrics=['accuracy'])
            self.discriminador_foto.compile(loss='mse',
                                            optimizer=Adam(self.tasa_aprendizaje, 0.5),
                                            metrics=['accuracy'])

            # Creamos los generadores
            if self.tipo_generador == 'unet':
                self.generador_foto = crear_generador_unet()
                self.generador_pintor = crear_generador_unet()
            else:
                self.generador_foto = crear_generador_resnet()
                self.generador_pintor = crear_generador_resnet()

            # Para el modelo combinado solo entrenamos los generadores
            self.discriminador_pintor.trainable = False
            self.discriminador_foto.trainable = False

            self.modelo_combinado = self._crear_modelo()

            self.modelo_combinado.compile(loss=['mse', 'mse',
                                                'mae', 'mae',
                                                'mae', 'mae'],
                                          loss_weights=[self.lambda_validacion, self.lambda_validacion,
                                                        self.lambda_reconstruccion, self.lambda_reconstruccion,
                                                        self.lambda_identidad, self.lambda_identidad],
                                          optimizer=Adam(self.tasa_aprendizaje, 0.5))

            self.discriminador_pintor.trainable = True
            self.discriminador_foto.trainable = True
            # cargamos los ultimos pesos, si hay
            self.cargar_ultimos_pesos()

    def train(self, lector_imagenes):

        sumario_entreno = tf.summary.create_file_writer(self.utils.obtener_ruta_logs_entreno())
        sumario_test = tf.summary.create_file_writer(self.utils.obtener_ruta_logs_test())

        imagen_muestra_pintor = preprocesar_imagen_individual(self.utils.obtener_archivo_muestra_pintor(),
                                                              self.dimensiones).numpy()
        imagen_muestra_foto = preprocesar_imagen_individual(self.utils.obtener_archivo_muestra_foto(),
                                                            self.dimensiones).numpy()

        # Umbrales para los errores adversativos
        umbral_verdad = np.ones((1,) + self.disc_patch)
        umbral_falso = np.zeros((1,) + self.disc_patch)

        numero_batches_entreno = lector_imagenes.calcular_n_batches(entreno=True)
        numero_batches_test = lector_imagenes.calcular_n_batches(entreno=False)

        # Inicializamos los arrays que contendrán los errores
        error_discriminadores_entreno = np.ones(numero_batches_entreno)
        precision_discriminadores_entreno = np.ones(numero_batches_entreno)
        error_generadores_entreno = np.ones(numero_batches_entreno)
        validez_generadores_entreno = np.ones(numero_batches_entreno)
        error_reconstruccion_generadores_entreno = np.ones(numero_batches_entreno)
        error_identidad_generadores_entreno = np.ones(numero_batches_entreno)

        error_discriminadores_test = np.ones(numero_batches_test)
        precision_discriminadores_test = np.ones(numero_batches_test)
        error_generadores_test = np.ones(numero_batches_test)
        validez_generadores_test = np.ones(numero_batches_test)
        error_reconstruccion_generadores_test = np.ones(numero_batches_test)
        error_identidad_generadores_test = np.ones(numero_batches_test)

        hay_gsutil = gsutil_disponible()  # Booleano para saber si tenemos que escribir o no en gcp

        # Comenzamos
        comienzo_entrenamiento = timestamp()

        for epoch in range(self.epoch_actual, self.utils.obtener_epochs()):

            if hay_gsutil:
                self.utils.copiar_logs_gcp()

            comienzo_epoch = timestamp()
            self.logger.info("epoch " + str(epoch))

            # paso de entreno
            for indice, (imagenes_pintor, imagenes_foto) in enumerate(lector_imagenes.cargar_batch(entreno=True)):
                error_discriminadores_actual = self._entrenar_discriminadores(imagenes_pintor, imagenes_foto,
                                                                              umbral_verdad, umbral_falso)
                error_generadores_actual = self._entrenar_generadores(imagenes_pintor, imagenes_foto, umbral_verdad)

                error_discriminadores_entreno[indice] = error_discriminadores_actual[0]
                precision_discriminadores_entreno[indice] = 100 * error_discriminadores_actual[1]
                error_generadores_entreno[indice] = error_generadores_actual[0]
                validez_generadores_entreno[indice] = np.mean(error_generadores_actual[1:3])
                error_reconstruccion_generadores_entreno[indice] = np.mean(error_generadores_actual[3:5])
                error_identidad_generadores_entreno[indice] = np.mean(error_generadores_actual[5:6])

            # paso test
            for indice, (imagenes_pintor, imagenes_foto) in enumerate(lector_imagenes.cargar_batch(entreno=False)):
                error_discriminadores_actual = self._entrenar_discriminadores(imagenes_pintor, imagenes_foto,
                                                                              umbral_verdad, umbral_falso)
                error_generadores_actual = self._entrenar_generadores(imagenes_pintor, imagenes_foto, umbral_verdad)

                error_discriminadores_test[indice] = error_discriminadores_actual[0]
                precision_discriminadores_test[indice] = 100 * error_discriminadores_actual[1]
                error_generadores_test[indice] = error_generadores_actual[0]
                validez_generadores_test[indice] = np.mean(error_generadores_actual[1:3])
                error_reconstruccion_generadores_test[indice] = np.mean(error_generadores_actual[3:5])
                error_identidad_generadores_test[indice] = np.mean(error_generadores_actual[5:6])

            self._escribir_metricas_escalares(error_discriminadores_entreno, precision_discriminadores_entreno,
                                              error_generadores_entreno, validez_generadores_entreno,
                                              error_reconstruccion_generadores_entreno,
                                              error_identidad_generadores_entreno, sumario_entreno, epoch)

            self._escribir_metricas_escalares(error_discriminadores_test, precision_discriminadores_test,
                                              error_generadores_test, validez_generadores_test,
                                              error_reconstruccion_generadores_test,
                                              error_identidad_generadores_test, sumario_test, epoch)

            self._imagen_muestra(imagen_muestra_pintor, imagen_muestra_foto, epoch)
            fin_epoch = timestamp()

            if epoch + 1 % 5 == 0:
                self._guardar_progreso(epoch)

            self.logger.info("epoch " + str(epoch) + " completado en " + str(fin_epoch - comienzo_epoch))

            self.epoch_actual += 1

        fin_entrenamiento = timestamp()
        self.logger.info("Entrenamiento completado en " + str(fin_entrenamiento - comienzo_entrenamiento))
        self._guardar_modelo()

    def convertir_imagen(self, ruta_imagen, modo_destino, factor_aumento):
        # obtenemos la extension de la imagen
        extension = ruta_imagen.split(".")[-1]
        if extension.lower() == "jpg":
            extension = "jpeg"
        # preprocesamos
        imagen = preprocesar_imagen_individual(ruta_imagen, self.dimensiones).numpy()
        # obtenemos la prediccion
        imagen_predecida = self._predecir_imagen(imagen, modo_destino)
        # ampliamos y devolvemos
        return ampliar(imagen_predecida, extension, factor_aumento, self.utils.obtener_url_api_aumento())

    def _crear_modelo(self):

        # Entrada de los dos dominios
        imagen_pintor = Input(shape=self.dimensiones)
        imagen_foto = Input(shape=self.dimensiones)

        # Traducción al otro dominio
        falsificacion_foto = self.generador_foto(imagen_pintor)
        falsificacion_pintor = self.generador_pintor(imagen_foto)
        # Reconstrucción de ambos dominios
        reconstruccion_pintor = self.generador_pintor(falsificacion_foto)
        reconstruccion_foto = self.generador_foto(falsificacion_pintor)
        # Generación al mismo dominio
        imagen_pintor_identidad = self.generador_pintor(imagen_pintor)
        imagen_foto_identidad = self.generador_foto(imagen_foto)

        # Los discriminadores determinan la validez de las imágenes "traducidas"
        validador_pintor = self.discriminador_pintor(falsificacion_pintor)
        validador_foto = self.discriminador_foto(falsificacion_foto)

        # El modelo cominado entrena los generadores para engañar a los discriminadores
        return Model(inputs=[imagen_pintor, imagen_foto],
                     outputs=[validador_pintor, validador_foto,
                              reconstruccion_pintor, reconstruccion_foto,
                              imagen_pintor_identidad, imagen_foto_identidad])

    def _entrenar_discriminadores(self, imagenes_pintor, imagenes_foto, umbral_verdad, umbral_falso):

        # Traducir imagenes
        falsificacion_foto = self.generador_foto.predict(imagenes_pintor)
        falsificacion_pintor = self.generador_pintor.predict(imagenes_foto)

        # TEntrenamos los discriminadores
        error_discriminador_pintor_real = self.discriminador_pintor.train_on_batch(imagenes_pintor, umbral_verdad)
        error_discriminador_pintor_falsificacion = self.discriminador_pintor.train_on_batch(
            falsificacion_pintor, umbral_falso)
        error_discriminador_foto_real = self.discriminador_foto.train_on_batch(imagenes_foto, umbral_verdad)
        error_discriminador_foto_falsificacion = self.discriminador_foto.train_on_batch(falsificacion_foto,
                                                                                        umbral_falso)
        # Metricas
        error_discriminador_pintor = 0.5 * np.add(error_discriminador_pintor_real,
                                                  error_discriminador_pintor_falsificacion)
        error_discriminador_foto = 0.5 * np.add(error_discriminador_foto_real,
                                                error_discriminador_foto_falsificacion)

        # Error total discriminador
        return 0.5 * np.add(error_discriminador_pintor, error_discriminador_foto)

    def _test_discriminadores(self, imagenes_pintor, imagenes_foto, umbral_verdad, umbral_falso):

        # Traducir imagenes
        falsificacion_foto = self.generador_foto.predict(imagenes_pintor)
        falsificacion_pintor = self.generador_pintor.predict(imagenes_foto)

        # TEntrenamos los discriminadores
        error_discriminador_pintor_real = self.discriminador_pintor.test_on_batch(imagenes_pintor, umbral_verdad)
        error_discriminador_pintor_falsificacion = self.discriminador_pintor.test_on_batch(
            falsificacion_pintor, umbral_falso)
        error_discriminador_foto_real = self.discriminador_foto.test_on_batch(imagenes_foto, umbral_verdad)
        error_discriminador_foto_falsificacion = self.discriminador_foto.test_on_batch(falsificacion_foto,
                                                                                       umbral_falso)
        # Metricas
        error_discriminador_pintor = 0.5 * np.add(error_discriminador_pintor_real,
                                                  error_discriminador_pintor_falsificacion)
        error_discriminador_foto = 0.5 * np.add(error_discriminador_foto_real,
                                                error_discriminador_foto_falsificacion)

        # Error total discriminador
        return 0.5 * np.add(error_discriminador_pintor, error_discriminador_foto)

    def _entrenar_generadores(self, imagenes_pintor, imagenes_foto, umbral_verdad):
        return self.modelo_combinado.train_on_batch([imagenes_pintor, imagenes_foto],
                                                    [umbral_verdad, umbral_verdad,
                                                     imagenes_pintor, imagenes_foto,
                                                     imagenes_pintor, imagenes_foto])

    def _test_generadores(self, imagenes_pintor, imagenes_foto, umbral_verdad):
        return self.modelo_combinado.test_on_batch([imagenes_pintor, imagenes_foto],
                                                   [umbral_verdad, umbral_verdad,
                                                    imagenes_pintor, imagenes_foto,
                                                    imagenes_pintor, imagenes_foto])

    def _predecir_imagen(self, imagen, modo_destino):
        if modo_destino.lower() == "pintor":
            imagen_predecida = self.generador_pintor.predict(imagen)
        else:
            imagen_predecida = self.generador_foto.predict(imagen)
        return imagen_predecida

    def _imagen_muestra(self, imagen_pintor, imagen_foto, epoch):
        file_writer = tf.summary.create_file_writer(self.utils.obtener_ruta_logs())

        filas, columnas = 2, 3

        estimacion_foto = self._predecir_imagen(imagen_pintor, "foto")
        estimacion_pintor = self._predecir_imagen(imagen_foto, "pintor")
        pintor_reconstruido = self._predecir_imagen(estimacion_foto, "pintor")
        foto_reconstruida = self._predecir_imagen(estimacion_pintor, "foto")

        imagenes = np.concatenate(
            [imagen_pintor, estimacion_foto, pintor_reconstruido, imagen_foto, estimacion_pintor, foto_reconstruida])

        # Rescale images 0 - 1
        #imagenes = 0.5 * imagenes + 0.5

        titles = ['Original', 'Traducida', 'Reconstruida']
        figura, ejes = plt.subplots(filas, columnas)
        contador = 0
        for fila in range(filas):
            for columna in range(columnas):
                ejes[fila, columna].imshow(imagenes[contador])
                ejes[fila, columna].set_title(titles[columna])
                ejes[fila, columna].axis('off')
                contador += 1

        """"Convierte un gráfico de matplotlib a un tensor de una imagen png para Tensorboard"""
        # Guardamos el grafico
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=400)
        # Cerramos la figura
        plt.close(figura)
        buffer.seek(0)
        # Pasamos el buffer png a un tensor primero 3D y luego 4D, lo que necesita tensorboard
        imagen_tf = tf.image.decode_png(buffer.getvalue(), channels=4)
        imagen_tf = tf.expand_dims(imagen_tf, 0)

        with file_writer.as_default():
            tf.summary.image("Imagen resumen", imagen_tf, step=epoch)

    def serializar_red(self):

        with open(self.utils.obtener_ruta_archivo_modelo_parametros(), 'wb') as archivo:
            pkl.dump([self.tasa_aprendizaje, self.lambda_reconstruccion, self.lambda_validacion, self.lambda_identidad,
                      self.dimensiones, self.filtros_generador, self.filtros_discriminador],
                     archivo)

        plot_model(self.modelo_combinado, to_file=self.utils.obtener_ruta_archivo_modelo_esquema(),
                   show_shapes=True, show_layer_names=True, dpi=300)
        plot_model(self.discriminador_pintor,
                   to_file=self.utils.obtener_ruta_archivo_discriminador_pintor_esquema(),
                   show_shapes=True, show_layer_names=True, dpi=300)
        plot_model(self.discriminador_foto, to_file=self.utils.obtener_ruta_archivo_discriminador_foto_esquema(),
                   show_shapes=True, show_layer_names=True, dpi=300)
        plot_model(self.generador_pintor, to_file=self.utils.obtener_ruta_archivo_generador_pintor_esquema(),
                   show_shapes=True, show_layer_names=True, dpi=300)
        plot_model(self.generador_foto, to_file=self.utils.obtener_ruta_archivo_generador_foto_esquema(),
                   show_shapes=True, show_layer_names=True, dpi=300)

    def _guardar_modelo(self):
        self.modelo_combinado.save(self.utils.obtener_ruta_fichero_modelo())
        self.discriminador_pintor.save(self.utils.obtener_ruta_fichero_discriminador_pintor())
        self.discriminador_foto.save(self.utils.obtener_ruta_fichero_discriminador_foto())
        self.generador_pintor.save(self.utils.obtener_ruta_fichero_generador_pintor())
        self.generador_foto.save(self.utils.obtener_ruta_fichero_generador_foto())

        with open(self.utils.obtener_ruta_archivo_modelo_objeto(), "wb") as archivo:
            pkl.dump(self, archivo)

    def cargar_ultimos_pesos(self):
        self.logger.info("Cargamos los pesos más recientes")
        ruta_ultimo_checkpoint, self.epoch_actual = self.utils.obtener_ultimos_pesos()
        self.logger.info("Los ultimos pesos eran del epoch: " + str(self.epoch_actual))
        if ruta_ultimo_checkpoint is not None:
            self.modelo_combinado.load_weights(ruta_ultimo_checkpoint)

    def _guardar_progreso(self, epoch):
        self.modelo_combinado.save_weights(self.utils.obtener_ruta_fichero_modelo_por_epoch(epoch + 1))

    @staticmethod
    def _escribir_metricas_escalares(error_discriminadores, precision_discriminadores, error_generadores,
                                     validez_generadores, error_reconstruccion_generadores,
                                     error_identidad_generadores, writer, step):

        with writer.as_default():
            tf.summary.scalar("Media del error de los discriminadores", np.mean(error_discriminadores), step=step)
            tf.summary.scalar("Media de la precision de los discriminadores", np.mean(precision_discriminadores),
                              step=step)
            tf.summary.scalar("Media del error de los generadores", np.mean(error_generadores), step=step)
            tf.summary.scalar("Media del error de la validez de los generadores", np.mean(validez_generadores),
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
            tf.summary.scalar("Desviación estándar de la validez de los generadores",
                              np.std(validez_generadores), step=step)
            tf.summary.scalar("Desviación estándar del error de reconstruccion de los generadores",
                              np.std(error_reconstruccion_generadores), step=step)
            tf.summary.scalar("Desviación estándar del error de identidad de los generadores",
                              np.std(error_identidad_generadores), step=step)
