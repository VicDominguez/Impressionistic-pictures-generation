from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

from parametros_modelo import *
from utilidades import *


def calcular_error_ciclo(imagen_real, imagen_ciclada):
    return parametro_lambda * tf.reduce_mean(tf.abs(imagen_real - imagen_ciclada))


def calcular_error_identidad(imagen_real, misma_imagen):
    return parametro_lambda * 0.5 * tf.reduce_mean(tf.abs(imagen_real - misma_imagen))


class CycleGAN:
    # TODO realizar slots
    # __slots__ = ['modelo_entrenado', 'disc_patch', 'discriminador_A', 'discriminador_B', 'generador_foto',
    # 'generador_pintor', 'modelo','utils', 'logger']

    # TODO A es pintor, B es foto

    def __init__(self):

        tf.autograph.set_verbosity(2)

        # TODO es necesario utils y logger?
        self.utils = Utilidades()
        self.logger = self.utils.obtener_logger("cyclegan")

        self.funcion_error = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.generador_foto = pix2pix.unet_generator(canales, norm_type='instancenorm')
        self.generador_pintor = pix2pix.unet_generator(canales, norm_type='instancenorm')

        self.discriminador_pintor = pix2pix.discriminator(norm_type='instancenorm', target=False)
        self.discriminador_foto = pix2pix.discriminator(norm_type='instancenorm', target=False)

        self.optimizador_discriminador_pintor = tf.keras.optimizers.Adam(learning_rate=tasa_aprendizaje, beta_1=0.5)
        self.optimizador_discriminador_foto = tf.keras.optimizers.Adam(learning_rate=tasa_aprendizaje, beta_1=0.5)
        self.optimizador_generador_pintor = tf.keras.optimizers.Adam(learning_rate=tasa_aprendizaje, beta_1=0.5)
        self.optimizador_generador_foto = tf.keras.optimizers.Adam(learning_rate=tasa_aprendizaje, beta_1=0.5)

        ckpt = tf.train.Checkpoint(generador_foto=self.generador_foto,
                                   generador_pintor=self.generador_pintor,
                                   discriminador_pintor=self.discriminador_pintor,
                                   discriminador_foto=self.discriminador_foto,
                                   optimizador_discriminador_pintor=self.optimizador_discriminador_pintor,
                                   optimizador_discriminador_foto=self.optimizador_discriminador_foto,
                                   optimizador_generador_pintor=self.optimizador_generador_pintor,
                                   optimizador_generador_foto=self.optimizador_generador_foto)

        self.ckpt_manager = tf.train.CheckpointManager(ckpt, self.utils.obtener_ruta_checkpoints(), max_to_keep=5)
        if self.ckpt_manager.latest_checkpoint:
            ckpt.restore(self.ckpt_manager.latest_checkpoint)
            self.logger("Punto de control restaurado")

    def train(self, lector_imagenes):

        comienzo_entrenamiento = timestamp()
        writer_train = tf.summary.create_file_writer(self.utils.obtener_ruta_logs_train())

        imagen_muestra_pintor = lector_imagenes.obtener_imagen_muestra_pintor()
        imagen_muestra_foto = lector_imagenes.obtener_imagen_muestra_foto()

        n_batches_train = lector_imagenes.calcular_n_batches(is_training=True)
        n_batches_test = lector_imagenes.calcular_n_batches(is_training=False)

        edp_train = np.ones(n_batches_train)
        edf_train = np.ones(n_batches_train)
        etgp_train = np.ones(n_batches_train)
        etgf_train = np.ones(n_batches_train)
        egp_train = np.ones(n_batches_train)
        egf_train = np.ones(n_batches_train)
        eigp_train = np.ones(n_batches_train)
        eigf_train = np.ones(n_batches_train)
        ect_train = np.ones(n_batches_train)

        edp_test = np.ones(n_batches_test)
        edf_test = np.ones(n_batches_test)
        etgp_test = np.ones(n_batches_test)
        etgf_test = np.ones(n_batches_test)
        egp_test = np.ones(n_batches_test)
        egf_test = np.ones(n_batches_test)
        eigp_test = np.ones(n_batches_test)
        eigf_test = np.ones(n_batches_test)
        ect_test = np.ones(n_batches_test)

        for epoch in range(epochs):
            self.utils.copiar_logs_gcp()
            comienzo_epoch = timestamp()
            self.logger.info("epoch " + str(epoch))
            for indice, (imagenes_pintor, imagenes_foto) in enumerate(lector_imagenes.cargar_batch()):
                edp_train[indice], edf_train[indice], etgp_train[indice], etgf_train[indice], egp_train[indice], \
                egf_train[indice], eigp_train[indice], eigf_train[indice], ect_train[indice] = self._train_step(
                    imagenes_pintor, imagenes_foto)

                # TODO test

            fin_epoch = timestamp()
            self.escribir_metricas_perdidas(edp_train, edf_train, etgp_train, etgf_train, egp_train, egf_train,
                                            eigp_train, eigf_train, ect_train, writer_train, epoch)
            self.imagenes_epoch(epoch, imagen_muestra_pintor, imagen_muestra_foto)

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                self.logger.info('Guardando checkpoint para el epoch {} en {}'.format(epoch + 1, ckpt_save_path))

            self.logger.info(
                'Tiempo transcurrido en el epoch {} es {} segundos'.format(epoch + 1, fin_epoch - comienzo_epoch))

        self._imagenes_final = self._images_final(timestamp_fancy(), imagen_muestra_pintor, imagen_muestra_foto)

        # batch_size = batch_size_train
        # # config = tf.compat.v1.ConfigProto()
        # # config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        #
        # writer_train = tf.summary.create_file_writer(self.utils.obtener_ruta_logs_train())
        #
        # start_time_train = timestamp()
        #
        # # Adversarial loss ground truths
        # valid = np.ones((batch_size,) + self.disc_patch)
        # fake = np.zeros((batch_size,) + self.disc_patch)
        #
        # imagen_muestra_pintor = lector_imagenes.obtener_imagen_muestra_pintor().numpy()
        # imagen_muestra_foto = lector_imagenes.obtener_imagen_muestra_foto().numpy()
        #
        # for epoch in range(epochs):
        #     start_time_epoch = timestamp()
        #     n_batches = lector_imagenes.get_n_batches(batch_size)
        #     d_losses = np.empty([n_batches, 2])
        #     g_losses = np.empty([n_batches, 7])
        #     for batch_i, (imgs_A, imgs_B) in enumerate(lector_imagenes.loadiscriminador_Batch()):
        #         d_losses[batch_i], g_losses[batch_i] = self._train_step(imgs_A.numpy(), imgs_B.numpy(), valid, fake)
        #
        #     self.escribir_metricas_perdidas(d_losses, g_losses, writer_train, epoch)
        #
        #     start_time_test = timestamp()
        #     elapsed_time_train = start_time_test - start_time_epoch
        #
        #     self.test(epoch, lector_imagenes)
        #
        #     elapsed_time_test = timestamp() - start_time_test
        #
        #     self.logger.info("[Epoch %d/%d] time training: %s time testing: %s Total time %s" % (
        #         epoch, epochs, elapsed_time_train, elapsed_time_test, (timestamp() - start_time_train)))
        #
        #     self._sample_images(epoch, imagen_muestra_pintor, imagen_muestra_foto)
        #
        # self._final_images(timestamp_fancy(), imagen_muestra_pintor, imagen_muestra_foto)
        # self._guardar_modelo(timestamp_fancy())

    @tf.function
    def _train_step(self, pintor_real, foto_real):
        # persistent is set to True because the tape is used more than
        # once to calculate the gradients.
        with tf.GradientTape(persistent=True) as tape:
            # El generator foto traduce pintor -> foto
            # El generador pintor traduce foto -> pintor

            estimacion_foto = self.generador_foto(pintor_real, training=True)
            pintor_ciclado = self.generador_pintor(estimacion_foto, training=True)

            estimacion_pintor = self.generador_pintor(foto_real, training=True)
            foto_ciclada = self.generador_foto(estimacion_pintor, training=True)

            # same_pintor and same_foto are used for identity loss.
            mismo_pintor = self.generador_pintor(pintor_real, training=True)
            mismo_foto = self.generador_foto(foto_real, training=True)

            disc_pintor_real = self.discriminador_pintor(pintor_real, training=True)
            disc_foto_real = self.discriminador_foto(foto_real, training=True)

            disc_estimacion_pintor = self.discriminador_pintor(estimacion_pintor, training=True)
            disc_estimacion_foto = self.discriminador_foto(estimacion_foto, training=True)

            # calculate the loss
            error_generador_foto = self.error_generador(disc_estimacion_foto)
            error_generador_pintor = self.error_generador(disc_estimacion_pintor)

            error_ciclo_total = calcular_error_ciclo(pintor_real, pintor_ciclado) + calcular_error_ciclo(foto_real,
                                                                                                         foto_ciclada)

            # Total generator loss = adversarial loss + cycle loss

            error_identidad_generador_foto = calcular_error_identidad(foto_real, mismo_foto)
            error_identidad_generador_pintor = calcular_error_identidad(pintor_real, mismo_pintor)
            error_total_generador_foto = error_generador_foto + error_ciclo_total + error_identidad_generador_foto
            error_total_generador_pintor = error_generador_pintor + error_ciclo_total + error_identidad_generador_pintor

            error_discriminador_pintor = self.error_discriminador(disc_pintor_real, disc_estimacion_pintor)
            error_discriminador_foto = self.error_discriminador(disc_foto_real, disc_estimacion_foto)

        # Calculate the gradients for generator and discriminator
        self.generador_foto_gradientes = tape.gradient(error_total_generador_foto,
                                                       self.generador_foto.trainable_variables)
        self.generador_pintor_gradientes = tape.gradient(error_total_generador_pintor,
                                                         self.generador_pintor.trainable_variables)
        # self.logger("generador_pintor_gradientes" + str(self.generador_foto_gradientes))

        discriminator_pintor_gradientes = tape.gradient(error_discriminador_pintor,
                                                        self.discriminador_pintor.trainable_variables)
        discriminator_foto_gradientes = tape.gradient(error_discriminador_foto,
                                                      self.discriminador_foto.trainable_variables)
        # self.logger("discriminator_foto_gradientes" + str(self.generador_foto_gradientes))

        # Apply the gradients to the optimizer
        self.optimizador_generador_foto.apply_gradients(zip(self.generador_foto_gradientes,
                                                            self.generador_foto.trainable_variables))

        self.optimizador_generador_pintor.apply_gradients(zip(self.generador_pintor_gradientes,
                                                              self.generador_pintor.trainable_variables))

        self.optimizador_discriminador_pintor.apply_gradients(zip(discriminator_pintor_gradientes,
                                                                  self.discriminador_pintor.trainable_variables))

        self.optimizador_discriminador_foto.apply_gradients(zip(discriminator_foto_gradientes,
                                                                self.discriminador_foto.trainable_variables))

        return error_discriminador_pintor, error_discriminador_foto, error_total_generador_pintor, \
               error_total_generador_foto, error_generador_pintor, error_generador_foto, \
               error_identidad_generador_pintor, error_identidad_generador_foto, error_ciclo_total

    def generator_loss(self, generated):
        return self.loss_obj(tf.ones_like(generated), generated)

    def discriminator_loss(self, real, generated):
        real_loss = self.loss_obj(tf.ones_like(real), real)

        generated_loss = self.loss_obj(tf.zeros_like(generated), generated)

        total_disc_loss = real_loss + generated_loss

        return total_disc_loss * 0.5

    def error_discriminador(self, real, generado):
        error_real = self.funcion_error(tf.ones_like(real), real)

        error_generado = self.funcion_error(tf.zeros_like(generado), generado)

        error_total_discriminador = error_real + error_generado

        return error_total_discriminador * 0.5

    def error_generador(self, generador):
        return self.funcion_error(tf.ones_like(generador), generador)

    def imagenes_epoch(self, epoch, imagen_pintor, imagen_foto):
        estimacion_foto, estimacion_pintor, pintor_reconstruido, foto_reconstruida = self._predecir_pares_imagenes(
            imagen_pintor, imagen_foto)
        self._imagenes_a_log(epoch, imagen_pintor, imagen_foto, estimacion_foto, estimacion_pintor, pintor_reconstruido,
                             foto_reconstruida)

    def _predecir_pares_imagenes(self, imagen_pintor, imagen_foto):
        # Translate images to the other domain
        estimacion_foto = self.generador_foto.predict(imagen_pintor)
        estimacion_pintor = self.generador_pintor.predict(imagen_foto)
        # Translate back to original domain
        pintor_reconstruido = self.generador_pintor.predict(estimacion_foto)
        foto_reconstruida = self.generador_foto.predict(estimacion_pintor)

        return estimacion_foto, estimacion_pintor, pintor_reconstruido, foto_reconstruida

    def _imagenes_a_log(self, epoch, imagen_pintor, imagen_foto, estimacion_foto, estimacion_pintor,
                        pintor_reconstruido, foto_reconstruida):

        file_writer = tf.summary.create_file_writer(self.utils.obtener_ruta_logs())

        # Using the file writer, log the reshaped image.
        with file_writer.as_default():
            tf.summary.image("Pintor original", imagen_pintor, step=epoch)
            tf.summary.image("Estimacion foto", estimacion_foto, step=epoch)
            tf.summary.image("Pintor reconstruida", pintor_reconstruido, step=epoch)
            tf.summary.image("Foto original", imagen_foto, step=epoch)
            tf.summary.image("Estimacion pintor", estimacion_pintor, step=epoch)
            tf.summary.image("Foto reconstruida", foto_reconstruida, step=epoch)

    def _imagenes_final(self, nombre, imagen_pintor, imagen_foto):
        filas, columnas = 2, 3

        estimacion_foto, estimacion_pintor, pintor_reconstruido, foto_reconstruida = self._predecir_pares_imagenes(
            imagen_pintor, imagen_foto)

        gen_imgs = np.concatenate(
            [imagen_pintor, estimacion_foto, pintor_reconstruido, imagen_foto, estimacion_pintor, foto_reconstruida])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(filas, columnas)
        cnt = 0
        for i in range(filas):
            for j in range(columnas):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(self.utils.obtener_archivo_imagen_a_guardar(nombre))
        plt.close()

    # def _guardar_modelo(self, nombre):
    #     self.modelo.save(self.utils.obtener_archivo_modelo_a_guardar(nombre))

    @staticmethod
    def escribir_metricas_perdidas(error_discriminador_pintor, error_discriminador_foto, error_total_generador_pintor,
                                   error_total_generador_foto, error_generador_pintor,
                                   error_generador_foto, error_identidad_generador_pintor,
                                   error_identidad_generador_foto, error_ciclo_total, writer, step):

        with writer.as_default():
            tf.summary.scalar("Media del error_discriminador_pintor", np.mean(error_discriminador_pintor), step=step)
            tf.summary.scalar("Media del error_discriminador_foto", np.mean(error_discriminador_foto), step=step)
            tf.summary.scalar("Media del error_total_generador_pintor", np.mean(error_total_generador_pintor),
                              step=step)
            tf.summary.scalar("Media del error_total_generador_foto", np.mean(error_total_generador_foto), step=step)
            tf.summary.scalar("Media del error_generador_pintor", np.mean(error_generador_pintor), step=step)
            tf.summary.scalar("Media del error_generador_foto", np.mean(error_generador_foto), step=step)
            tf.summary.scalar("Media del error_identidad_generador_pintor", np.mean(error_identidad_generador_pintor),
                              step=step)
            tf.summary.scalar("Media del error_identidad_generador_foto", np.mean(error_identidad_generador_foto),
                              step=step)
            tf.summary.scalar("Media del error_ciclo_total", np.mean(error_ciclo_total), step=step)

            tf.summary.scalar("Desviación estándar del error_discriminador_pintor", np.std(error_discriminador_pintor),
                              step=step)
            tf.summary.scalar("Desviación estándar del error_discriminador_foto", np.std(error_discriminador_foto),
                              step=step)
            tf.summary.scalar("Desviación estándar del error_total_generador_pintor",
                              np.std(error_total_generador_pintor), step=step)
            tf.summary.scalar("Desviación estándar del error_total_generador_foto", np.std(error_total_generador_foto),
                              step=step)
            tf.summary.scalar("Desviación estándar del error_generador_pintor", np.std(error_generador_pintor),
                              step=step)
            tf.summary.scalar("Desviación estándar del error_generador_foto", np.std(error_generador_foto), step=step)
            tf.summary.scalar("Desviación estándar del error_identidad_generador_pintor",
                              np.std(error_identidad_generador_pintor), step=step)
            tf.summary.scalar("Desviación estándar del error_identidad_generador_foto",
                              np.std(error_identidad_generador_foto), step=step)
            tf.summary.scalar("Desviación estándar del error_ciclo_total", np.std(error_ciclo_total), step=step)

    #         writer.flush()
    #
    # def load_model(self, ruta):
    #     self.modelo = load_model(ruta)
    #
    # def crear_imagen(self, imagen, pintar_cuadro=True):
    #     # if not self.modelo_entrenado:
    #     #     if ruta_modelo is not None:
    #     #         self.load_model(ruta_modelo)
    #     #     else:
    #     #         print("No hay modelo entrenado para generar la imagen")
    #     if pintar_cuadro:
    #         resultado = self.generador_foto.predict(imagen)
    #     else:
    #         resultado = self.generador_pintor.predict(imagen)
    #     return resultado
