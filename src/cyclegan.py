from __future__ import division, print_function

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Concatenate, Dropout, Input
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.models import load_model, Model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from tensorflow.keras.optimizers import Adam

from src.parametros_modelo import *
from src.utilidades import *


def _build_generator():
    """U-Net Generator"""

    def conv2d(layer_input, filters, f_size=4):
        """Layers used during downsampling"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        d = InstanceNormalization()(d)
        return d

    def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
        """Layers used during upsampling"""
        u = UpSampling2D(size=2)(layer_input)
        u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
        if dropout_rate:
            u = Dropout(dropout_rate)(u)
        u = InstanceNormalization()(u)
        u = Concatenate()([u, skip_input])
        return u

    # Image input
    d0 = Input(shape=(ancho, alto, canales))

    # Downsampling
    d1 = conv2d(d0, gf)
    d2 = conv2d(d1, gf * 2)
    d3 = conv2d(d2, gf * 4)
    d4 = conv2d(d3, gf * 8)

    # Upsampling
    u1 = deconv2d(d4, d3, gf * 4)
    u2 = deconv2d(u1, d2, gf * 2)
    u3 = deconv2d(u2, d1, gf)

    u4 = UpSampling2D(size=2)(u3)
    output_img = Conv2D(canales, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

    return Model(d0, output_img)


def _build_discriminator():
    def d_layer(layer_input, filters, f_size=4, normalization=True):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if normalization:
            d = InstanceNormalization()(d)
        return d

    img = Input(shape=(ancho, alto, canales))

    d1 = d_layer(img, df, normalization=False)
    d2 = d_layer(d1, df * 2)
    d3 = d_layer(d2, df * 4)
    d4 = d_layer(d3, df * 8)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

    return Model(img, validity)


class CycleGAN:
    __slots__ = ['modelo_entrenado', 'disc_patch', 'd_A', 'd_B', 'g_AB', 'g_BA', 'modelo', 'utils']

    def __init__(self):

        self.modelo_entrenado = False

        self.utils = Utilidades()

        # Calculate output shape of D (PatchGAN)
        patch = int(alto / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        optimizer = Adam(learning_rate, 0.5)

        # Build and compile the discriminators
        self.d_A = _build_discriminator()
        self.d_B = _build_discriminator()
        self.d_A.compile(loss='mse',
                         optimizer=optimizer,
                         metrics=['accuracy'])
        self.d_B.compile(loss='mse',
                         optimizer=optimizer,
                         metrics=['accuracy'])

        # -------------------------
        # Construct Computational
        #   Graph of Generators
        # -------------------------

        # Build the generators
        self.g_AB = _build_generator()
        self.g_BA = _build_generator()

        # Input images from both domains
        img_A = Input(shape=(ancho, alto, canales))
        img_B = Input(shape=(ancho, alto, canales))

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.modelo = Model(inputs=[img_A, img_B],
                            outputs=[valid_A, valid_B,
                                     reconstr_A, reconstr_B,
                                     img_A_id, img_B_id])
        self.modelo.compile(loss=['mse', 'mse',
                                  'mae', 'mae',
                                  'mae', 'mae'],
                            loss_weights=[1, 1,
                                          lambda_cycle, lambda_cycle,
                                          lambda_id, lambda_id],
                            optimizer=optimizer)

    def train(self, lector_imagenes, epochs=300, batch_size=1):

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU

        writer_train = tf.summary.create_file_writer(self.utils.obtener_ruta_logs_train())

        start_time = timestamp()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        imagen_muestra_pintor = lector_imagenes.obtener_imagen_muestra_pintor()
        imagen_muestra_real = lector_imagenes.obtener_imagen_muestra_real()

        for epoch in range(epochs):
            n_batches = lector_imagenes.get_n_batches(batch_size)
            d_losses = np.empty([n_batches, 2])
            g_losses = np.empty([n_batches, 7])
            for batch_i, (imgs_A, imgs_B) in enumerate(lector_imagenes.load_batch(batch_size)):
                # ----------------------
                #  Train Discriminators
                # ----------------------
                # Translate images to opposite domain
                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)

                # Train the discriminators (original images = real / translated = Fake)
                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                # Total disciminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)

                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.modelo.train_on_batch([imgs_A, imgs_B],
                                                    [valid, valid,
                                                     imgs_A, imgs_B,
                                                     imgs_A, imgs_B])

                d_losses[batch_i] = d_loss
                g_losses[batch_i] = g_loss

            self.escribir_metricas_perdidas(d_losses, g_losses, writer_train, epoch)

            start_time_test = timestamp()
            elapsed_time_train = start_time_test - start_time

            self.test(epoch, lector_imagenes)

            elapsed_time_test = timestamp() - start_time_test

            print("[Epoch %d/%d] time training: %s time testing: %s " % (
                epoch, epochs, elapsed_time_train, elapsed_time_test))

            self._sample_images(epoch, imagen_muestra_pintor, imagen_muestra_real)

        self._final_images(timestamp_fancy(), imagen_muestra_pintor, imagen_muestra_real)
        self._guardar_modelo(timestamp_fancy())

        self.modelo_entrenado = True

    def test(self, step, lector_imagenes):
        writer_test = tf.summary.create_file_writer(self.utils.obtener_ruta_logs_test())
        batch_size = 32
        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        n_batches = lector_imagenes.get_n_batches(batch_size)
        d_losses = np.empty([n_batches, 2])
        g_losses = np.empty([n_batches, 7])

        for batch_i, (imgs_A, imgs_B) in enumerate(lector_imagenes.load_batch(batch_size, is_training=False)):
            fake_B = self.g_AB.predict(imgs_A)
            fake_A = self.g_BA.predict(imgs_B)

            # Train the discriminators (original images = real / translated = Fake)
            dA_loss_real = self.d_A.test_on_batch(imgs_A, valid)
            dA_loss_fake = self.d_A.test_on_batch(fake_A, fake)
            dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

            dB_loss_real = self.d_B.test_on_batch(imgs_B, valid)
            dB_loss_fake = self.d_B.test_on_batch(fake_B, fake)
            dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

            # Total disciminator loss
            d_loss = 0.5 * np.add(dA_loss, dB_loss)

            g_loss = self.modelo.test_on_batch([imgs_A, imgs_B],
                                               [valid, valid,
                                                imgs_A, imgs_B,
                                                imgs_A, imgs_B])

            d_losses[batch_i] = d_loss
            g_losses[batch_i] = g_loss

        self.escribir_metricas_perdidas(d_losses, g_losses, writer_test, step)

    def _sample_images(self, epoch, imagen_pintor, imagen_real):

        # Translate images to the other domain
        fake_B = self.g_AB.predict(imagen_pintor)
        fake_A = self.g_BA.predict(imagen_real)
        # Translate back to original domain
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)

        file_writer = tf.summary.create_file_writer(self.utils.obtener_ruta_logs())

        # Using the file writer, log the reshaped image.
        with file_writer.as_default():
            tf.summary.image("Pintor original", imagen_pintor, step=epoch)
            tf.summary.image("Estimacion foto", fake_B, step=epoch)
            tf.summary.image("Pintor reconstruida", reconstr_A, step=epoch)
            tf.summary.image("Foto original", imagen_real, step=epoch)
            tf.summary.image("Estimacion pintor", fake_A, step=epoch)
            tf.summary.image("Foto reconstruida", reconstr_B, step=epoch)

    def _final_images(self, nombre, imagen_pintor, imagen_real):
        r, c = 2, 3

        # Translate images to the other domain
        fake_B = self.g_AB.predict(imagen_pintor)
        fake_A = self.g_BA.predict(imagen_real)
        # Translate back to original domain
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)

        gen_imgs = np.concatenate([imagen_pintor, fake_B, reconstr_A, imagen_real, fake_A, reconstr_B])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig(self.utils.obtener_ruta_imagen_a_guardar(nombre))
        plt.close()

    def _guardar_modelo(self, nombre):
        self.modelo.save(self.utils.obtener_ruta_modelo_a_guardar(nombre))

    @staticmethod
    def escribir_metricas_perdidas(d_losses, g_losses, writer, step):
        d_loss = d_losses[:, 0]
        d_loss_acc = 100 * d_losses[:, 1]
        g_loss = g_losses[:, 0]
        g_loss_adv = g_losses[:, 1:3]
        g_loss_recon = g_losses[:, 3:5]
        g_loss_id = g_losses[:, 5:6]

        with writer.as_default():
            tf.summary.scalar("d_loss mean", np.mean(d_loss), step=step)
            tf.summary.scalar("d_loss_acc mean", np.mean(d_loss_acc), step=step)
            tf.summary.scalar("g_loss mean", np.mean(g_loss), step=step)
            tf.summary.scalar("g_loss_adv mean", np.mean(g_loss_adv), step=step)
            tf.summary.scalar("g_loss_recon mean", np.mean(g_loss_recon), step=step)
            tf.summary.scalar("g_loss_id mean", np.mean(g_loss_id), step=step)
            tf.summary.scalar("d_loss std", np.std(d_loss), step=step)
            tf.summary.scalar("d_loss_acc std", np.std(d_loss_acc), step=step)
            tf.summary.scalar("g_loss std", np.std(g_loss), step=step)
            tf.summary.scalar("g_loss_adv std", np.std(g_loss_adv), step=step)
            tf.summary.scalar("g_loss_recon std", np.std(g_loss_recon), step=step)
            tf.summary.scalar("g_loss_id std", np.std(g_loss_id), step=step)
            writer.flush()

    def load_model(self, ruta):
        self.modelo = load_model(ruta)

    def crear_imagen(self, imagen, ruta_modelo=None, pintar_cuadro=True):
        if not self.modelo_entrenado:
            if ruta_modelo is not None:
                self.load_model(ruta_modelo)
            else:
                print("No hay modelo entrenado para generar la imagen")
        if pintar_cuadro:
            resultado = self.g_AB.predict(imagen)
        else:
            resultado = self.g_BA.predict(imagen)
        return resultado
