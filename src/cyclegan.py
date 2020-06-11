from __future__ import print_function, division

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from keras.layers import Input, Dropout, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam

import funciones
from constantes import Constantes
from lector_imagenes import LectorImagenes


class CycleGAN:
    def __init__(self, learning_rate=0.0002, lambda_cycle=10.0):
        # Input shape

        # Crear lector de datos y constantes
        self.lector_imagenes = LectorImagenes()
        self.constantes = Constantes()

        # Calculate output shape of D (PatchGAN)
        patch = int(self.constantes.alto / 2 ** 4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 64

        # Loss weights
        self.lambda_cycle = lambda_cycle  # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle  # Identity loss

        optimizer = Adam(learning_rate, 0.5)

        # Build and compile the discriminators
        self.d_A = self._build_discriminator()
        self.d_B = self._build_discriminator()
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
        self.g_AB = self._build_generator()
        self.g_BA = self._build_generator()

        # Input images from both domains
        img_A = Input(shape=self.constantes.forma)
        img_B = Input(shape=self.constantes.forma)

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
                                          self.lambda_cycle, self.lambda_cycle,
                                          self.lambda_id, self.lambda_id],
                            optimizer=optimizer)

    def _build_generator(self):
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
        d0 = Input(shape=self.constantes.forma)

        # Downsampling
        d1 = conv2d(d0, self.gf)
        d2 = conv2d(d1, self.gf * 2)
        d3 = conv2d(d2, self.gf * 4)
        d4 = conv2d(d3, self.gf * 8)

        # Upsampling
        u1 = deconv2d(d4, d3, self.gf * 4)
        u2 = deconv2d(u1, d2, self.gf * 2)
        u3 = deconv2d(u2, d1, self.gf)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.constantes.canales, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model(d0, output_img)

    def _build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.constantes.forma)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df * 2)
        d3 = d_layer(d2, self.df * 4)
        d4 = d_layer(d3, self.df * 8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)

    def train(self, epochs, batch_size=1):

        tensorboard = TensorBoard(
            log_dir=self.constantes.ruta_logs,
            histogram_freq=0,
            write_graph=True,
        )
        tensorboard.set_model(self.modelo)

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU

        start_time = funciones.timestamp()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B) in enumerate(self.lector_imagenes.load_batch(batch_size)):
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
                                                     imgs_A, imgs_B], )

            elapsed_time = funciones.timestamp() - start_time

            metricas = self._hacer_metricas(d_loss, g_loss)
            # Plot the progress
            print(
                "[Epoch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, "
                "id: %05f] time: %s "
                % (epoch, epochs,
                   metricas.get("D loss"),
                   metricas.get("D loss acc"),
                   metricas.get("G loss"),
                   metricas.get("G loss adv"),
                   metricas.get("G loss recon"),
                   metricas.get("G loss id"),
                   elapsed_time))

            self._sample_images(epoch)
            self._guardar_modelo(self.constantes.ruta_checkpoints_modelo, str(epoch))
            tensorboard.on_epoch_end(epoch, metricas)

        self._final_images(self.constantes.ruta_imagenes, funciones.timestamp_fancy())
        self._guardar_modelo(self.constantes.ruta_modelo, funciones.timestamp_fancy())

    def _sample_images(self, epoch):

        # Demo (for GIF)
        imgs_A = self.lector_imagenes.load_single_img(self.constantes.imagen_muestra_pintor)
        imgs_B = self.lector_imagenes.load_single_img(self.constantes.imagen_muestra_real)

        # Translate images to the other domain
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)

        file_writer = tf.summary.create_file_writer(self.constantes.ruta_logs)

        # Using the file writer, log the reshaped image.
        with file_writer.as_default():
            tf.summary.image("Pintor original", imgs_A, step=epoch)
            tf.summary.image("Estimacion foto", fake_B, step=epoch)
            tf.summary.image("Pintor reconstruida", reconstr_A, step=epoch)
            tf.summary.image("Foto original", imgs_B, step=epoch)
            tf.summary.image("Estimacion pintor", fake_A, step=epoch)
            tf.summary.image("Foto reconstruida", reconstr_B, step=epoch)

    def _final_images(self, directorio, texto):
        r, c = 2, 3

        # Demo (for GIF)
        imgs_A = self.lector_imagenes.load_single_img(self.constantes.imagen_muestra_pintor)
        imgs_B = self.lector_imagenes.load_single_img(self.constantes.imagen_muestra_real)

        # Translate images to the other domain
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)

        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])

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
        fig.savefig(directorio + self.constantes.sep + texto + ".png")
        plt.close()

    def _guardar_modelo(self, directorio, texto):
        self.modelo.save(directorio + self.constantes.sep + texto + ".h5")

    @staticmethod
    def _hacer_metricas(d_loss, g_loss):
        return {
            "D loss": d_loss[0],
            "D loss acc": 100 * d_loss[1],
            "G loss": g_loss[0],
            "G loss adv": np.mean(g_loss[1:3]),
            "G loss recon": np.mean(g_loss[3:5]),
            "G loss id": np.mean(g_loss[5:6])
        }
