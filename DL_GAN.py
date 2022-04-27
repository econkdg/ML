# ====================================================================================================
# GAN: general adversarial networks
# ====================================================================================================
# import library

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
# ----------------------------------------------------------------------------------------------------
# import data (1)

mnist = tf.keras.datasets.mnist

(train_x, train_y), _ = mnist.load_data()
train_x = train_x[np.where(train_y == 2)]
train_x = train_x/255.0
train_x = train_x.reshape(-1, 784)
# ----------------------------------------------------------------------------------------------------
# GAN (1): class version


class GAN:

    def __init__(self, G_n_input, G1_n_unit, G2_n_unit, D_n_input, D1_n_unit, D2_n_unit, D_LR, combined_LR, n_iter, n_batch):

        self.G_n_input = G_n_input
        self.G1_n_unit = G1_n_unit
        self.G2_n_unit = G2_n_unit

        self.D_n_input = D_n_input
        self.D1_n_unit = D1_n_unit
        self.D2_n_unit = D2_n_unit

        self.D_LR = D_LR
        self.combined_LR = combined_LR

        self.n_iter = n_iter
        self.n_batch = n_batch

    def GAN_TensorFlow(self):

        generator = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=self.G1_n_unit,
                                  input_dim=self.G_n_input, activation='relu'),
            tf.keras.layers.Dense(units=self.G2_n_unit, activation='sigmoid')
        ])

        discriminator = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=self.D1_n_unit,
                                  input_dim=self.D_n_input, activation='relu'),
            tf.keras.layers.Dense(units=self.D2_n_unit, activation='sigmoid'),
        ])

        discriminator.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=self.D_LR), loss='binary_crossentropy')

        combined_input = tf.keras.layers.Input(shape=(self.G_n_input,))
        generated = generator(combined_input)

        # split learning generator & discriminator -> competing
        discriminator.trainable = False  # discriminator(fixed)

        combined_output = discriminator(generated)

        combined_Model = tf.keras.models.Model(
            inputs=combined_input, outputs=combined_output)

        combined_Model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=self.combined_LR), loss='binary_crossentropy')

        n_iter = self.n_iter
        batch_size = self.n_batch

        fake = np.zeros(batch_size)
        real = np.ones(batch_size)

        for i in range(n_iter):

            # train discriminator
            # sampling from Gaussian distribution of latent codes
            noise = np.random.normal(0, 1, [self.n_batch, 100])
            generated_images = generator.predict(noise)

            idx = np.random.randint(0, train_x.shape[0], batch_size)
            real_images = train_x[idx]

            D_loss_real = discriminator.train_on_batch(real_images, real)
            D_loss_fake = discriminator.train_on_batch(generated_images, fake)
            D_loss = D_loss_real + D_loss_fake

            # train generator
            noise = np.random.normal(0, 1, [self.n_batch, 100])
            G_loss = combined_Model.train_on_batch(noise, real)

            if i % 5000 == 0:

                print('discriminator Loss: ', D_loss)
                print('generator Loss: ', G_loss)

                noise = np.random.normal(0, 1, [10, 100])

                pick_generated_images = generator.predict(noise)
                pick_generated_images = pick_generated_images.reshape(
                    10, 28, 28)

                plt.figure(figsize=(90, 10))

                for i in range(10):

                    plt.subplot(1, 10, i+1)
                    plt.imshow(pick_generated_images[i],
                               'gray', interpolation='nearest')
                    plt.axis('off')
                    plt.tight_layout()

                plt.show()


back_test = [GAN(100, 256, 784, 784, 256, 1, 0.0001, 0.0002, 1000, 100)]
# ----------------------------------------------------------------------------------------------------
# GAN (2)

generator = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=256, input_dim=100, activation='relu'),
    tf.keras.layers.Dense(units=784, activation='sigmoid')
])

discriminator = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=256, input_dim=784, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid'),
])

discriminator.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.0001), loss='binary_crossentropy')

combined_input = tf.keras.layers.Input(shape=(100,))
generated = generator(combined_input)

# split learning generator & discriminator -> competing
discriminator.trainable = False  # discriminator(fixed)

combined_output = discriminator(generated)

combined_Model = tf.keras.models.Model(
    inputs=combined_input, outputs=combined_output)

combined_Model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.0002), loss='binary_crossentropy')


def make_noise(samples):

    return np.random.normal(0, 1, [samples, 100])


def plot_generated_images(generator, samples=10):

    noise = make_noise(samples)

    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(samples, 28, 28)

    plt.figure(figsize=(90, 10))

    for i in range(samples):
        plt.subplot(1, samples, i+1)
        plt.imshow(generated_images[i], 'gray', interpolation='nearest')
        plt.axis('off')
        plt.tight_layout()

    plt.show()


n_iter = 1000
batch_size = 100

fake = np.zeros(batch_size)
real = np.ones(batch_size)

for i in range(n_iter):

    # train discriminator
    # sampling from Gaussian distribution of latent codes
    noise = make_noise(batch_size)
    generated_images = generator.predict(noise)

    idx = np.random.randint(0, train_x.shape[0], batch_size)
    real_images = train_x[idx]

    D_loss_real = discriminator.train_on_batch(real_images, real)
    D_loss_fake = discriminator.train_on_batch(generated_images, fake)
    D_loss = D_loss_real + D_loss_fake

    # train generator
    noise = make_noise(batch_size)
    G_loss = combined_Model.train_on_batch(noise, real)

    if i % 5000 == 0:

        print('discriminator Loss: ', D_loss)
        print('generator Loss: ', G_loss)

        plot_generated_images(generator)
# ====================================================================================================
