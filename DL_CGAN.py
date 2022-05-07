# ====================================================================================================
# CGAN: conditional general adversarial networks
# ====================================================================================================
# import library

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm_notebook
# ----------------------------------------------------------------------------------------------------
# import data (2)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0
x_train, x_test = x_train.reshape(-1, 784), x_test.reshape(-1, 784)
y_train, y_test = y_train.reshape(-1, 1), y_test.reshape(-1, 1)
# ----------------------------------------------------------------------------------------------------
# CGAN (1): class version


class CGAN:

    def __init__(self, D_LR, combined_LR, n_iter, n_batch):

        self.D_LR = D_LR
        self.combined_LR = combined_LR

        self.n_iter = n_iter
        self.n_batch = n_batch

    def CGAN_TensorFlow(self):

        # generator
        generator = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=256, input_dim=138, activation='relu'),
            tf.keras.layers.Dense(units=784, activation='sigmoid')
        ])

        noise = tf.keras.layers.Input(shape=(128,))
        label = tf.keras.layers.Input(shape=(1,))
        label_onehot = tf.keras.layers.CategoryEncoding(
            10, output_mode='one_hot')(label)

        model_input = tf.keras.layers.concatenate(
            [noise, label_onehot], axis=1)
        generated_image = generator(model_input)

        generator_model = tf.keras.models.Model(
            [noise, label], generated_image)

        # discriminator
        discriminator = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=256, input_dim=794, activation='relu'),
            tf.keras.layers.Dense(units=1, activation='sigmoid')
        ])

        input_image = tf.keras.layers.Input(shape=(784,))
        label = tf.keras.layers.Input(shape=(1,))
        label_onehot = tf.keras.layers.CategoryEncoding(
            10, output_mode='one_hot')(label)

        model_input = tf.keras.layers.concatenate(
            [input_image, label_onehot], axis=1)
        validity = discriminator(model_input)

        discriminator_model = tf.keras.models.Model(
            [input_image, label], validity)

        discriminator_model.compile(
            loss=['binary_crossentropy'], optimizer=tf.keras.optimizers.Adam(learning_rate=self.D_LR))

        noise = tf.keras.layers.Input(shape=(128,))
        label = tf.keras.layers.Input(shape=(1,))
        generated_image = generator_model([noise, label])

        # split learning generator & discriminator -> competing
        discriminator.trainable = False  # discriminator(fixed)

        validity = discriminator_model([generated_image, label])

        combined_model = tf.keras.models.Model([noise, label], validity)

        combined_model.compile(
            loss=['binary_crossentropy'], optimizer=tf.keras.optimizers.Adam(learning_rate=self.combined_LR))

        def create_noise(samples):

            return np.random.normal(0, 1, [samples, 128])

        def plot_generated_images(generator_model):

            noise = create_noise(10)
            label = np.arange(0, 10).reshape(-1, 1)

            generated_images = generator_model.predict([noise, label])

            plt.figure(figsize=(90, 10))

            for i in range(generated_images.shape[0]):

                plt.subplot(1, 10, i + 1)
                plt.imshow(generated_images[i].reshape(
                    (28, 28)), 'gray', interpolation='nearest')
                plt.title('digit: {}'.format(i), fontsize=15)
                plt.axis('off')

            plt.show()

        n_iter = self.n_iter
        batch_size = self.n_batch

        valid = np.ones(batch_size)
        fake = np.zeros(batch_size)

        for i in range(n_iter):

            # train discriminator
            # sampling from latent codes
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            real_images, labels = x_train[idx], y_train[idx]

            noise = create_noise(batch_size)
            generated_images = generator_model.predict([noise, labels])

            D_loss_real = discriminator_model.train_on_batch(
                [real_images, labels], valid)
            D_loss_fake = discriminator_model.train_on_batch(
                [generated_images, labels], fake)
            D_loss = D_loss_real + D_loss_fake

            # train generator
            noise = create_noise(batch_size)
            labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)

            G_loss = combined_model.train_on_batch([noise, labels], valid)

        if i % 5000 == 0:

            print('discriminator loss: ', D_loss)
            print('generator loss: ', G_loss)

            plot_generated_images(generator_model)


j = 0
for i in tqdm_notebook(range(10000000)):
    j += 1

back_test = [CGAN(0.0002, 0.0002, 1000, 100)]
# ----------------------------------------------------------------------------------------------------
# CGAN (2)

generator = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=256, input_dim=138, activation='relu'),
    tf.keras.layers.Dense(units=784, activation='sigmoid')
])

noise = tf.keras.layers.Input(shape=(128,))
label = tf.keras.layers.Input(shape=(1,))
label_onehot = tf.keras.layers.CategoryEncoding(
    10, output_mode='one_hot')(label)

model_input = tf.keras.layers.concatenate([noise, label_onehot], axis=1)
generated_image = generator(model_input)

generator_model = tf.keras.models.Model([noise, label], generated_image)

discriminator = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=256, input_dim=794, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

input_image = tf.keras.layers.Input(shape=(784,))
label = tf.keras.layers.Input(shape=(1,))
label_onehot = tf.keras.layers.CategoryEncoding(
    10, output_mode='one_hot')(label)

model_input = tf.keras.layers.concatenate([input_image, label_onehot], axis=1)
validity = discriminator(model_input)

discriminator_model = tf.keras.models.Model([input_image, label], validity)

discriminator_model.compile(
    loss=['binary_crossentropy'], optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002))

noise = tf.keras.layers.Input(shape=(128,))
label = tf.keras.layers.Input(shape=(1,))
generated_image = generator_model([noise, label])

# split learning generator & discriminator -> competing
discriminator.trainable = False  # discriminator(fixed)

validity = discriminator_model([generated_image, label])

combined_model = tf.keras.models.Model([noise, label], validity)

combined_model.compile(loss=['binary_crossentropy'],
                       optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002))


def create_noise(samples):

    return np.random.normal(0, 1, [samples, 128])


def plot_generated_images(generator_model):

    noise = create_noise(10)
    label = np.arange(0, 10).reshape(-1, 1)

    generated_images = generator_model.predict([noise, label])

    plt.figure(figsize=(90, 10))

    for i in range(generated_images.shape[0]):

        plt.subplot(1, 10, i + 1)
        plt.imshow(generated_images[i].reshape(
            (28, 28)), 'gray', interpolation='nearest')
        plt.title('digit: {}'.format(i), fontsize=15)
        plt.axis('off')

    plt.show()


n_iter = 100000
batch_size = 100

valid = np.ones(batch_size)
fake = np.zeros(batch_size)

for i in range(n_iter):

    # train discriminator
    idx = np.random.randint(0, x_train.shape[0], batch_size)
    real_images, labels = x_train[idx], y_train[idx]

    noise = create_noise(batch_size)
    generated_images = generator_model.predict([noise, labels])

    D_loss_real = discriminator_model.train_on_batch(
        [real_images, labels], valid)
    D_loss_fake = discriminator_model.train_on_batch(
        [generated_images, labels], fake)
    D_loss = D_loss_real + D_loss_fake

    # train generator
    # sampling from latent codes
    noise = create_noise(batch_size)
    labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)

    G_loss = combined_model.train_on_batch([noise, labels], valid)

    if i % 5000 == 0:

        print('discriminator loss: ', D_loss)
        print('generator loss: ', G_loss)

        plot_generated_images(generator_model)

j = 0
for i in tqdm_notebook(range(10000000)):
    j += 1
# ====================================================================================================
