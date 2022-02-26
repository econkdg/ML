# ====================================================================================================
# autoencoder
# ====================================================================================================
# import library

from calendar import EPOCH
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# ----------------------------------------------------------------------------------------------------
# generate MNIST data set

mnist = tf.keras.datasets.mnist

(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x = train_x.reshape((train_x.shape[0], 28*28)) / 255
test_x = test_x.reshape((test_x.shape[0], 28*28)) / 255
# ----------------------------------------------------------------------------------------------------
# use only 1, 6, 9 digits to visualize

train_idx1 = np.array(np.where(train_y == 1))
train_idx6 = np.array(np.where(train_y == 6))
train_idx9 = np.array(np.where(train_y == 9))
train_idx = np.sort(np.concatenate(
    (train_idx1, train_idx6, train_idx9), axis=None))

test_idx1 = np.array(np.where(test_y == 1))
test_idx6 = np.array(np.where(test_y == 6))
test_idx9 = np.array(np.where(test_y == 9))
test_idx = np.sort(np.concatenate(
    (test_idx1, test_idx6, test_idx9), axis=None))

train_imgs = train_x[train_idx]
train_labels = train_y[train_idx]

test_imgs = test_x[test_idx]
test_labels = test_y[test_idx]

n_train = train_imgs.shape[0]
n_test = test_imgs.shape[0]

print("The number of training images : {}, shape : {}".format(
    n_train, train_imgs.shape))
print("The number of testing images : {}, shape : {}".format(n_test, test_imgs.shape))
# ----------------------------------------------------------------------------------------------------
# autoencoder


class AUTOENCODER:

    def __init__(self, idx, new_z1, new_z2, en_l1_n_unit, en_l2_n_unit, latent_dim, de_l1_n_unit, de_l2_n_unit, n_batch, n_epoch, LR):

        self.idx = idx
        self.new_z1 = new_z1
        self.new_z2 = new_z2
        self.en_l1_n_unit = en_l1_n_unit
        self.en_l2_n_unit = en_l2_n_unit
        self.latent_dim = latent_dim
        self.de_l1_n_unit = de_l1_n_unit
        self.de_l2_n_unit = de_l2_n_unit
        self.n_batch = n_batch
        self.n_epoch = n_epoch
        self.LR = LR

    def encoder(self):

        encoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.en_l1_n_unit, activation='relu',
                                  input_shape=(28*28,)),
            tf.keras.layers.Dense(self.en_l2_n_unit, activation='relu'),
            tf.keras.layers.Dense(self.latent_dim, activation=None)
        ])

        return encoder

    def decoder(self):

        decoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(
                self.de_l1_n_unit, activation='relu', input_shape=(2,)),
            tf.keras.layers.Dense(self.de_l2_n_unit, activation='relu'),
            tf.keras.layers.Dense(28*28, activation=None)
        ])

        return decoder

    def autoencoder(self):

        autoencoder = tf.keras.models.Sequential(
            [self.encoder(), self.decoder()])

        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(self.LR),
                            loss='mean_squared_error',
                            metrics=['mse'])

        training = autoencoder.fit(
            train_imgs, train_imgs, batch_size=self.n_batch, epochs=self.n_epoch)

        test_scores = autoencoder.evaluate(test_imgs, test_imgs, verbose=2)

        print('test loss: {}'.format(test_scores[0]))
        print('mean squared error: {} %'.format(test_scores[1]*100))

        return autoencoder

    def autoencoder_plot_1(self):

        # rand_idx = np.random.randint(1, test_imgs.shape[0])
        # test_img = test_imgs[rand_idx]

        test_img = test_imgs[self.idx]

        reconst_img = self.autoencoder().predict(test_img.reshape(1, 28*28))

        plt.figure(figsize=(10, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(test_img.reshape(28, 28), 'gray')
        plt.title('input image', fontsize=12)
        plt.xticks([])
        plt.yticks([])
        plt.subplot(1, 2, 2)
        plt.imshow(reconst_img.reshape(28, 28), 'gray')
        plt.title('reconstructed image', fontsize=12)
        plt.xticks([])
        plt.yticks([])

        fig1 = plt.show()

    def autoencoder_plot_2(self):

        encoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(self.en_l1_n_unit, activation='relu',
                                  input_shape=(28*28,)),
            tf.keras.layers.Dense(self.en_l2_n_unit, activation='relu'),
            tf.keras.layers.Dense(self.latent_dim, activation=None)
        ])

        decoder = tf.keras.models.Sequential([
            tf.keras.layers.Dense(
                self.de_l1_n_unit, activation='relu', input_shape=(2,)),
            tf.keras.layers.Dense(self.de_l2_n_unit, activation='relu'),
            tf.keras.layers.Dense(28*28, activation=None)
        ])

        autoencoder = tf.keras.models.Sequential(
            [encoder, decoder])

        autoencoder.compile(optimizer=tf.keras.optimizers.Adam(self.LR),
                            loss='mean_squared_error',
                            metrics=['mse'])

        training = autoencoder.fit(
            train_imgs, train_imgs, batch_size=self.n_batch, epochs=self.n_epoch)

        test_scores = autoencoder.evaluate(test_imgs, test_imgs, verbose=2)

        print('test loss: {}'.format(test_scores[0]))
        print('mean squared error: {} %'.format(test_scores[1]*100))

        idx_set = np.random.randint(0, len(test_labels), 500)

        test_X, test_Y = test_imgs[idx_set], test_labels[idx_set]

        test_latent = encoder.predict(test_X)

        plt.figure(figsize=(10, 10))
        plt.scatter(test_latent[test_Y == 1, 0],
                    test_latent[test_Y == 1, 1], label='1')
        plt.scatter(test_latent[test_Y == 6, 0],
                    test_latent[test_Y == 6, 1], label='6')
        plt.scatter(test_latent[test_Y == 9, 0],
                    test_latent[test_Y == 9, 1], label='9')
        plt.title('latent space', fontsize=15)
        plt.xlabel('Z1', fontsize=15)
        plt.ylabel('Z2', fontsize=15)
        plt.legend(fontsize=15)
        plt.axis('equal')

        fig2 = plt.show()

        new_data = np.array([[self.new_z1, self.new_z2]])

        fake_image = decoder.predict(new_data)

        plt.figure(figsize=(16, 7))
        plt.subplot(1, 2, 1)
        plt.scatter(test_latent[test_Y == 1, 0],
                    test_latent[test_Y == 1, 1], label='1')
        plt.scatter(test_latent[test_Y == 6, 0],
                    test_latent[test_Y == 6, 1], label='6')
        plt.scatter(test_latent[test_Y == 9, 0],
                    test_latent[test_Y == 9, 1], label='9')
        plt.scatter(new_data[:, 0], new_data[:, 1], c='k',
                    marker='o', s=200, label='new data')
        plt.title('latent space', fontsize=15)
        plt.xlabel('Z1', fontsize=15)
        plt.ylabel('Z2', fontsize=15)
        plt.legend(loc=2, fontsize=12)
        plt.axis('equal')
        plt.subplot(1, 2, 2)
        plt.imshow(fake_image.reshape(28, 28), 'gray')
        plt.title('generated fake image', fontsize=15)
        plt.xticks([])
        plt.yticks([])

        fig3 = plt.show()

        return fig2, fig3


back_test = [AUTOENCODER(50, 0, 0, 500, 300, 2, 300, 500, 50, 10, 0.001)]
# ====================================================================================================
