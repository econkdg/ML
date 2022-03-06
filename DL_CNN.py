# ====================================================================================================
# CNN
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

train_x, test_x = train_x/255.0, test_x/255.0

train_x = train_x.reshape((train_x.shape[0], 28, 28, 1))
test_x = test_x.reshape((test_x.shape[0], 28, 28, 1))
# ----------------------------------------------------------------------------------------------------
# CNN


class CNN:

    def __init__(self, idx, n_unit, n_batch, n_epoch, LR):

        self.idx = idx
        self.n_unit = n_unit
        self.n_epoch = n_epoch
        self.n_batch = n_batch
        self.LR = LR

    def CNN_TensorFlow(self):

        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=32,
                                   kernel_size=(3, 3),
                                   activation='relu',
                                   padding='SAME',
                                   input_shape=(28, 28, 1)),

            tf.keras.layers.MaxPool2D((2, 2)),

            tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(3, 3),
                                   activation='relu',
                                   padding='SAME',
                                   input_shape=(14, 14, 32)),

            tf.keras.layers.MaxPool2D((2, 2)),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(units=self.n_unit, activation='relu'),

            tf.keras.layers.Dense(units=10, activation='softmax')
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.LR),
                      loss='sparse_categorical_crossentropy',
                      metrics='accuracy')

        model.fit(train_x, train_y, batch_size=self.n_batch,
                  epochs=self.n_epoch)

        test_loss, test_acc = model.evaluate(test_x, test_y)

        test_img = test_x[[self.idx]]

        predict = model.predict(test_img)
        predict_value = np.argmax(predict, axis=1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(test_img.reshape(28, 28), 'gray')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.stem(predict[0])

        fig = plt.show()

        result = print('prediction : {}'.format(predict_value[0]))

        return fig, result


back_test = [CNN(1495, 128, 50, 3, 0.001),
             CNN(1300, 100, 50, 3, 0.001)]
# ====================================================================================================
