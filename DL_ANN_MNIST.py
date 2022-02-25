# ====================================================================================================
# ANN MNIST
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

train_x.shape
train_y.shape

test_x.shape
test_y.shape
# ----------------------------------------------------------------------------------------------------
# ANN with TensorFlow


class ANN:

    def __init__(self, idx, l1_n_unit, l2_n_unit, n_batch, n_epoch, LR):

        self.idx = idx
        self.l1_n_unit = l1_n_unit
        self.l2_n_unit = l2_n_unit
        self.n_batch = n_batch
        self.n_epoch = n_epoch
        self.LR = LR

    def img_output(self):

        test_img = np.reshape(test_x[self.idx], (28, 28))
        # img = test_x[5].reshape(28, 28)

        plt.figure(figsize=(6, 6))
        plt.imshow(test_img.reshape(28, 28), 'gray')
        plt.xticks([])
        plt.yticks([])

        fig1 = plt.show()

        return fig1

    def ANN_TensorFlow(self):

        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(units=self.l1_n_unit, activation='relu'),
            tf.keras.layers.Dense(units=self.l2_n_unit, activation='softmax')
        ])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.LR), loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        # train model
        loss = model.fit(train_x, train_y,
                         batch_size=self.n_batch, epochs=self.n_epoch)

        # evaluate test data(out-of-sample)
        test_loss, test_acc = model.evaluate(test_x, test_y)

        test_img = np.expand_dims(test_x[self.idx], axis=0)
        # test_img = test_x[np.random.choice(test_x.shape[0], 1)]

        predict = model.predict_on_batch(test_img)
        predict_value = np.argmax(predict, axis=1)

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(test_img.reshape(28, 28), 'gray')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.stem(predict[0])

        fig2 = plt.show()

        result = print('prediction : {}'.format(predict_value[0]))

        return fig2, result


back_test = [ANN(1597, 100, 10, 50, 10, 0.01),
             ANN(2362, 100, 10, 50, 10, 0.05),
             ANN(4963, 100, 10, 50, 10, 0.1)]
# ====================================================================================================
