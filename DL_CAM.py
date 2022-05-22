# ====================================================================================================
# CAM
# ====================================================================================================
# import library

from calendar import EPOCH
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
# ----------------------------------------------------------------------------------------------------
# generate MNIST data set

mnist = tf.keras.datasets.mnist

(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x, test_x = train_x/255.0, test_x/255.0

train_x = train_x.reshape((train_x.shape[0], 28, 28, 1))
test_x = test_x.reshape((test_x.shape[0], 28, 28, 1))
# ----------------------------------------------------------------------------------------------------
# CAM


class CAM:

    def __init__(self, idx, n_batch, n_epoch, LR):

        self.idx = idx
        self.n_epoch = n_epoch
        self.n_batch = n_batch
        self.LR = LR

    def CAM_TensorFlow(self):

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

            tf.keras.layers.Conv2D(filters=64,
                                   kernel_size=(3, 3),
                                   activation='relu',
                                   padding='SAME',
                                   input_shape=(7, 7, 64)),

            tf.keras.layers.GlobalAveragePooling2D(),

            tf.keras.layers.Dense(units=10, activation='softmax')
        ])

        model.summary()

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.LR),
                      loss='sparse_categorical_crossentropy',
                      metrics='accuracy')

        model.fit(train_x, train_y, batch_size=self.n_batch,
                  epochs=self.n_epoch)

        # accuracy test
        test_loss, test_acc = model.evaluate(test_x, test_y)

        # get global average pooling layer and fully connected layer
        conv_layer = model.get_layer('conv2d_2')
        fc_layer = model.layers[-1].get_weights()[0]

        # CAM(class activation map)
        activation_map = tf.matmul(conv_layer.output, fc_layer)
        CAM_model = tf.keras.Model(inputs=model.inputs, outputs=activation_map)

        test_img = test_x[[self.idx]]

        pred = np.argmax(model.predict(test_img), axis=1)
        predCAM = CAM_model.predict(test_img)

        attention = predCAM[:, :, :, pred]
        attention = np.abs(np.reshape(attention, (7, 7)))

        resized_test_x = cv2.resize(test_img.reshape(
            28, 28), (28*5, 28*5), interpolation=cv2.INTER_CUBIC)

        resized_attention = cv2.resize(
            attention, (28*5, 28*5), interpolation=cv2.INTER_CUBIC)

        plt.figure(figsize=(10, 15))
        plt.subplot(3, 2, 1)
        plt.imshow(test_img.reshape(28, 28), 'gray')
        plt.axis('off')

        plt.subplot(3, 2, 2)
        plt.imshow(attention)
        plt.axis('off')

        plt.subplot(3, 2, 3)
        plt.imshow(resized_test_x, 'gray')
        plt.axis('off')

        plt.subplot(3, 2, 4)
        plt.imshow(resized_attention, 'jet', alpha=0.5)
        plt.axis('off')

        plt.subplot(3, 2, 5)
        plt.imshow(resized_test_x, 'gray')
        plt.imshow(resized_attention, 'jet', alpha=0.5)
        plt.axis('off')

        fig = plt.show()

        result = print('prediction : {}'.format(pred[0]))

        return fig, result


back_test = [CAM(1495, 50, 3, 0.001),
             CAM(1300, 50, 5, 0.001)]
# ====================================================================================================
