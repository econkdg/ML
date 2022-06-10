# ====================================================================================================
# ANN(artificial neural network) DNN
# ====================================================================================================
# import library

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
# ----------------------------------------------------------------------------------------------------
# generate training data(1)

m = 1000

x1 = 8*np.random.rand(m, 1)
x2 = 7*np.random.rand(m, 1) - 4

g = 0.8*x1 + x2 - 3

class_1 = np.where(g >= 0)[0]
class_0 = np.where(g < 0)[0]

N = class_1.shape[0]
M = class_0.shape[0]

m = N + M

X1 = np.hstack([np.ones([N, 1]), x1[class_1], x2[class_1]])
X0 = np.hstack([np.ones([M, 1]), x1[class_0], x2[class_0]])

train_X = np.vstack([X1, X0])
train_y = np.vstack([np.ones([N, 1]), -np.ones([M, 1])])

train_X = np.asmatrix(train_X)
train_y = np.asmatrix(train_y)

plt.figure(figsize=(10, 8))
plt.plot(x1[class_1], x2[class_1], 'ro', alpha=0.4, label='class 1')
plt.plot(x1[class_0], x2[class_0], 'bo', alpha=0.4, label='class 0')
plt.legend(loc=1, fontsize=15)
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)

plt.show()
# ----------------------------------------------------------------------------------------------------
# ANN logistic regression (1)


class ANN_logistic:

    def __init__(self, in_dim, num_unit, LR, num_epoch):

        self.in_dim = in_dim
        self.num_unit = num_unit
        self.LR = LR
        self.num_epoch = num_epoch

    def ANN_logistic_estimation(self):

        LogisticRegression = tf.keras.models.Sequential([
            tf.keras.layers.Dense(input_dim=self.in_dim,
                                  units=self.num_unit, activation='sigmoid')
        ])

        LogisticRegression.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=self.LR), loss='binary_crossentropy')

        loss = LogisticRegression.fit(train_X, train_y, epochs=self.num_epoch)

        w = LogisticRegression.layers[0].get_weights()[0]
        b = LogisticRegression.layers[0].get_weights()[1]

        return loss, w, b

    def ANN_logistic_plot(self):

        x1p = np.arange(np.min(train_X[:, 1]), np.max(
            train_X[:, 1]), 0.01).reshape(-1, 1)
        x2p = - self.ANN_logistic_estimation()[1][0, 0]/self.ANN_logistic_estimation(
        )[1][1, 0]*x1p - self.ANN_logistic_estimation()[2][0]/self.ANN_logistic_estimation()[1][1, 0]

        plt.figure(figsize=(10, 8))
        plt.plot(x1[class_1], x2[class_1], 'ro', alpha=0.4, label='class 1')
        plt.plot(x1[class_0], x2[class_0], 'bo', alpha=0.4, label='class 0')
        plt.plot(x1p, x2p, 'g', linewidth=3, label='')
        plt.xlim([np.min(train_X[:, 1]), np.max(train_X[:, 1])])
        plt.xlabel('$x_1$', fontsize=15)
        plt.ylabel('$x_2$', fontsize=15)
        plt.legend(loc=1, fontsize=12)

        fig = plt.show()

        return fig


testing_data = [ANN_logistic(2, 1, 0.01, 10)]
# ----------------------------------------------------------------------------------------------------
# generate training data(2)

m = 1000

x1 = 10*np.random.rand(m, 1) - 5
x2 = 8*np.random.rand(m, 1) - 4

g = - 0.5*(x1-1)**2 + 2*x2 + 5

class_1 = np.where(g >= 0)[0]
class_0 = np.where(g < 0)[0]

N = class_1.shape[0]
M = class_0.shape[0]

m = N + M

X1 = np.hstack([x1[class_1], x2[class_1]])
X0 = np.hstack([x1[class_0], x2[class_0]])

train_X = np.vstack([X1, X0])
train_y = np.vstack([np.ones([N, 1]), np.zeros([M, 1])])

train_X = np.asmatrix(train_X)
train_y = np.asmatrix(train_y)

plt.figure(figsize=(10, 8))
plt.plot(x1[class_1], x2[class_1], 'ro', alpha=0.4, label='class 1')
plt.plot(x1[class_0], x2[class_0], 'bo', alpha=0.4, label='class 0')
plt.legend(loc=1, fontsize=15)
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)

plt.show()
# ----------------------------------------------------------------------------------------------------
# ANN logistic regression (2)


class ANN_logistic:

    def __init__(self, in_dim, l1_num_unit, l2_num_unit, LR, num_epoch):

        self.in_dim = in_dim
        self.l1_num_unit = l1_num_unit
        self.l2_num_unit = l2_num_unit
        self.LR = LR
        self.num_epoch = num_epoch

    def ANN_logistic_estimation(self):

        LogisticRegression = tf.keras.models.Sequential([
            tf.keras.layers.Dense(input_dim=self.in_dim,
                                  units=self.l1_num_unit, activation='sigmoid'),
            tf.keras.layers.Dense(units=self.l2_num_unit, activation='sigmoid')
        ])

        LogisticRegression.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=self.LR), loss='binary_crossentropy')

        loss = LogisticRegression.fit(train_X, train_y, epochs=self.num_epoch)

        w1 = LogisticRegression.layers[0].get_weights()[0]
        b1 = LogisticRegression.layers[0].get_weights()[1]

        w2 = LogisticRegression.layers[1].get_weights()[0]
        b2 = LogisticRegression.layers[1].get_weights()[1]

        return loss, w1, b1, w2, b2

    def ANN_logistic_plot(self):

        H = train_X * \
            self.ANN_logistic_estimation()[1] + \
            self.ANN_logistic_estimation()[2]
        H = 1/(1 + np.exp(-H))

        x1p = np.arange(0, 1, 0.01).reshape(-1, 1)
        x2p = - self.ANN_logistic_estimation()[3][0, 0]/self.ANN_logistic_estimation(
        )[3][1, 0]*x1p - self.ANN_logistic_estimation()[4][0]/self.ANN_logistic_estimation()[3][1, 0]

        plt.figure(figsize=(10, 8))
        plt.plot(H[0:N, 0], H[0:N, 1], 'ro', alpha=0.4, label='class 1')
        plt.plot(H[N:m, 0], H[N:m, 1], 'bo', alpha=0.4, label='class 0')
        plt.plot(x1p, x2p, 'k', linewidth=3, label='')
        plt.xlabel('$z_1$', fontsize=15)
        plt.ylabel('$z_2$', fontsize=15)
        plt.legend(loc=1, fontsize=15)
        plt.axis('equal')
        plt.xlim([0, 1])
        plt.ylim([0, 1])

        fig1 = plt.show()

        x1p = np.arange(np.min(train_X[:, 0]), np.max(
            train_X[:, 0]), 0.01).reshape(-1, 1)
        x2p = - self.ANN_logistic_estimation()[1][0, 0]/self.ANN_logistic_estimation(
        )[1][1, 0]*x1p - self.ANN_logistic_estimation()[2][0]/self.ANN_logistic_estimation()[1][1, 0]
        x3p = - self.ANN_logistic_estimation()[1][0, 1]/self.ANN_logistic_estimation(
        )[1][1, 1]*x1p - self.ANN_logistic_estimation()[2][1]/self.ANN_logistic_estimation()[1][1, 1]

        plt.figure(figsize=(10, 8))
        plt.plot(x1[class_1], x2[class_1], 'ro', alpha=0.4, label='class 1')
        plt.plot(x1[class_0], x2[class_0], 'bo', alpha=0.4, label='class 0')
        plt.plot(x1p, x2p, 'k', linewidth=3, label='')
        plt.plot(x1p, x3p, 'g', linewidth=3, label='')
        plt.xlabel('$x_1$', fontsize=15)
        plt.ylabel('$x_2$', fontsize=15)
        plt.legend(loc=1, fontsize=15)
        plt.axis('equal')
        plt.xlim([np.min(train_X[:, 0]), np.max(train_X[:, 0])])
        plt.ylim([np.min(train_X[:, 0]), np.max(train_X[:, 0])])

        fig2 = plt.show()

        return fig1, fig2


testing_data = [ANN_logistic(2, 2, 1, 0.1, 10)]
# ====================================================================================================
