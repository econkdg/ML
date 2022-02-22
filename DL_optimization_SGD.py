# ====================================================================================================
# DL_optimization SGD
# ====================================================================================================
# batch gradient descent(=gradient descent)
# stochastic gradient descent
# mini-batch gradient descent

# step size(=learning rate)
# ----------------------------------------------------------------------------------------------------
# import library

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
# ----------------------------------------------------------------------------------------------------
# generate training data

m = 10000

true_w = np.array([[-6], [2], [1]])
train_X = np.hstack(
    [np.ones([m, 1]), 5*np.random.rand(m, 1), 4*np.random.rand(m, 1)])

true_w = np.asmatrix(true_w)
train_X = np.asmatrix(train_X)

train_Y = 1/(1 + np.exp(-train_X*true_w)) > 0.5

class_1 = np.where(train_Y == True)[0]
class_0 = np.where(train_Y == False)[0]

train_Y = np.empty([m, 1])
train_Y[class_1] = 1
train_Y[class_0] = 0

plt.figure(figsize=(10, 8))
plt.plot(train_X[class_1, 1], train_X[class_1, 2],
         'r.', alpha=0.3, label='class 1')
plt.plot(train_X[class_0, 1], train_X[class_0, 2],
         'b.', alpha=0.3, label='class 0')
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)
plt.legend(loc=1, fontsize=12)
plt.axis('equal')
plt.xlim([np.min(train_X[:, 1]), np.max(train_X[:, 1])])
plt.ylim([np.min(train_X[:, 2]), np.max(train_X[:, 2])])

plt.show()
# ----------------------------------------------------------------------------------------------------
# batch gradient descent

# matmul: matrix multiplication


class batch_GD:

    def __init__(self, v1, v2, LR, n_iter):

        self.v1 = v1
        self.v2 = v2
        self.LR = LR
        self.n_iter = n_iter

    def batch_GD_estimation(self):

        w = tf.Variable([[0], [0], [0]], dtype=tf.float32)

        train_x = tf.constant(train_X, dtype=tf.float32)
        train_y = tf.constant(train_Y, dtype=tf.float32)

        start_time = time.time()

        loss_record = []

        for i in range(self.n_iter):

            with tf.GradientTape() as tape:

                y_pred = tf.sigmoid(tf.matmul(train_x, w))

                loss = - train_y * \
                    tf.math.log(y_pred) - (1-train_y)*tf.math.log(1-y_pred)
                loss = tf.reduce_mean(loss)

                w_grad = tape.gradient(loss, w)

            loss_record.append(loss)
            w.assign_sub(self.LR * w_grad)

            print("\n w hat =", w.numpy())

        training_time = time.time() - start_time

        optimal_w_value = w.numpy()

        optimal_w_result = print("\n optimal w =", optimal_w_value)

        V = [1, self.v1, self.v2]
        V = np.asmatrix(V)

        if 1/(1 + np.exp(-V[0, :]*optimal_w_value)) > 0.5:
            v_predict = 1

        else:
            v_predict = 0

        predict = print("\n class(v): ", v_predict)

        return loss_record, optimal_w_value, predict

    def batch_GD_plot_1(self):

        plt.figure(figsize=(10, 8))
        plt.plot(self.batch_GD_estimation()[0])
        plt.xlabel('iteration', fontsize=15)
        plt.ylabel('loss', fontsize=15)

        fig1 = plt.show()

        return fig1

    def batch_GD_plot_2(self):

        xp = np.arange(np.min(train_X[:, 1]), np.max(
            train_X[:, 1]), 0.01).reshape(-1, 1)
        yp = - self.batch_GD_estimation()[1][1, 0]/self.batch_GD_estimation(
        )[1][2, 0] * xp - self.batch_GD_estimation()[1][0, 0]/self.batch_GD_estimation()[1][2, 0]

        plt.figure(figsize=(10, 8))
        plt.plot(self.v1, self.v2, 'ko', label='testing data')
        plt.plot(train_X[class_1, 1], train_X[class_1, 2],
                 'ro', alpha=0.3, label='class 1')
        plt.plot(train_X[class_0, 1], train_X[class_0, 2],
                 'bo', alpha=0.3, label='class 0')
        plt.plot(xp, yp, 'g', linewidth=3,
                 label='logistic regression(batch GD)')
        plt.xlabel(r'$x_1$', fontsize=15)
        plt.ylabel(r'$x_2$', fontsize=15)
        plt.legend(loc=1, fontsize=12)
        plt.axis('equal')
        plt.xlim([np.min(train_X[:, 1]), np.max(train_X[:, 1])])
        plt.ylim([np.min(train_X[:, 2]), np.max(train_X[:, 2])])

        fig2 = plt.show()

        return fig2


testing_data = [batch_GD(2, 3, 0.05, 10000)]

# ----------------------------------------------------------------------------------------------------
# SGD(stochastic gradient descent)


class SGD:

    def __init__(self, v1, v2, LR, n_iter):

        self.v1 = v1
        self.v2 = v2
        self.LR = LR
        self.n_iter = n_iter

    def SGD_estimation(self):

        w = tf.Variable([[0], [0], [0]], dtype=tf.float32)

        train_x = tf.constant(train_X, dtype=tf.float32)
        train_y = tf.constant(train_Y, dtype=tf.float32)

        start_time = time.time()

        loss_record = []

        for i in range(self.n_iter):

            with tf.GradientTape() as tape:

                idx = np.random.choice(m)

                batch_x = np.expand_dims(train_x[idx, :], axis=0)
                batch_y = np.expand_dims(train_y[idx], axis=0)

                y_pred = tf.sigmoid(tf.matmul(batch_x, w))

                loss = - batch_y * \
                    tf.math.log(y_pred) - (1-batch_y)*tf.math.log(1-y_pred)
                loss = tf.reduce_mean(loss)

                w_grad = tape.gradient(loss, w)

            loss_record.append(loss)
            w.assign_sub(self.LR * w_grad)

            print("\n w hat =", w.numpy())

        training_time = time.time() - start_time

        optimal_w_value = w.numpy()

        optimal_w_result = print("\n optimal w =", optimal_w_value)

        V = [1, self.v1, self.v2]
        V = np.asmatrix(V)

        if 1/(1 + np.exp(-V[0, :]*optimal_w_value)) > 0.5:
            v_predict = 1

        else:
            v_predict = 0

        predict = print("\n class(v): ", v_predict)

        return loss_record, optimal_w_value, predict

    def SGD_plot_1(self):

        plt.figure(figsize=(10, 8))
        plt.plot(self.SGD_estimation()[0])
        plt.xlabel('iteration', fontsize=15)
        plt.ylabel('loss', fontsize=15)

        fig1 = plt.show()

        return fig1

    def SGD_plot_2(self):

        xp = np.arange(np.min(train_X[:, 1]), np.max(
            train_X[:, 1]), 0.01).reshape(-1, 1)
        yp = - self.SGD_estimation()[1][1, 0]/self.SGD_estimation(
        )[1][2, 0] * xp - self.SGD_estimation()[1][0, 0]/self.SGD_estimation()[1][2, 0]

        plt.figure(figsize=(10, 8))
        plt.plot(self.v1, self.v2, 'ko', label='testing data')
        plt.plot(train_X[class_1, 1], train_X[class_1, 2],
                 'ro', alpha=0.3, label='class 1')
        plt.plot(train_X[class_0, 1], train_X[class_0, 2],
                 'bo', alpha=0.3, label='class 0')
        plt.plot(xp, yp, 'g', linewidth=3, label='logistic regression(SGD)')
        plt.xlabel(r'$x_1$', fontsize=15)
        plt.ylabel(r'$x_2$', fontsize=15)
        plt.legend(loc=1, fontsize=12)
        plt.axis('equal')
        plt.xlim([np.min(train_X[:, 1]), np.max(train_X[:, 1])])
        plt.ylim([np.min(train_X[:, 2]), np.max(train_X[:, 2])])

        fig2 = plt.show()

        return fig2


testing_data = [SGD(2, 3, 0.05, 10000)]
# ----------------------------------------------------------------------------------------------------
# mini-batch gradient descent


class mini_batch_GD:

    def __init__(self, v1, v2, LR, n_iter, n_batch):

        self.v1 = v1
        self.v2 = v2
        self.LR = LR
        self.n_iter = n_iter
        self.n_batch = n_batch

    def mini_batch_GD_estimation(self):

        w = tf.Variable([[0], [0], [0]], dtype=tf.float32)

        loss_record = []

        for i in range(self.n_iter):

            with tf.GradientTape() as tape:

                idx = np.random.choice(m, size=self.n_batch)

                batch_x = tf.constant(train_X[idx, :], dtype=tf.float32)
                batch_y = tf.constant(train_Y[idx], dtype=tf.float32)

                y_pred = tf.sigmoid(tf.matmul(batch_x, w))

                loss = - batch_y * \
                    tf.math.log(y_pred) - (1-batch_y)*tf.math.log(1-y_pred)
                loss = tf.reduce_mean(loss)

                w_grad = tape.gradient(loss, w)

            loss_record.append(loss)
            w.assign_sub(self.LR * w_grad)

            print("\n w hat =", w.numpy())

        optimal_w_value = w.numpy()

        optimal_w_result = print("\n optimal w =", optimal_w_value)

        V = [1, self.v1, self.v2]
        V = np.asmatrix(V)

        if 1/(1 + np.exp(-V[0, :]*optimal_w_value)) > 0.5:
            v_predict = 1

        else:
            v_predict = 0

        predict = print("\n class(v): ", v_predict)

        return loss_record, optimal_w_value, predict

    def mini_batch_GD_plot_1(self):

        plt.figure(figsize=(10, 8))
        plt.plot(self.mini_batch_GD_estimation()[0])
        plt.xlabel('iteration', fontsize=15)
        plt.ylabel('loss', fontsize=15)

        fig1 = plt.show()

        return fig1

    def mini_batch_GD_plot_2(self):

        xp = np.arange(np.min(train_X[:, 1]), np.max(
            train_X[:, 1]), 0.01).reshape(-1, 1)
        yp = - self.mini_batch_GD_estimation()[1][1, 0]/self.mini_batch_GD_estimation(
        )[1][2, 0] * xp - self.mini_batch_GD_estimation()[1][0, 0]/self.mini_batch_GD_estimation()[1][2, 0]

        plt.figure(figsize=(10, 8))
        plt.plot(self.v1, self.v2, 'ko', label='testing data')
        plt.plot(train_X[class_1, 1], train_X[class_1, 2],
                 'ro', alpha=0.3, label='class 1')
        plt.plot(train_X[class_0, 1], train_X[class_0, 2],
                 'bo', alpha=0.3, label='class 0')
        plt.plot(xp, yp, 'g', linewidth=3,
                 label='logistic regression(mini-batch GD)')
        plt.xlabel(r'$x_1$', fontsize=15)
        plt.ylabel(r'$x_2$', fontsize=15)
        plt.legend(loc=1, fontsize=12)
        plt.axis('equal')
        plt.xlim([np.min(train_X[:, 1]), np.max(train_X[:, 1])])
        plt.ylim([np.min(train_X[:, 2]), np.max(train_X[:, 2])])

        fig2 = plt.show()

        return fig2


testing_data = [mini_batch_GD(2, 3, 0.05, 10000, 50)]
# ====================================================================================================
