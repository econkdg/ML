# ====================================================================================================
# TF(tensor flow)
# ====================================================================================================
# import library

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# ----------------------------------------------------------------------------------------------------
# tensor flow

# tensor flow: open-source software library for deep learning

# tensor -> matrix
# flow -> flow through layer

# tf.constant -> constant

# tf.Variable -> decision variable
# updated using numerical iteration
# initialization

# tf.placeholder -> space for training data

# session
# high computation -> efficient

# example
a = tf.constant([1, 2, 3])
b = tf.constant(4, shape=[1, 3])

A = a + b
B = a*b
# ----------------------------------------------------------------------------------------------------
# optimization solver


def tf_optm_sol(init, LR, iter):

    w = tf.Variable(init, dtype=tf.float32)  # initial value = 0
    cost = w*w - 8*w + 16

    cost_record = []

    for i in range(iter):

        with tf.GradientTape() as tape:  # input -> loss (forward pass)

            cost = w*w - 8*w + 16
            w_grad = tape.gradient(cost, w)  # loss -> input (backward pass)

        cost_record.append(cost)

        w.assign_sub(LR * w_grad)

        print("\n cost =", cost.numpy())
        print("\n w hat =", w.numpy())

    cost_val = cost.numpy()
    optimal_w_val = w.numpy()

    cost_result = print("\n cost =", cost_val)
    optimal_w_result = print("\n optimal w =", optimal_w_val)

    plt.figure(figsize=(10, 8))
    plt.plot(cost_record)
    plt.xlabel('iteration', fontsize=15)
    plt.ylabel('cost', fontsize=15)

    fig = plt.show()

    return cost_result, optimal_w_result, fig
# ----------------------------------------------------------------------------------------------------
# linear regression


# generate random data
n = 200

train_x = np.random.randn(n, 1)
train_noise = np.random.randn(n, 1)

train_y = 2 + 0.70*train_x + train_noise

m = train_x.shape[0]

# tensor flow


def linear_reg(init, LR, iter):

    w1 = tf.Variable([[init]], dtype=tf.float32)
    w0 = tf.Variable([[init]], dtype=tf.float32)

    loss_record = []

    for i in range(iter):

        with tf.GradientTape() as tape:

            cost = tf.reduce_mean(tf.square(w0 + w1*train_x - train_y))
            w0_grad, w1_grad = tape.gradient(cost, [w0, w1])

        loss_record.append(cost)

        w0.assign_sub(LR * w0_grad)
        w1.assign_sub(LR * w1_grad)

        print("\n w0 hat =", w0.numpy())
        print("\n w1 hat =", w1.numpy())

    optimal_w0_val = w0.numpy()
    optimal_w1_val = w1.numpy()

    optimal_w0_result = print("\n optimal w0 =", optimal_w0_val)
    optimal_w1_result = print("\n optimal w1 =", optimal_w1_val)

    plt.figure(figsize=(10, 8))
    plt.plot(loss_record)
    plt.xlabel('iteration', fontsize=15)
    plt.ylabel('loss', fontsize=15)

    fig1 = plt.show()

    xp = np.arange(np.min(train_x), np.max(train_x), 0.01).reshape(-1, 1)
    yp = optimal_w0_val + optimal_w1_val*xp

    plt.figure(figsize=(10, 8))
    plt.plot(train_x, train_y, 'ko')
    plt.plot(xp, yp, 'r')
    plt.title('data', fontsize=15)
    plt.xlabel('X', fontsize=15)
    plt.ylabel('Y', fontsize=15)
    plt.axis('equal')
    plt.grid(alpha=0.3)
    plt.xlim([np.min(train_x), np.max(train_x)])

    fig2 = plt.show()

    return optimal_w0_result, optimal_w1_result, fig1, fig2
# ----------------------------------------------------------------------------------------------------
# logistic regression


# generate random data
m = 200

true_w = np.array([[-6], [2], [1]])
train_X = np.hstack(
    [np.ones([m, 1]), 5*np.random.rand(m, 1), 4*np.random.rand(m, 1)])

true_w = np.asmatrix(true_w)
train_X = np.asmatrix(train_X)

train_y = 1/(1 + np.exp(-train_X*true_w)) > 0.5

class_1 = np.where(train_y == True)[0]
class_0 = np.where(train_y == False)[0]

# tensor flow


def logistic_reg(LR, iter):

    train_y = np.empty([m, 1])
    train_y[class_1] = 1
    train_y[class_0] = 0

    w = tf.Variable([[0], [0], [0]], dtype=tf.float32)

    train_x = tf.constant(train_X, dtype=tf.float32)
    train_y = tf.constant(train_y, dtype=tf.float32)

    loss_record = []

    for i in range(iter):

        with tf.GradientTape() as tape:

            y_pred = tf.sigmoid(tf.matmul(train_x, w))

            loss = - train_y*tf.math.log(y_pred) - \
                (1-train_y)*tf.math.log(1-y_pred)
            loss = tf.reduce_mean(loss)

            w_grad = tape.gradient(loss, w)

        loss_record.append(loss)
        w.assign_sub(LR * w_grad)

        print("\n w hat =", w.numpy())

    optimal_w_value = w.numpy()

    optimal_w_result = print("\n optimal w =", optimal_w_value)

    plt.figure(figsize=(10, 8))
    plt.plot(loss_record)
    plt.xlabel('iteration', fontsize=15)
    plt.ylabel('loss', fontsize=15)

    fig1 = plt.show()

    xp = np.arange(np.min(train_X[:, 1]), np.max(
        train_X[:, 1]), 0.01).reshape(-1, 1)
    yp = - optimal_w_value[1, 0]/optimal_w_value[2, 0] * \
        xp - optimal_w_value[0, 0]/optimal_w_value[2, 0]

    plt.figure(figsize=(10, 8))
    plt.plot(train_X[class_1, 1], train_X[class_1, 2],
             'ro', alpha=0.3, label='class 1')
    plt.plot(train_X[class_0, 1], train_X[class_0, 2],
             'bo', alpha=0.3, label='class 0')
    plt.plot(xp, yp, 'g', linewidth=3, label='logistic regression')
    plt.xlabel(r'$x_1$', fontsize=15)
    plt.ylabel(r'$x_2$', fontsize=15)
    plt.legend(loc=1, fontsize=12)
    plt.axis('equal')
    plt.xlim([np.min(train_X[:, 1]), np.max(train_X[:, 1])])
    plt.ylim([np.min(train_X[:, 2]), np.max(train_X[:, 2])])

    fig2 = plt.show()

    return optimal_w_result, fig1, fig2
# ====================================================================================================
