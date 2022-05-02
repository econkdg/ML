# ====================================================================================================
# LSTM: long short-term memory
# ====================================================================================================
# import library

import tensorflow as tf
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import numpy as np
from six.moves import cPickle
# ----------------------------------------------------------------------------------------------------
# import data(1)

# acceleration signal of rotation machinery data
data = pd.read_pickle('rnn_time_signal.pkl')

plt.figure(figsize=(10, 6))
plt.title('time signal for RNN', fontsize=15)
plt.plot(data[0:2000])
plt.xlim(0, 2000)

plt.show()
# ----------------------------------------------------------------------------------------------------
# LSTM(1)


class LSTM_1:

    def __init__(self, n_samples, n_step, n_input, n_lstm1, n_lstm2, n_hidden, n_output, epoch):

        self.n_samples = n_samples

        self.n_step = n_step
        self.n_input = n_input

        # LSTM shape
        self.n_lstm1 = n_lstm1
        self.n_lstm2 = n_lstm2

        # fully connected
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.epoch = epoch

    def dataset(self):

        train_x_list = []
        train_y_list = []

        stride = 5

        for i in range(self.n_samples):

            train_x = data[i*stride:i*stride + self.n_step*self.n_input]
            train_x = train_x.reshape(self.n_step, self.n_input)
            train_x_list.append(train_x)

            train_y = data[i*stride + self.n_step*self.n_input:i *
                           stride + self.n_step*self.n_input + self.n_output]
            train_y_list.append(train_y)

        train_data = np.array(train_x_list)
        train_label = np.array(train_y_list)

        test_data = data[10000:10000 + self.n_step*self.n_input]
        test_data = test_data.reshape(1, self.n_step, self.n_input)

        return train_data, train_label, test_data

    def LSTM_TensorFlow(self):

        lstm_network = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(self.n_step, self.n_input)),
            tf.keras.layers.LSTM(self.n_lstm1, return_sequences=True),
            tf.keras.layers.LSTM(self.n_lstm2),
            tf.keras.layers.Dense(self.n_hidden),
            tf.keras.layers.Dense(self.n_output),
        ])

        lstm_network.summary()

        lstm_network.compile(
            optimizer='adam', loss='mean_squared_error', metrics=['mse'])

        lstm_network.fit(self.dataset()[0], self.dataset()[
                         1], epochs=self.epoch)

        test_pred = lstm_network.predict(self.dataset()[2]).ravel()
        test_label = data[10000:10000 +
                          self.n_step*self.n_input + self.n_input]

        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(0, self.n_step*self.n_input + self.n_input),
                 test_label, 'b', label='ground truth')
        plt.plot(np.arange(self.n_step*self.n_input, self.n_step *
                 self.n_input + self.n_input), test_pred, 'r', label='prediction')
        plt.vlines(self.n_step*self.n_input, -1, 1,
                   colors='r', linestyles='dashed')
        plt.legend(fontsize=15, loc='upper left')
        plt.xlim(0, len(test_label))

        fig1 = plt.show()

        return fig1


back_test = [LSTM_1(5000, 25, 100, 100, 100, 100, 100, 3)]


class LSTM_2:

    def __init__(self, n_samples, n_step, n_input, n_lstm1, n_lstm2, n_hidden, n_output, epoch):

        self.n_samples = n_samples

        self.n_step = n_step
        self.n_input = n_input

        # LSTM shape
        self.n_lstm1 = n_lstm1
        self.n_lstm2 = n_lstm2

        # fully connected
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.epoch = epoch

    def dataset(self):

        train_x_list = []
        train_y_list = []

        stride = 5

        for i in range(self.n_samples):

            train_x = data[i*stride:i*stride + self.n_step*self.n_input]
            train_x = train_x.reshape(self.n_step, self.n_input)
            train_x_list.append(train_x)

            train_y = data[i*stride + self.n_step*self.n_input:i *
                           stride + self.n_step*self.n_input + self.n_output]
            train_y_list.append(train_y)

        train_data = np.array(train_x_list)
        train_label = np.array(train_y_list)

        test_data = data[10000:10000 + self.n_step*self.n_input]
        test_data = test_data.reshape(1, self.n_step, self.n_input)

        return train_data, train_label, test_data

    def LSTM_TensorFlow(self):

        lstm_network = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(self.n_step, self.n_input)),
            tf.keras.layers.LSTM(self.n_lstm1, return_sequences=True),
            tf.keras.layers.LSTM(self.n_lstm2),
            tf.keras.layers.Dense(self.n_hidden),
            tf.keras.layers.Dense(self.n_output),
        ])

        lstm_network.summary()

        lstm_network.compile(
            optimizer='adam', loss='mean_squared_error', metrics=['mse'])

        lstm_network.fit(self.dataset()[0], self.dataset()[
                         1], epochs=self.epoch)

        gen_signal = []

        for i in range(self.n_step):

            test_pred = lstm_network.predict(self.dataset()[2])
            gen_signal.append(test_pred.ravel())
            test_pred = test_pred[:, np.newaxis, :]

            test_data = self.dataset()[2][:, 1:, :]
            test_data = np.concatenate([test_data, test_pred], axis=1)

        gen_signal = np.concatenate(gen_signal)

        test_label = data[10000:10000 + self.n_step *
                          self.n_input + self.n_step*self.n_input]

        plt.figure(figsize=(10, 6))
        plt.plot(np.arange(0, self.n_step*self.n_input + self.n_step *
                 self.n_input), test_label, 'b', label='ground truth')
        plt.plot(np.arange(self.n_step*self.n_input,  self.n_step*self.n_input +
                 self.n_step*self.n_input), gen_signal, 'r', label='prediction')
        plt.vlines(self.n_step*self.n_input, -1, 1,
                   colors='r', linestyles='dashed')
        plt.legend(fontsize=15, loc='upper left')
        plt.xlim(0, len(test_label))

        fig2 = plt.show()

        return fig2


back_test = [LSTM_2(5000, 25, 100, 100, 100, 100, 100, 3)]

# from line 104 to line 128 -> error
# ----------------------------------------------------------------------------------------------------
# import data(2)

# Microsoft stock price data
df_MSFT = yf.download('MSFT', start="1990-01-01", end="2022-03-10")
df_MSFT['ln_Close'] = np.log(df_MSFT['Close'])
df_MSFT['r_Close'] = df_MSFT['Close']-df_MSFT['Close'].shift(1)

df = df_MSFT[['Close', 'ln_Close', 'r_Close']]
df.dropna(inplace=True)

data = df['ln_Close'].values
# ----------------------------------------------------------------------------------------------------
# LSTM(2)


class LSTM_1:

    def __init__(self, n_samples, n_step, n_input, n_lstm1, n_lstm2, n_hidden, n_output, epoch):

        self.n_samples = n_samples

        self.n_step = n_step
        self.n_input = n_input

        # LSTM shape
        self.n_lstm1 = n_lstm1
        self.n_lstm2 = n_lstm2

        # fully connected
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.epoch = epoch

    def dataset(self):

        train_x_list = []
        train_y_list = []

        stride = 5

        for i in range(self.n_samples):

            train_x = data[i*stride:i*stride + self.n_step*self.n_input]
            train_x = train_x.reshape(self.n_step, self.n_input)
            train_x_list.append(train_x)

            train_y = data[i*stride + self.n_step*self.n_input:i *
                           stride + self.n_step*self.n_input + self.n_output]
            train_y_list.append(train_y)

        train_data = np.array(train_x_list)
        train_label = np.array(train_y_list)

        test_data = data[1000:1000 + self.n_step*self.n_input]
        test_data = test_data.reshape(1, self.n_step, self.n_input)

        return train_data, train_label, test_data

    def LSTM_TensorFlow(self):

        lstm_network = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(self.n_step, self.n_input)),
            tf.keras.layers.LSTM(self.n_lstm1, return_sequences=True),
            tf.keras.layers.LSTM(self.n_lstm2),
            tf.keras.layers.Dense(self.n_hidden),
            tf.keras.layers.Dense(self.n_output),
        ])

        lstm_network.summary()

        lstm_network.compile(
            optimizer='adam', loss='mean_squared_error', metrics=['mse'])

        lstm_network.fit(self.dataset()[0], self.dataset()[
                         1], epochs=self.epoch)

        test_pred = lstm_network.predict(self.dataset()[2]).ravel()
        test_label = data[1000:1000 +
                          self.n_step*self.n_input + self.n_input]

        plt.figure(figsize=(12, 6))
        plt.plot(np.arange(0, self.n_step*self.n_input + self.n_input),
                 test_label, 'b', label='ground truth')
        plt.plot(np.arange(self.n_step*self.n_input, self.n_step *
                 self.n_input + self.n_input), test_pred, 'r', label='prediction')
        plt.vlines(self.n_step*self.n_input, -1, 1,
                   colors='r', linestyles='dashed')
        plt.legend(fontsize=15, loc='upper left')
        plt.xlim(0, len(test_label))

        fig1 = plt.show()

        return fig1


firm_set = [LSTM_1(500, 10, 100, 100, 100, 100, 100, 500)]


class LSTM_2:

    def __init__(self, n_samples, n_step, n_input, n_lstm1, n_lstm2, n_hidden, n_output, epoch):

        self.n_samples = n_samples

        self.n_step = n_step
        self.n_input = n_input

        # LSTM shape
        self.n_lstm1 = n_lstm1
        self.n_lstm2 = n_lstm2

        # fully connected
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.epoch = epoch

    def dataset(self):

        train_x_list = []
        train_y_list = []

        stride = 5

        for i in range(self.n_samples):

            train_x = data[i*stride:i*stride + self.n_step*self.n_input]
            train_x = train_x.reshape(self.n_step, self.n_input)
            train_x_list.append(train_x)

            train_y = data[i*stride + self.n_step*self.n_input:i *
                           stride + self.n_step*self.n_input + self.n_output]
            train_y_list.append(train_y)

        train_data = np.array(train_x_list)
        train_label = np.array(train_y_list)

        test_data = data[1000:1000 + self.n_step*self.n_input]
        test_data = test_data.reshape(1, self.n_step, self.n_input)

        return train_data, train_label, test_data

    def LSTM_TensorFlow(self):

        lstm_network = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(self.n_step, self.n_input)),
            tf.keras.layers.LSTM(self.n_lstm1, return_sequences=True),
            tf.keras.layers.LSTM(self.n_lstm2),
            tf.keras.layers.Dense(self.n_hidden),
            tf.keras.layers.Dense(self.n_output),
        ])

        lstm_network.summary()

        lstm_network.compile(
            optimizer='adam', loss='mean_squared_error', metrics=['mse'])

        lstm_network.fit(self.dataset()[0], self.dataset()[
                         1], epochs=self.epoch)

        gen_signal = []

        for i in range(self.n_step):

            test_pred = lstm_network.predict(self.dataset()[2])
            gen_signal.append(test_pred.ravel())
            test_pred = test_pred[:, np.newaxis, :]

            test_data = self.dataset()[2][:, 1:, :]
            test_data = np.concatenate([test_data, test_pred], axis=1)

        gen_signal = np.concatenate(gen_signal)

        test_label = data[1000:1000 + self.n_step *
                          self.n_input + self.n_step*self.n_input]

        plt.figure(figsize=(12, 6))
        plt.plot(np.arange(0, self.n_step*self.n_input + self.n_step *
                 self.n_input), test_label, 'b', label='ground truth')
        plt.plot(np.arange(self.n_step*self.n_input,  self.n_step*self.n_input +
                 self.n_step*self.n_input), gen_signal, 'r', label='prediction')
        plt.vlines(self.n_step*self.n_input, -1, 1,
                   colors='r', linestyles='dashed')
        plt.legend(fontsize=15, loc='upper left')
        plt.xlim(0, len(test_label))

        fig2 = plt.show()

        return fig2


firm_set = [LSTM_1(500, 10, 100, 100, 100, 100, 100, 500)]

# from line 104 to line 128 -> error
# ====================================================================================================
