# ====================================================================================================
# GCN: graph convolution network
# ====================================================================================================
# GNN: graph neural networks

# 3 applications
# task 1: node classification(fixed total graph)
# task 2: edge prediction
# task 3: graph classification
# ----------------------------------------------------------------------------------------------------
# install library

# !pip install spektral==0.6.0
# !pip install tensorflow==2.2.0
# !pip install keras==2.3.0
# ----------------------------------------------------------------------------------------------------
# import library

from sklearn.manifold import TSNE
import tensorflow as tf
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import scipy as sp
from scipy.linalg import fractional_matrix_power
import spektral
# ----------------------------------------------------------------------------------------------------
# graph

# undirected graph(asymmetric)
# directed graph(symmetric)

# G(V, E): graph
# V: a set of vertices(nodes)
# E: a set of edges(links, relations)
# W: weight(edge property)
# ----------------------------------------------------------------------------------------------------
# graph (1)

# define graph g
g = nx.Graph()

# add edges to graph g
g.add_edge('a', 'b')
g.add_edge('b', 'c')
g.add_edge('a', 'c')
g.add_edge('c', 'd')

# draw a graph with nodes and edges
nx.draw(g)
plt.show()
# ----------------------------------------------------------------------------------------------------
# graph (2)

# define graph g
g = nx.Graph()

# add edges to graph g
g.add_edge('a', 'b')
g.add_edge('b', 'c')
g.add_edge('a', 'c')
g.add_edge('c', 'd')

# draw a graph with node labels
pos = nx.spring_layout(g)

nx.draw(g, pos, node_size=500)
nx.draw_networkx_labels(g, pos, font_size=10)
plt.show()
# ----------------------------------------------------------------------------------------------------
# graph (3)

G = nx.Graph()
# G = nx.DiGraph()

G.add_nodes_from([1, 2, 3, 4, 5])
G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4), (3, 5), (4, 5)])

# plot a graph
pos = nx.spring_layout(G)

nx.draw(G, pos, node_size=500)
nx.draw_networkx_labels(G, pos, font_size=10)
plt.show()

print(nx.number_of_nodes(G))
print(nx.number_of_edges(G))
print(G.nodes())
print(G.edges())
# ----------------------------------------------------------------------------------------------------
# adjacent matrix <-> kernel

A = nx.adjacency_matrix(G)
A.todense()

A = np.array([[0, 1, 1, 1],
              [1, 0, 0, 0],
              [1, 0, 0, 1],
              [1, 0, 1, 0]])
# ----------------------------------------------------------------------------------------------------
# self-connecting edges
A_self = A + np.eye(4)

# degree
D = np.array(A_self.sum(1)).flatten()
D = np.diag(D)

# neighborhood normalization
# 1st attempt -> not symmetric
A_norm = np.linalg.inv(D).dot(A_self)

# 2nd attempt -> symmetric
D_half_norm = fractional_matrix_power(D, -0.5)

A_self = np.asmatrix(A_self)
D_half_norm = np.asmatrix(D_half_norm)

A_half_norm = D_half_norm*A_self*D_half_norm
# ----------------------------------------------------------------------------------------------------
# example

# graph
G = nx.Graph()

G.add_nodes_from([1, 2, 3, 4, 5, 6])
G.add_edges_from([(1, 2), (1, 3), (2, 3), (1, 4), (4, 5), (4, 6), (5, 6)])

nx.draw(G, with_labels=True, node_size=600, font_size=22)
plt.show()

# adjacent matrix <-> kernel
A = nx.adjacency_matrix(G).todense()
H = np.matrix([1, 0, 0, -1, 0, 0]).T
A*H

# self-connecting edges
A_self = A + np.eye(6)
A_self*H

# degree
D = np.array(A_self.sum(1)).flatten()
D = np.diag(D)

# neighborhood normalization
# 2nd attempt -> symmetric
D_half_norm = fractional_matrix_power(D, -0.5)

A_self = np.asmatrix(A_self)
D_half_norm = np.asmatrix(D_half_norm)

A_half_norm = D_half_norm*A_self*D_half_norm

A_half_norm*H

# build 2-layer GCN using ReLU as the activation function
np.random.seed(20)

W1 = np.random.randn(1, 4)  # input: 1 -> hidden: 4
W2 = np.random.randn(4, 2)  # hidden: 4 -> output: 2

# layer(GCN_func)

# step 1: message aggregation from local neighborhood
# message(H)
# aggregation(A_self)
# normalization(D_half_norm)

# step 2: update
# update 1(W, linear combination)
# update 2(ReLU or sigmoid, non-linear function)


def ReLU(x):

    return np.maximum(0, x)


def GCN_func(A_self, H, W):

    D = np.diag(np.array(A_self.sum(1)).flatten())

    D_half_norm = fractional_matrix_power(D, -0.5)

    H_new = D_half_norm*A_self*D_half_norm*H*W

    return ReLU(H_new)


H1 = H
H2 = GCN_func(A_self, H1, W1)
H3 = GCN_func(A_self, H2, W2)
# ----------------------------------------------------------------------------------------------------
# import CORA data

nodes = np.load('cora_nodes.npy')
edge_list = np.load('cora_edges.npy')

labels_encoded = np.load('cora_labels_encoded.npy')

H = np.load('cora_features.npy')
data_mask = np.load('cora_mask.npy')

N = H.shape[0]
F = H.shape[1]

print('H shape: ', H.shape)
print('the number of nodes (N): ', N)
print('the number of features (F) of each node: ', F)

num_classes = 7
print('the number of classes: ', num_classes)

# index of node for train model
train_mask = data_mask[0]

# index of node for test model
test_mask = data_mask[1]

print("the number of trainig data: ", np.sum(train_mask))
print("the number of test data: ", np.sum(test_mask))
# ----------------------------------------------------------------------------------------------------
# GCN: message passing framework

# G(V,E): graph inputs -> graph conv -> graph conv -> readout -> softmax -> prediction


class GCN:

    def __init__(self, LR, epoch):

        self.LR = LR
        self.epoch = epoch

    def GCN_TensorFlow(self):

        G = nx.Graph(name='Cora')

        G.add_nodes_from(nodes)
        G.add_edges_from(edge_list)

        print('Graph info: ', nx.info(G))

        # message passing with self-loops

        # adjacent matrix
        A = nx.adjacency_matrix(G)
        I = np.eye(A.shape[-1])

        # self-connecting edges
        A_self = A + I

        # degree
        D = np.diag(np.array(A_self.sum(1)).flatten())

        # neighborhood normalization
        D_half_norm = fractional_matrix_power(D, -0.5)
        A_half_norm = D_half_norm * A_self * D_half_norm
        A_half_norm = np.array(A_half_norm)

        H = np.load('cora_features.npy')
        H = np.array(H)

        # graph convolutional networks

        H_in = tf.keras.layers.Input(shape=(F, ))
        A_in = tf.keras.layers.Input(shape=(N, ))

        # 1st graph convolution layer
        graph_conv_1 = spektral.layers.GCSConv(
            channels=16, activation='relu')([H_in, A_in])

        # 2nd graph convolution layer
        graph_conv_2 = spektral.layers.GCSConv(
            channels=7, activation='softmax')([graph_conv_1, A_in])

        # build model
        model = tf.keras.models.Model(
            inputs=[H_in, A_in], outputs=graph_conv_2)

        # compile model with optimizer, loss and weighted metrics
        model.compile(optimizer=tf.keras.optimizers.Adam(
            learning_rate=self.LR), loss='categorical_crossentropy', weighted_metrics=['acc'])

        # summary of model
        model.summary()

        # train model with x, y, sample weight, epochs, batch size and shuffle
        # fixed total graph -> batch size(N) & shuffle(False)
        model.fit([H, A_half_norm], labels_encoded,
                  sample_weight=train_mask, epochs=self.epoch, batch_size=N, shuffle=False)

        # model evaluation
        y_pred = model.evaluate(
            [H, A_half_norm], labels_encoded, sample_weight=test_mask, batch_size=N)

        # visualization(16 dims -> 2 dims)
        layer_outputs = [layer.output for layer in model.layers]

        activation_model = tf.keras.models.Model(
            inputs=model.input, outputs=layer_outputs)

        activations = activation_model.predict([H, A_half_norm], batch_size=N)

        x_tsne = TSNE(n_components=2).fit_transform(activations[2])

        def plot_tSNE(labels_encoded, x_tsne):

            color_map = np.argmax(labels_encoded, axis=1)

            plt.figure(figsize=(10, 10))

            for cl in range(num_classes):

                indices = np.where(color_map == cl)
                indices = indices[0]

                plt.scatter(x_tsne[indices, 0], x_tsne[indices, 1], label=cl)

            plt.legend()
            plt.show()

        return plot_tSNE(labels_encoded, x_tsne)


back_test = [GCN(1e-2, 30)]
# ====================================================================================================
