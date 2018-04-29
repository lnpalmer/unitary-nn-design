import random
from math import sqrt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
import torch.optim as optim
import networkx as nx
from pygraphviz import AGraph

from dagnn import gen_dagnn
from binary import gen_binary_program
from adam_quickfix import SparseAdamQuickfixed

N = 128
I = 16
O = 16
M = 1000
E = 100
train_size = 50000
test_size = 10000

primary_network = gen_dagnn(N, I, O)
primary_network.gen_parameters()
dense_optimizer = optim.Adam(primary_network.dense_parameters(), lr=5e-3)
sparse_optimizer = SparseAdamQuickfixed(primary_network.sparse_parameters(), lr=5e-3)

bp = gen_binary_program(I, O, random.randint(32, 32))

x_train = torch.randn(train_size, I) > .5
y_train = bp.run(x_train)
x_train, y_train = x_train.float(), y_train.float()

x_test = torch.randn(test_size, I) > .5
y_test = bp.run(x_test)
x_test, y_test = x_test.float(), y_test.float()

def draw_net(net, path):
    N, I, O = net.N, net.I, net.O

    net.sync_graph()
    AG = AGraph(directed=True, dpi=300)
    AG.graph_attr["outputorder"] = "edgesfirst"
    AG.graph_attr["bgcolor"] = "#000000"

    for node in net._graph.nodes(data=True):
        i, props = node
        random.seed(i)
        if i < I:
            AG.add_node(i, shape="point", pos=("%f,%f!" % (0 - .4, 4. * i / (I - 1))))
        elif i >= N - O:
            i_ = i - (N - O)
            AG.add_node(i, shape="point", pos=("%f,%f!" % (6 + .4, 4. * i_ / (O - 1))))
        else:
            i_ = i - I
            AG.add_node(i,
                        shape="point",
                        pos=("%f,%f!" % (6. * i_ / (N - (I + O) + 1), .25 + 3.5 * random.random())),
                        label="")

    for edge in net._graph.edges(data=True):
        j, i, props = edge
        w_ij = props["weight"]
        color = "#FFAF7F" if w_ij > 0. else "#CF7FCF"
        AG.add_edge(j, i, penwidth=abs(w_ij) * .3, arrowsize=abs(w_ij) * .05 + .3, color=color)

    AG.draw(path, prog="neato")

for e in range(E):

    idxs = np.arange(train_size)
    np.random.shuffle(idxs)

    for start in range(0, train_size, M):
        primary_network.zero_grad()

        idxs_mb = idxs[start:start + M]
        x_mb = x_train[idxs_mb]
        y_mb = y_train[idxs_mb]

        one_probs = Fnn.sigmoid(primary_network(x_mb))
        bit_probs = y_mb * one_probs + (1 - y_mb) * (1 - one_probs)
        loss = -bit_probs.log().sum() / M

        loss.backward()
        dense_optimizer.step()
        sparse_optimizer.step()

    one_probs = Fnn.sigmoid(primary_network(x_test))
    bit_probs = y_test * one_probs + (1 - y_test) * (1 - one_probs)
    loss = -bit_probs.log().sum() / test_size

    correct = ((bit_probs > .5).sum(1) == O).sum().detach().numpy()
    accuracy = float(correct) / float(test_size)

    print("completed epoch %i, test loss: %f, test accuracy: %f" % (e, loss.detach().numpy(), accuracy))

for step in bp.steps:
    print(step)
