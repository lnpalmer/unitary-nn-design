import random
import torch
import torch.nn as nn
from torch.autograd import Function
import networkx as nx

import dagnn_cpp

class DAGNN(nn.Module):

    def __init__(self, N, I, O):
        super().__init__()

        self.N = N
        self.I = I
        self.O = O

        self.I_, self.O_ = torch.IntTensor([I]), torch.IntTensor([O])

        self._graph = nx.DiGraph()

        for i in range(I):
            self._populate(i, 0)

        for i in range(N - O, N):
            self._populate(i, 0)

        for i in range(N - O, N):
            for j in range(I):
                self.connect(j, i, random.random() -.5)

        self._needs_gen = True

    def forward(self, x):
        if self._needs_gen:
            self._gen_parameters()

        N, W, b, O, I_, O_ = self.N, self.W, self.b, self.O, self.I_, self.O_

        a = DAGNNFunction.apply(x, W, b, I_, O_)
        y = a[:, N - O:N]

        return y

    """ Considered an implementation detail, since it does not guarantee a clean graph """
    def _populate(self, i, b_i):
        self._graph.add_node(i, b=b_i)
        self._needs_gen=True

    def connect(self, j, i, w_ij):
        self._graph.add_edge(j, i, weight=w_ij)
        self._needs_gen=True

    def addunit(self, j, i, b_k, w_ik, w_kj):
        N, I, O = self.N, self.I, self.O
        while True:
            k = random.randint(max(I, j + 1), min(N - O, i))
            if not self._graph.has_node(k):
                self._populate(k, b_k)
                self.connect(j, k, w_kj)
                self.connect(k, i, w_ik)
                self._needs_gen = True
                return

    def gen_parameters(self):
        N = self.N

        nodes = self._graph.nodes(data=True)
        edges = self._graph.edges(data=True)

        b = torch.zeros(N)
        for node in nodes:
            i, b_i = node
            b_i = b_i["b"]

            b[i] = b_i
        self.b = nn.Parameter(b)

        W_i = torch.zeros(2, len(edges)).long()
        W_v = torch.zeros(len(edges))
        w_pos = 0
        for edge in edges:
            j, i, w_ij = edge
            w_ij = w_ij["weight"]

            W_i[0][w_pos] = i
            W_i[1][w_pos] = j
            W_v[w_pos] = w_ij

            w_pos += 1
        self.W = nn.Parameter(torch.sparse_coo_tensor(W_i, W_v, (N, N)))
        self._needs_gen = False

    def dense_parameters(self):
        dp = []
        for param in self.parameters():
            if not param.is_sparse:
                dp.append(param)
        return dp

    def sparse_parameters(self):
        sp = []
        for param in self.parameters():
            if param.is_sparse:
                sp.append(param)
        return sp

class DAGNNFunction(Function):

    @staticmethod
    def forward(ctx, x, W, b, i, o):
        a = dagnn_cpp.forward(x, W, b, i, o)
        ctx.save_for_backward(W, b, i, o, a)
        return a

    @staticmethod
    def backward(ctx, da):
        W, b, i, o, a = ctx.saved_variables
        dx, dW, db = dagnn_cpp.backward(W, b, i, o, a, da)
        return dx, dW, db, torch.zeros(1), torch.zeros(1)
