from math import sqrt
import random
import numpy as np
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
            self.populate(i, 0.)

        for i in range(N - O, N):
            self.populate(i, 0.)

        # bare minimum W, updated immediately
        self.W = nn.Parameter(
            torch.sparse.FloatTensor(
                torch.zeros(2, 1).long(),
                torch.ones(1),
                torch.Size([N, N])))
        self.b = nn.Parameter(torch.zeros(N))

        self._needs_gen = True

    def forward(self, x):
        if self._needs_gen:
            self.gen_parameters()

        N, W, b, O, I_, O_ = self.N, self.W, self.b, self.O, self.I_, self.O_

        y, _ = DAGNNFunction.apply(x, W, b, I_, O_)

        return y

    def populate(self, i, b_i):
        self._graph.add_node(i, b=b_i)
        self._needs_gen=True

    def connect(self, j, i, w_ij):
        N, I, O = self.N, self.I, self.O

        if self._graph.has_edge(j ,i) or i <= j or i < I and j < I or i >= N - O and j >= N - O:
            return False
        else:
            self._graph.add_edge(j, i, weight=w_ij)
            self._needs_gen = True
            return True

    def disconnect(self, j, i):
        if self._graph.has_edge(j, i):
            self._graph.remove_edge(j, i)
            self._needs_gen = True
            self.autoremove()
            return True
        else:
            return False

    def addunit(self, j, i, b_k, w_ik, w_kj):
        N, I, O = self.N, self.I, self.O

        k_candidates = range(max(I, j + 1), min(N - O, i))
        k_candidates = [k for k in k_candidates if not self._graph.has_node(k)]

        if len(k_candidates) > 0:
            k = random.choice(k_candidates)
            self.populate(k, b_k)
            self.connect(j, k, w_kj)
            self.connect(k, i, w_ik)
            self._needs_gen = True
            return True
        else:
            return False

    def delunit(self, i):
        N, I, O = self.N, self.I, self.O

        if i < I or i >= N - O or not self._graph.has_node(i):
            return False
        else:
            self._graph.remove_node(i)
            self.autoremove()
            return True

    """ Update torch parameters from graphical network description. """
    def gen_parameters(self):
        N = self.N

        nodes = self._graph.nodes(data=True)
        edges = self._graph.edges(data=True)

        b_new = np.zeros(N)
        for node in nodes:
            i, b_i = node
            b_i = b_i['b']

            b_new[i] = b_i
        b_new = torch.Tensor(b_new)

        # HACK to avoid creating a new parameter, which disrupts optimizers
        self.b.data.zero_().add_(b_new)

        W_i = torch.zeros(2, len(edges)).long()
        W_v = torch.zeros(len(edges))
        w_pos = 0
        for edge in edges:
            j, i, w_ij = edge
            w_ij = w_ij['weight']

            W_i[0][w_pos] = i
            W_i[1][w_pos] = j
            W_v[w_pos] = w_ij

            w_pos += 1

        # HACK to avoid creating a new parameter, which disrupts optimizers
        W_new = torch.sparse_coo_tensor(W_i, W_v, (N, N))
        self.W.data.zero_().add_(W_new)

        self._needs_gen = False

    """ Update the graph representation to match the actual network parameters. """
    def sync_graph(self):
        if self._needs_gen:
            self.gen_parameters()

        b = self.b.data.numpy()
        for i in range(self.N):
            if self._graph.has_node(i):
                self._graph.add_node(i, b=b[i])

        W_i_T = self.W._indices().transpose(0, 1).detach().numpy()
        W_v = self.W._values().detach().numpy()
        W_map = {}
        for pos_W in range(W_v.shape[0]):
            i, j = W_i_T[pos_W]
            w_ij = W_v[pos_W]
            W_map[(i, j)] = float(w_ij)

        for edge in self._graph.edges():
            j, i = edge
            self._graph.add_edge(j, i, weight=W_map[(i, j)])

    def units(self):
        return self._graph.nodes()

    def autoremove(self):
        N, I, O = self.N, self.I, self.O

        clean = False
        while not clean:
            clean = True
            for i in self._graph.nodes():
                if i >= I and i < N - O:
                    F_i = self._graph.in_edges([i])
                    B_i = self._graph.out_edges([i])
                    if len(F_i) == 0 or len(B_i) == 0:
                        self.delunit(i)
                        clean = False
                        break

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

def gen_dagnn(N, I, O, n_H = None):
    if n_H is None:
        n_H = N - (I + O)
    nn = DAGNN(N, I, O)
    H = random.sample(range(I, N - O), n_H)

    for i in H:
        nn.populate(i, random.gauss(0, 1))
    units = nn.units()

    IH = list(range(0, I)) + H
    OH = list(range(N - O, N)) + H

    for i in OH:
        n_f_i = random.randint(1, 8)
        f_i = random.sample([j for j in IH if j < i], n_f_i)
        for j in f_i:
            nn.connect(j, i, random.gauss(0, 1. / sqrt(n_f_i)))

    for j in IH:
        n_b_j = random.randint(1, 8)
        b_j = random.sample([i for i in OH if j < i], n_b_j)
        for i in b_j:
            nn.connect(j, i, random.gauss(0, 1. / sqrt(n_b_j)))

    return nn

class DAGNNFunction(Function):

    @staticmethod
    def forward(ctx, x, W, b, i, o):
        a, z = dagnn_cpp.forward(x, W, b, i, o)
        x_s = torch.LongTensor([x.size()[1]])
        ctx.save_for_backward(W, b, i, o, a, x_s)

        N = W.size()[0]
        I = i.detach().numpy()[0]
        O = o.detach().numpy()[0]
        X_s = x_s.detach().numpy()[0]

        if X_s > I:
            x.data[:, I:X_s] = z.data[:, I:X_s]

        y = a[:, N - O:N]
        return y, z

    @staticmethod
    def backward(ctx, dy, dz):
        W, b, i, o, a, x_s = ctx.saved_variables
        dz, dW, db = dagnn_cpp.backward(W, b, i, o, a, dy)

        N = W.size()[0]
        X_s = x_s.detach().numpy()[0]
        dx = dz[:, :X_s]
        return dx, dW, db, torch.zeros(1), torch.zeros(1)
