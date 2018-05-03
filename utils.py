import random
import torch
import torch.multiprocessing as mp
from pygraphviz import AGraph

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

def gae(rewards, dones, values, gamma=0.99, lambda_=0.95):
    T = len(rewards)

    values = [value.detach().data for value in values]

    returns = [None] * T
    advantages = [None] * T

    return_ = values[T]
    advantage = torch.zeros(1, 1)

    for t in reversed(range(T)):
        return_ = rewards[t] + return_ * gamma

        delta = gamma * values[t + 1] + float(rewards[t]) - values[t]
        advantage = delta + advantage * gamma * lambda_

        if dones[t]:
            return_ = torch.zeros(1, 1)
            advantage = torch.zeros(1, 1)

        returns[t] = return_
        advantages[t] = advantage

    returns = torch.cat(returns)
    advantages = torch.cat(advantages)
    return returns, advantages

class MPSwitch():

    def __init__(self):
        self._value = mp.Value("b", False)
        self._lock = mp.Lock()

    def get(self):
        with self._lock:
            return self._value.value

    def set(self, value):
        with self._lock:
            self._value.value = value

    def flip(self):
        with self._lock:
            self._value.value = not self._value.value

class MPCounter():

    def __init__(self):
        self._value = mp.Value("i", 0)
        self._lock = mp.Lock()

    def get(self):
        with self._lock:
            return self._value.value

    def increment(self):
        with self._lock:
            self._value.value += 1

    def reset(self):
        with self._lock:
            self._value.value = 0

def push_grads(src_model, dest_model):
    dest_params = dict(dest_model.named_parameters())
    for name, src_param in src_model.named_parameters():
        dest_param = dest_params[name]
        dest_param.grad.data.add_(src_param.grad.data)
