import random
from math import sqrt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
import torch.optim as optim
import networkx as nx
from pygraphviz import AGraph
import gym
import time

from dagnn import gen_dagnn
from binary import gen_binary_program
from adam_quickfix import SparseAdamQuickfixed
from designer import DesignerNetwork
from dppo import DPPO
import design_env

N = 48
I = 8
O = 8
lr = 3e-4
T = 128
W = 4
D = 1
M = 3
timesteps = 8192

model = DesignerNetwork(N, I, O)
optimizer = optim.Adam(model.parameters(), lr=lr)
def clone_model(model):
    clone_N, clone_I, clone_O = model.N, model.I, model.O
    clone = DesignerNetwork(clone_N, clone_I, clone_O)
    clone.load_state_dict(model.state_dict())
    return clone

rl = DPPO(
    env_name="unitary-design-v0", model=model, clone_model=clone_model,
    optimizer=optimizer, T=T, W=W, D=D, M=M)
rl.run(timesteps)
