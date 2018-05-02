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
from utils import draw_net
from dppo import DPPO
import design_env

# env = gym.make('unitary-design-v0')
# ob = env.reset()
#
# designer_network = DesignerNetwork(env.N, env.I, env.O)
#
# draw_net(env.primary_network, "before.png")
#
# for _ in range(200):
#     action_probs, v = designer_network(ob)
#     action = designer_network.choose_action(action_probs, epsilon=.1)
#     action_prob = designer_network.prob_action(action_probs, action)
#     ob, reward, done, _ = env.step(action)
#     print(f"action: {str(action).ljust(20)} (prob {action_prob:.4f}), reward: {reward:.4f}")
#
# draw_net(env.primary_network, "after.png")

N = 48
I = 8
O = 8
lr = 3e-5
T = 1024
W = 4
timesteps = 1024

model = DesignerNetwork(N, I, O)
optimizer = optim.Adam(model.parameters(), lr=lr)
def clone_model(model):
    clone_N, clone_I, clone_O = model.N, model.I, model.O
    clone = DesignerNetwork(clone_N, clone_I, clone_O)
    clone_parameters = dict(clone.named_parameters())
    for name, param in model.named_parameters():
        clone_param = clone_parameters[name]
        clone_param.data.copy_(param.data)
    return clone

rl = DPPO(
    env_name="unitary-design-v0", model=model, clone_model=clone_model,
    optimizer=optimizer, T=T, W=W)
rl.run(timesteps)
