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
import design_env

env = gym.make('unitary-design-v0')
ob = env.reset()

designer_network = DesignerNetwork(env.N, env.I, env.O)

draw_net(env.primary_network, "before.png")

for _ in range(200):
    action_probs, v = designer_network(ob)
    action = designer_network.choose_action(action_probs, epsilon=.1)
    action_prob = designer_network.prob_action(action_probs, action)
    ob, reward, done, _ = env.step(action)
    print(f"action: {str(action).ljust(20)} (prob {action_prob:.4f}), reward: {reward:.4f}")

draw_net(env.primary_network, "after.png")
