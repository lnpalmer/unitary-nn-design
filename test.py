import torch
import torch.optim as optim
import gym
import time
import os.path

from designer import DesignerNetwork
from dppo import DPPO
import design_env

N = design_env.N
I = design_env.I
O = design_env.O

lr = 1e-3
W = 4
T = W * 32
D = 1
M = 3
timesteps = 1000000
model_path = "designer_params"

model = DesignerNetwork(N, I, O)
optimizer = optim.Adam(model.parameters(), lr=lr)
def clone_model(model):
    clone_N, clone_I, clone_O = model.N, model.I, model.O
    clone = DesignerNetwork(clone_N, clone_I, clone_O)
    clone.load_state_dict(model.state_dict())
    return clone

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path))

rl = DPPO(
    env_name="unitary-design-v0", model=model, clone_model=clone_model,
    optimizer=optimizer, T=T, W=W, D=D, M=M)
rl.run(timesteps)

torch.save(model.state_dict(), model_path)
