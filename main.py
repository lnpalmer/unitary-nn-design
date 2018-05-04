import argparse
import torch
import torch.optim as optim
import gym
import time
import os.path

from designer import DesignerNetwork
from dppo import DPPO
import design_env

parser = argparse.ArgumentParser(
    description="Unitary Neural Network Design ||| Lukas Palmer",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-W", type=int, default=4, help="number of workers (DPPO)")
parser.add_argument("-D", type=int, default=1, help="tolerance for incomplete workers (DPPO)")
parser.add_argument("-M", type=int, default=3, help="number of optimizer steps per round (DPPO)")
parser.add_argument("-T", type=int, default=128, help="environment steps per round (DPPO)")
parser.add_argument("--timesteps", type=int, default=200000, help="total number of timesteps (DPPO)")
parser.add_argument("--lr", type=float, default=1e-3, help="designer network learning rate")
parser.add_argument("--model-path", type=str, default="designer_params", help="path to save designer to (and load from if the file already exists)")
parser.add_argument("--value-coeff", type=float, default=.5, help="coeffecient for value loss term")
parser.add_argument("--S_phi", type=int, default=60, help="size of RNN hidden states for the designer network")
parser.add_argument("--S_rho", type=int, default=60, help="size of primary unit representations for the designer network")

args = parser.parse_args()

N = design_env.N
I = design_env.I
O = design_env.O

model = DesignerNetwork(N, I, O, S_example=design_env.S_example, S_phi=args.S_phi, S_rho=args.S_rho)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
def clone_model(model):
    clone_N, clone_I, clone_O = model.N, model.I, model.O
    clone_S_example, clone_S_phi, clone_S_rho = model.S_example, model.S_phi, model.S_rho
    clone = DesignerNetwork(
        clone_N,
        clone_I,
        clone_O,
        S_example=clone_S_example,
        S_phi=clone_S_phi,
        S_rho=clone_S_rho)
    clone.load_state_dict(model.state_dict())
    return clone

def logger(**kwargs):
    rank = kwargs["rank"]
    if rank == 0:
        rewards = kwargs["rewards"]
        env_losses = kwargs["env_losses"]
        env_model = kwargs["env_model"]
        timesteps_done = kwargs["timesteps_done"]
        print("@%i avg. designer reward: %f, avg. primary loss: %f, primary net: %iN, %iW" % (
            timesteps_done,
            sum(rewards) / float(len(rewards)),
            sum(env_losses) / float(len(env_losses)),
            len(env_model._graph.nodes()),
            len(env_model._graph.edges())))

if os.path.exists(args.model_path):
    model.load_state_dict(torch.load(args.model_path))

rl = DPPO(
    env_name="unitary-design-v0",
    model=model,
    clone_model=clone_model,
    optimizer=optimizer,
    T=args.T,
    W=args.W,
    D=args.D,
    M=args.M,
    logger=logger,
    value_coeff=args.value_coeff)
rl.run(args.timesteps)

torch.save(model.state_dict(), args.model_path)
