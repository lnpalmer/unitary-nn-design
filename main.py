import argparse
import torch
import torch.optim as optim
import gym
import time
import random
import os.path

from utils import draw_net, static_var
from designer import DesignerNetwork, info_action_prob
from dppo import DPPO
import design_env

parser = argparse.ArgumentParser(
    description="Unitary Neural Network Design ||| Lukas Palmer",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-W", type=int, default=4, help="number of workers (DPPO)")
parser.add_argument("-D", type=int, default=1, help="tolerance for incomplete workers (DPPO)")
parser.add_argument("-M", type=int, default=3, help="number of optimizer steps per round (DPPO)")
parser.add_argument("-T", type=int, default=512, help="environment steps per round (DPPO)")
parser.add_argument("--timesteps", type=int, default=int(1e6), help="total number of timesteps (DPPO)")
parser.add_argument("--lr", type=float, default=3e-4, help="designer network learning rate")
parser.add_argument("--value-coeff", type=float, default=.5, help="coeffecient for value loss term")
parser.add_argument("--S_phi", type=int, default=100, help="size of RNN hidden states for the designer network")
parser.add_argument("--S_rho", type=int, default=100, help="size of primary unit representations for the designer network")
parser.add_argument("--model-path", type=str, default="./designer_params", help="path on which to load and save designer")
parser.add_argument("--load-model", action="store_true", help="load the model at --model-path")
parser.add_argument("--save-model", action="store_true", help="save the model")
parser.add_argument("--save-interval", type=int, default=10000, help="step interval on which to save a checkpoint")

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

if args.load_model and os.path.exists(args.model_path):
    model.load_state_dict(torch.load(args.model_path))

@static_var("start_time", time.time())
@static_var("next_save", args.save_interval)
def logger(**kwargs):
    rank = kwargs["rank"]
    if rank == 0:
        rewards = kwargs["rewards"]
        env_losses = kwargs["env_losses"]
        env_model = kwargs["env_model"]
        agent_model = kwargs["agent_model"]
        timesteps_done = kwargs["timesteps_done"]

        print(("=" * 20 + " %i steps " + "=" * 20) % timesteps_done)
        print()
        
        print("Avg. designer reward: %f, avg. primary loss: %f, primary net: %iN, %iW" % (
            sum(rewards) / float(len(rewards)),
            sum(env_losses) / float(len(env_losses)),
            len(env_model._graph.nodes()),
            len(env_model._graph.edges())))
        print()

        print("Sample action probabilities:")
        sample_action_prob = kwargs["sample_action_prob"]
        info_action_prob(sample_action_prob)
        print()

        print("Average speed of %f steps / second" % (timesteps_done / (time.time() - logger.start_time)))
        print()

        draw_net(env_model, "current.png")
        
        if args.save_model and timesteps_done >= logger.next_save:
            torch.save(agent_model.state_dict(), args.model_path)
            with open(args.model_path + ".steps", "w+") as file:
                file.write("%i\n" % timesteps_done)
            logger.next_save += args.save_interval

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

if args.save_model:
    torch.save(model.state_dict(), args.model_path)
