import time
import numpy as np
import torch
import torch.multiprocessing as mp
import gym

from utils import gae, MPSwitch, MPCounter, push_grads, draw_net, log_param_stats

class DPPO:

    def __init__(self, **kwargs):
        self.env_name = kwargs["env_name"]
        self.model = kwargs["model"]
        self.clone_model = kwargs["clone_model"]
        self.optimizer = kwargs["optimizer"]
        self.T = kwargs["T"]
        self.W = kwargs["W"]
        self.D = kwargs["D"]
        self.M = kwargs["M"]
        self.logger = kwargs["logger"]
        self.value_coeff = kwargs["value_coeff"]

        self.step_counter = MPCounter()
        self.worker_counter = MPCounter()
        self.accepting_gradients = MPSwitch()

        for param in self.model.parameters():
            param.grad = torch.zeros_like(param)
        self.model.share_memory()

    def run(self, timesteps):
        processes = []

        self.model.zero_grad()

        # launch worker processes
        for w in range(self.W):
            process = mp.Process(
                target=dppo_worker,
                kwargs={
                    "rank": w,
                    "timesteps": timesteps,
                    "env_name": self.env_name,
                    "T": self.T,
                    "W": self.W,
                    "D": self.D,
                    "M": self.M,
                    "shared_model": self.model,
                    "clone_model": self.clone_model,
                    "step_counter": self.step_counter,
                    "worker_counter": self.worker_counter,
                    "accepting_gradients": self.accepting_gradients,
                    "logger": self.logger,
                    "value_coeff": self.value_coeff})
            process.start()
            processes.append(process)

        # launch the chief process
        process = mp.Process(
            target=dppo_chief,
            kwargs={
                "timesteps": timesteps,
                "T": self.T,
                "W": self.W,
                "D": self.D,
                "M": self.M,
                "shared_model": self.model,
                "optimizer": self.optimizer,
                "step_counter": self.step_counter,
                "worker_counter": self.worker_counter,
                "accepting_gradients": self.accepting_gradients})
        process.start()
        processes.append(process)

        for process in processes:
            process.join()

def dppo_worker(**kwargs):
    rank = kwargs["rank"]
    timesteps = kwargs["timesteps"]
    env_name = kwargs["env_name"]
    T = kwargs["T"]
    W = kwargs["W"]
    M = kwargs["M"]
    shared_model = kwargs["shared_model"]
    clone_model = kwargs["clone_model"]
    step_counter = kwargs["step_counter"]
    worker_counter = kwargs["worker_counter"]
    accepting_gradients = kwargs["accepting_gradients"]
    logger = kwargs["logger"]
    value_coeff = kwargs["value_coeff"]

    model = clone_model(shared_model)
    model_old = clone_model(shared_model)

    env = gym.make(env_name)
    done = True

    T_worker = T // W

    while step_counter.get() < (timesteps // T) * M:
        # interact with the environment
        obs = []
        rewards = []
        dones = []
        actions = []
        values = []
        env_losses = []

        for i in range(T_worker):
            if done:
                ob = env.reset()

            obs.append(ob)

            action_prob, value = model([ob])
            action = model.choose_action(action_prob, epsilon=.1)
            ob, reward, done, env_loss = env.step(action)

            rewards.append(reward)
            dones.append(done)
            actions.append(action)
            values.append(value)
            env_losses.append(env_loss)

        # get a final value to bootstrap returns with
        _, value = model([ob])
        values.append(value)

        returns, advantages = gae(rewards, dones, values)

        # perform optimization
        model_old.load_state_dict(model.state_dict())

        logger(
            rank=rank,
            rewards=rewards,
            env_model=env.primary_network,
            env_losses=env_losses,
            agent_model=shared_model,
            timesteps_done=(step_counter.get() // M) * T,
            sample_action_prob=action_prob)

        while True:
            step = step_counter.get()

            model.load_state_dict(shared_model.state_dict())

            model.zero_grad()
            model_old.zero_grad()

            action_prob, values = model(obs)
            action_taken_probs = model.prob_action(action_prob, actions)

            action_prob_old, _ = model_old(obs)
            action_taken_probs_old = model_old.prob_action(action_prob_old, actions).detach()

            loss = -ppo_objective(
                action_taken_probs,
                action_taken_probs_old,
                advantages)

            loss += value_coeff * (((values - returns) ** 2) / 2.).mean()
            loss.backward()

            if accepting_gradients.get():
                push_grads(model, shared_model)
                worker_counter.increment()

            while True:
                target_step = step_counter.get()
                if target_step > step:
                    break

                time.sleep(.1)

            if target_step % M == 0:
                break

        model.load_state_dict(shared_model.state_dict())

torch.nn.Parameter

def dppo_chief(**kwargs):
    timesteps = kwargs["timesteps"]
    T = kwargs["T"]
    W = kwargs["W"]
    D = kwargs["D"]
    M = kwargs["M"]
    shared_model = kwargs["shared_model"]
    optimizer = kwargs["optimizer"]
    step_counter = kwargs["step_counter"]
    worker_counter = kwargs["worker_counter"]
    accepting_gradients = kwargs["accepting_gradients"]

    step_counter.reset()
    accepting_gradients.set(True)

    for _ in range(timesteps // T):
        for _ in range(M):
            while worker_counter.get() < W - D:
                time.sleep(.1)

            accepting_gradients.set(False)

            for name, param in shared_model.named_parameters():
                # print("%s: mean %f, std %f" % (name, param.data.mean(), param.data.std()))
                # print("%s.grad: mean %f, std %f" % (name, param.grad.data.mean(), param.grad.data.std()))
                param.grad.data.div_(worker_counter.get())
            
            optimizer.step()

            shared_model.zero_grad()
            worker_counter.reset()
            step_counter.increment()
            accepting_gradients.set(True)

def ppo_objective(
        action_taken_probs,
        action_taken_probs_old,
        advantages,
        clip=.2):
        advantages = (advantages - advantages.mean()) / advantages.std()

        ratio = action_taken_probs / (action_taken_probs_old + 1e-8)

        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, min=1. - clip, max=1. + clip) * advantages
        return (torch.min(unclipped, clipped)).mean()
