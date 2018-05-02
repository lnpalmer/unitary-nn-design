import torch.multiprocessing as mp
import gym

class DPPO:

    def __init__(self, **kwargs):
        self.env_name = kwargs["env_name"]
        self.model = kwargs["model"]
        self.clone_model = kwargs["clone_model"]
        self.optimizer = kwargs["optimizer"]
        self.T = kwargs["T"]
        self.W = kwargs["W"]

    def run(self, timesteps):
        processes = []
        for w in range(self.W):
            process = mp.Process(
                target=dppo_worker,
                kwargs={
                    "timesteps": timesteps,
                    "env_name": self.env_name,
                    "T": self.T,
                    "W": self.W,
                    "shared_model": self.model,
                    "clone_model": self.clone_model})
            process.start()
            processes.append(process)
        for process in processes:
            process.join()

def dppo_worker(**kwargs):
    timesteps = kwargs["timesteps"]
    env_name = kwargs["env_name"]
    T = kwargs["T"]
    W = kwargs["W"]
    shared_model = kwargs["shared_model"]
    clone_model = kwargs["clone_model"]

    model = clone_model(shared_model)
    old_model = clone_model(shared_model)

    env = gym.make(env_name)

    for _ in range(timesteps // T):

        obs = []
        rewards = []
        dones = []
        actions = []
        values = []

        done = True
        for i in range(T // W):
            if done:
                ob = env.reset()

            obs.append(ob)

            action_probs, v = model(ob)
            action = model.choose_action(action_probs, epsilon=.1)
            ob, reward, done, _ = env.step(action)

            actions.append(action)
            rewards.append(reward)
            print(i)

class PPOObjective:

    def __init__(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError
