import random
import gym
from gym.envs.registration import register
import numpy as np
import torch
import torch.nn.functional as Fnn
import torch.optim as optim

from dagnn import gen_dagnn
from binary import gen_binary_program
from adam_quickfix import SparseAdamQuickfixed

# TODO cleanup name ambiguity: ACTION(i, j) vs. j, i from matrix standard

train_size = 50000
test_size = 10000
bp_step_range = 16, 16
lr=3e-5
M = 100
N_mb = 100
delta_warmup = 1e-2
S_example = 20 # TODO: rename

class DesignEnv(gym.Env):

    def __init__(self, N, I, O, H):
        self.N = N
        self.I = I
        self.O = O
        self.H = H

    def step(self, action):
        # apply the modification action to the network
        instr = action[0]
        if instr in ["CON", "DISCON", "ADDUNIT"]:
            _, i, j = action

            if instr == "CON":
                self.primary_network.connect(i, j, random.gauss(0, 1))

            if instr == "DISCON":
                self.primary_network.disconnect(i, j)

            if instr == "ADDUNIT":
                self.primary_network.addunit(
                    i,
                    j,
                    random.gauss(0, 1),
                    random.gauss(0, 1),
                    random.gauss(0, 1))

        if instr == "DELUNIT":
            _, i = action
            self.primary_network.delunit(i)

        # train for N_mb minibatches
        test_loss = self._train()

        # make observation of activations, gradients
        self.primary_network.zero_grad()
        idxs = np.arange(train_size)
        np.random.shuffle(idxs)
        idxs_ob = idxs[:S_example]
        x_ob = self.x_train[idxs_ob]
        y_ob = self.y_train[idxs_ob]
        a = self.primary_network(x_ob)

        # TODO: get z, dz
        ob = torch.zeros(1, 1)
        done = False
        reward = self._get_reward(test_loss)

        return ob, reward, done, None

    def reset(self):
        N, I, O, H = self.N, self.I, self.O, self.H

        # generate primary network
        self.primary_network = gen_dagnn(N, I, O, n_H=H)

        self.dense_optimizer = optim.Adam(self.primary_network.dense_parameters(), lr=lr)
        self.sparse_optimizer = SparseAdamQuickfixed(self.primary_network.sparse_parameters(), lr=lr)

        # generate binary program and dataset
        self.bp = gen_binary_program(I, O, random.randint(*bp_step_range))

        x_train = torch.randn(train_size, I) > .5
        y_train = self.bp.run(x_train)
        x_train, y_train = x_train.float(), y_train.float()

        x_test = torch.randn(test_size, I) > .5
        y_test = self.bp.run(x_test)
        x_test, y_test = x_test.float(), y_test.float()

        self.x_train, self.y_train, self.x_test, self.y_test = x_train, y_train, x_test, y_test

        self.prev_loss = None

        # train the network to a relatively stable point
        delta_loss = 1e9
        while abs(delta_loss) > delta_warmup:
            test_loss = self._train()
            delta_loss = -self._get_reward(test_loss)

    def _train(self):
        idxs = np.arange(train_size)
        np.random.shuffle(idxs)

        for mb in range(N_mb):
            self.primary_network.zero_grad()

            start = mb * M
            idxs_mb = idxs[start:start + M]
            x_mb = self.x_train[idxs_mb]
            y_mb = self.y_train[idxs_mb]

            loss = self._loss(x_mb, y_mb)
            loss.backward()
            self.dense_optimizer.step()
            self.sparse_optimizer.step()

        loss = self._loss(self.x_test, self.y_test)
        loss = loss.detach().numpy()

        return loss

    def _loss(self, x, y):
        one_probs = Fnn.sigmoid(self.primary_network(x))
        bit_probs = y * one_probs + (1. - y) * (1. - one_probs)
        loss = -bit_probs.log().sum() / x.size()[0]
        return loss

    def _get_reward(self, test_loss):
        if not self.prev_loss:
            reward = 1e9
        else:
            reward = - (test_loss - self.prev_loss)
        self.prev_loss = test_loss
        return reward

register(
    id="unitary-design-v0",
    entry_point="design_env:DesignEnv",
    kwargs={
        "N": 94,
        "I": 12,
        "O": 12,
        "H": 48})
