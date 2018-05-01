import random
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

N_actions = 5
N_unit_roles = 7
instrs = ["CON", "DISCON", "ADDUNIT", "DELUNIT", "NOOP"]

class DesignerNetwork(nn.Module):

    def __init__(self, N, I, O, S_example=10, S_phi=40, S_rho=40):
        super().__init__()

        self.N, self.I, self.O = N, I, O
        self.S_example, self.S_phi, self.S_rho = S_example, S_phi, S_rho

        self.gru_forward = nn.GRU(S_rho, S_phi)
        self.fc_forward = nn.Sequential(
            nn.Linear(S_phi + S_example * 2, S_rho),
            nn.Tanh())

        self.gru_backward = nn.GRU(S_rho, S_phi)
        self.fc_backward = nn.Sequential(
            nn.Linear(S_phi + S_example * 2, S_rho))

        # global actor-critic outputs
        self.fc_actor = nn.Linear(S_rho * 2, N_actions)
        self.fc_critic = nn.Linear(S_rho * 2, 1)

        # unit role probabilities
        self.fc_units = nn.Linear(S_rho * 2, N_unit_roles)

    def forward(self, x):
        rho_forward = {}
        rho_backward = {}

        M = None

        # 'forward' pass
        for node in sorted(x.nodes(data=True), key=lambda x: x[0]):
            i, props = node

            # get minibatch size (should be 1)
            if not M:
                M = props["dz"].size()[0]

            F_i = sorted([edge[0] for edge in x.in_edges([i])])
            if len(F_i) > 0:
                input = torch.stack([rho_forward[j] for j in F_i], 0)
                _, phi_forward_i = self.gru_forward(input)
                phi_forward_i = phi_forward_i.squeeze(0)
            else:
                phi_forward_i = torch.zeros(M, self.S_phi)

            input = torch.cat([
                phi_forward_i,
                props["z"],
                props["dz"]], 1)
            rho_forward[i] = self.fc_forward(input)

        # 'backward' pass
        for node in sorted(x.nodes(data=True), key=lambda x: x[0], reverse=True):
            j, props = node

            B_j = sorted([edge[1] for edge in x.out_edges([j])], reverse=True)
            if len(B_j) > 0:
                input = torch.stack([rho_backward[i] for i in B_j], 0)
                _, phi_backward_j = self.gru_backward(input)
                phi_backward_j = phi_backward_j.squeeze(0)
            else:
                phi_backward_j = torch.zeros(M, self.S_phi)

            input = torch.cat([
                phi_backward_j,
                props["z"],
                props["dz"]], 1)
            rho_backward[j] = self.fc_backward(input)

        N, I, O = self.N, self.I, self.O

        # calculate final forward and backward representations
        input = torch.stack([rho_forward[i] for i in range(N - O, N)], 0)
        _, alpha_forward = self.gru_forward(input)
        alpha_forward = alpha_forward.squeeze(0)
        input = torch.stack([rho_backward[i] for i in range(I - 1, -1, -1)], 0)
        _, alpha_backward = self.gru_backward(input)
        alpha_backward = alpha_backward.squeeze(0)

        # global results
        input = torch.cat([alpha_forward, alpha_backward], 1)
        omega = self.fc_actor(input)
        v = self.fc_critic(input)

        # unit results
        psi = [None] * N
        for i in range(N):
            if i in rho_forward.keys():
                input = torch.cat([rho_forward[i], rho_backward[i]], 1)
                psi[i] = self.fc_units(input)
            else:
                psi[i] = torch.ones(M, N_unit_roles) * -60.

        instr_probs = Fnn.softmax(omega, dim=1)
        role_probs = Fnn.softmax(torch.stack(psi, 1), dim=1)

        return (instr_probs, role_probs), v

    """ Choose an action ϵ-greedily from instruction and unit role probabilities """
    def choose_action(self, action_probs, epsilon=0.):
        instr_probs, role_probs = action_probs
        instr_probs, role_probs = instr_probs.squeeze(0), role_probs.squeeze(0)

        _, instr = torch.max(instr_probs, 0)
        instr = instr.detach().numpy()
        instr = instrs[instr]

        _, roles = torch.max(role_probs, 0)
        roles = roles.detach().numpy()

        if random.random() < epsilon:
            temp = role_probs[:, 0].detach().numpy()
            units = [i for i in range(self.N) if temp[i] > 1e-20]

            instr = random.choice(instrs)
            if instr in ["CON", "DISCON", "ADDUNIT"]:
                j = int(random.choice(units))
                i = int(random.choice([i for i in units if i >= j]))

                action = instr, j, i

            if instr == "DELUNIT":
                action = instr, int(random.choice(units))

            if instr == "NOOP":
                action = (instr,)

        else:
            if instr == "CON":
                action = instr, int(roles[0]), int(roles[1])

            if instr == "DISCON":
                action = instr, int(roles[2]), int(roles[3])

            if instr == "ADDUNIT":
                action = instr, int(roles[4]), int(roles[5])

            if instr == "DELUNIT":
                action = instr, int(roles[6])

            if instr == "NOOP":
                action = (instr,)

        return action

    """ The probability of taking an action given probabilities """
    def prob_action(self, probs, action):
        instr_probs, role_probs = probs
        instr_probs, role_probs = instr_probs.squeeze(0), role_probs.squeeze(0)

        instr = action[0]

        if instr in ["CON", "DISCON", "ADDUNIT"]:
            _, j, i = action

            if instr == "CON":
                return instr_probs[0] * role_probs[j, 0] * role_probs[i, 1]

            if instr == "DISCON":
                return instr_probs[1] * role_probs[j, 2] * role_probs[i, 3]

            if instr == "ADDUNIT":
                return instr_probs[2] * role_probs[j, 4] * role_probs[i, 5]

        if instr == "DELUNIT":
            _, i = action
            return instr_probs[3] * role_probs[i, 6]

        if instr == "NOOP":
            return instr_probs[4]
