import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
import networkx as nx

from utils import interpolate, log_param_stats, pad_str

N_actions = 5
N_unit_roles = 7
instrs = ["CON", "DISCON", "ADDUNIT", "DELUNIT", "NOOP"]
role_names = ["CON FROM", "CON TO", "DISCON FROM", "DISCON TO", "ADDUNIT FROM", "ADDUNIT TO", "DELUNIT"]

def info_action_prob(action_prob, num_top_roles=5):
    instr_prob, role_prob = action_prob
    instr_prob, role_prob = instr_prob.squeeze(0).data.numpy(), role_prob.squeeze(0).data.numpy()
    print(", ".join(["%s: %f" % (instr, instr_prob[i]) for i, instr in enumerate(instrs)]))
    sort_args = np.argsort(role_prob, axis=1)
    for i, role_name in enumerate(role_names):
        print(pad_str("Top candidates for " + role_name + ":", 36) + np.array_str(sort_args[i,::-1][:num_top_roles])[1:-1])

def get_net_bias(net, i):
    if net.has_node(i):
        return net.nodes[i]['b']
    else:
        return 0.

def get_net_weight(net, j, i):
    if net.has_edge(j, i):
        return net.edges[j, i]['weight']
    else:
        return 0.

def get_net_z(net, i, S_example):
    if net.has_node(i):
        return net.nodes[i]['z']
    else:
        return torch.zeros(S_example)

def get_net_dz(net, i, S_example):
    if net.has_node(i):
        return net.nodes[i]['dz']
    else:
        return torch.zeros(S_example)

class DesignerNetwork(nn.Module):

    def __init__(self, N, I, O, S_example=10, S_phi=60, S_rho=60):
        super().__init__()

        self.N, self.I, self.O = N, I, O
        self.S_example, self.S_phi, self.S_rho = S_example, S_phi, S_rho

        self.gru_forward = nn.GRUCell(S_rho, S_phi)
        self.fc_forward = nn.Sequential(
            nn.Linear(S_phi + S_example * 2, S_rho),
            nn.Tanh())

        self.gru_backward = nn.GRUCell(S_rho, S_phi)
        self.fc_backward = nn.Sequential(
            nn.Linear(S_phi + S_example * 2, S_rho))

        # global actor-critic outputs
        self.fc_actor = nn.Linear(S_phi * 2, N_actions)
        self.fc_critic = nn.Linear(S_phi * 2, 1)

        # unit role probabilities
        self.fc_units = nn.Linear(S_rho * 2, N_unit_roles)

        for name, param in self.named_parameters():
            if param.dim() >= 2:
                nn.init.xavier_uniform_(param.data)
            if param.dim() == 1:
                nn.init.normal_(param.data)

    """
    Forward pass for the designer network.

    Args:
        x:
            A list (batch) of B observations.
            Observations are networkx.DiGraph instances, populated with a bias for each node where
            applicable and neural network connections as edges,
            with edge weights as the neural network weights.
    Returns: (instr_prob, role_prob), value
        instr_prob: [B x N_actions] instruction probabilities
        role_prob: [B x N_roles x number of units] unit role probabilities
        value: [B x 1] state value estimate
    """
    def forward(self, x):
        B = len(x)

        # take the union of the primary network graphs
        x_union = nx.DiGraph()
        for x_b in x:
            x_union.add_nodes_from(x_b.nodes())
            x_union.add_edges_from(x_b.edges())

        # batch node data
        for i in x_union.nodes():
            has_i = torch.Tensor([float(x[b].has_node(i)) for b in range(B)]).unsqueeze(1)
            b_i = torch.Tensor([get_net_bias(x[b], i) for b in range(B)]).unsqueeze(1)
            z_i = torch.stack([get_net_z(x[b], i, self.S_example) for b in range(B)], 0)
            dz_i = torch.stack([get_net_dz(x[b], i, self.S_example) for b in range(B)], 0)
            x_union.add_node(i, has=has_i, b=b_i, z=z_i, dz=dz_i)

        # batch edge data
        for edge in x_union.edges():
            j, i = edge
            w_ij = torch.Tensor([get_net_weight(x[b], j, i) for b in range(B)]).unsqueeze(1)
            has_ij = torch.Tensor([float(x[b].has_edge(j, i)) for b in range(B)]).unsqueeze(1)
            x_union.add_edge(j, i, w=w_ij, has=has_ij)

        # forward pass over the primary network
        for i in sorted(x_union.nodes()):
            node = x_union.nodes[i]
            b_i = node['b']
            z_i = node['z']
            dz_i = node['dz']
            F_i = sorted([edge[0] for edge in x_union.in_edges([i])])

            h = torch.zeros(B, self.S_phi)
            for F_i_t in F_i:
                node_F_i_t = x_union.nodes[F_i_t]
                has_F_i_t = node_F_i_t['has']
                rho_F_i_t = node_F_i_t['rho_forward']
                h = interpolate(h, self.gru_forward(rho_F_i_t, h), has_F_i_t)

            phi_i = h
            input = torch.cat([phi_i, z_i, dz_i], 1)
            rho_i = self.fc_forward(input)
            x_union.nodes[i]['rho_forward'] = rho_i

        # backward pass over the primary network
        for i in reversed(sorted(x_union.nodes())):
            node = x_union.nodes[i]
            b_i = node['b']
            z_i = node['z']
            dz_i = node['dz']
            B_i = reversed(sorted([edge[1] for edge in x_union.out_edges([i])]))

            h = torch.zeros(B, self.S_phi)
            for B_i_t in B_i:
                node_B_i_t = x_union.nodes[B_i_t]
                has_B_i_t = node_B_i_t['has']
                rho_B_i_t= node_B_i_t['rho_backward']
                h = interpolate(h, self.gru_backward(rho_B_i_t, h), has_B_i_t)

            phi_i = h
            input = torch.cat([phi_i, z_i, dz_i], 1)
            rho_i = self.fc_backward(input)
            x_union.nodes[i]['rho_backward'] = rho_i

        # final forward output
        h = torch.zeros(B, self.S_rho)
        for i in range(self.N - self.O, self.N):
            node_i = x_union.nodes[i]
            has_i = node_i['has']
            rho_i = node_i['rho_forward']
            h = interpolate(h, self.gru_forward(rho_i, h), has_i)
        alpha_forward = h

        # final backward output
        h = torch.zeros(B, self.S_rho)
        for i in reversed(range(self.I)):
            node_i = x_union.nodes[i]
            has_i = node_i['has']
            rho_i = node_i['rho_backward']
            h = interpolate(h, self.gru_backward(rho_i, h), has_i)
        alpha_backward = h

        # general action logits
        input = torch.cat([alpha_forward, alpha_backward], 1)
        omega = self.fc_actor(input)
        value = self.fc_critic(input)

        def get_has(i):
            if x_union.has_node(i):
                return x_union.nodes[i]['has']
            else:
                return torch.zeros(B, 1)

        def get_psi(i):
            if x_union.has_node(i):
                node_i = x_union.nodes[i]
                input = torch.cat([node_i['rho_forward'], node_i['rho_backward']], 1)
                return self.fc_units(input)
            else:
                return torch.zeros(B, N_unit_roles)

        has = torch.cat([get_has(i) for i in range(self.N)], 1)
        psi = torch.stack([get_psi(i) for i in range(self.N)], 2)

        # prevent usage of nonexistant units
        psi = interpolate(-60., psi, has.unsqueeze(1))

        instr_prob = Fnn.softmax(omega, dim=1)
        role_prob = Fnn.softmax(psi, dim=2)

        return (instr_prob, role_prob), value

    """ Choose an action ϵ-greedily from instruction and unit role probabilities """
    def choose_action(self, action_prob, epsilon=0.):
        instr_prob, role_prob = action_prob
        instr_prob, role_prob = instr_prob.squeeze(0), role_prob.squeeze(0)

        _, instr = torch.max(instr_prob, 0)
        instr = instr.detach().numpy()
        instr = instrs[instr]

        _, roles = torch.max(role_prob, 1)
        roles = roles.detach().numpy()

        if random.random() < epsilon:
            temp = role_prob[0, :].detach().numpy()
            units = [i for i in range(self.N) if temp[i] > 1e-10]

            instr = random.choice(instrs)
            if instr in ["CON", "DISCON", "ADDUNIT"]:
                j = int(random.choice(units))
                i_units = [i for i in units if i > j]
                if len(i_units) > 0:
                    i = int(random.choice(i_units))
                    action = instr, j, i
                else:
                    action = ("NOOP",)

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
    
    """
    Gets the probability of taking an action given probabilities

    Args:
        prob: (instr_prob, role_prob)
            instr_prob: [B x N_actions] instruction probabilities
            role_prob: [B x N_roles x number of units] unit role probabilities
        action: list of B actions sampled
    Returns:
        action_prob: [B] action probabilities
    """
    def prob_action(self, prob, action):
        instr_prob, role_prob = prob

        B = len(action)
        action_prob = [None] * B
        for b in range(B):
            instr = action[b][0]

            if instr in ["CON", "DISCON", "ADDUNIT"]:
                _, j, i = action[b]

                if instr == "CON":
                    action_prob[b] = instr_prob[b, 0] * role_prob[b, 0, j] * role_prob[b, 1, i]

                if instr == "DISCON":
                    action_prob[b] = instr_prob[b, 1] * role_prob[b, 2, j] * role_prob[b, 3, i]
                
                if instr == "ADDUNIT":
                    action_prob[b] = instr_prob[b, 2] * role_prob[b, 4, j] * role_prob[b, 5, i]
                
            if instr == "DELUNIT":
                _, i = action[b]
                action_prob[b] = instr_prob[b, 3] * role_prob[b, 6, i]
            
            if instr == "NOOP":
                action_prob[b] = instr_prob[b, 4]
        
        return torch.stack(action_prob, 0).unsqueeze(1)

    # """ The probability of taking an action given probabilities """
    # def prob_action(self, prob, action):
    #     instr_prob, role_prob = prob
    #     instr_prob, role_prob = instr_prob.squeeze(0), role_prob.squeeze(0)

    #     instr = action[0]

    #     if instr in ["CON", "DISCON", "ADDUNIT"]:
    #         _, j, i = action

    #         if instr == "CON":
    #             return instr_prob[0] * role_prob[0, j] * role_prob[1, i]

    #         if instr == "DISCON":
    #             return instr_prob[1] * role_prob[2, j] * role_prob[3, i]

    #         if instr == "ADDUNIT":
    #             return instr_prob[2] * role_prob[4, j] * role_prob[5, i]

    #     if instr == "DELUNIT":
    #         _, i = action
    #         return instr_prob[3] * role_prob[6, i]

    #     if instr == "NOOP":
    #         return instr_prob[4]
