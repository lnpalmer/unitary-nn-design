import random
from math import floor
import torch

class BinaryProgram:

    def __init__(self, I, O, steps = []):
        self.I = I
        self.O = O
        self.steps = steps

    def run(self, x):
        O = self.O

        y = x.transpose(0, 1).contiguous()

        for step in self.steps:
            instr = step[0]
            if instr in ["XOR", "OR", "AND"]:
                _, i, j, k, l = step
                ir = y[i:i + l]
                jr = y[j:j + l]

                if instr == "XOR":
                    y[k:k + l] = ir + jr - ir * jr * 2

                if instr == "OR":
                    y[k:k + l] = ir + jr - ir * jr

                if instr == "AND":
                    y[k:k + l] = ir * jr

            if instr in ["NOT", "REVERSE"]:
                _, i, l = step

                if instr == "NOT":
                    y[i:i + l] = 1 - y[i:i + l]

                if instr == "REVERSE":
                    rev_idx = torch.arange((i + l) - 1, i - 1, -1).long()
                    y[i:i + l] = y[rev_idx]

        return y[0:O].transpose(0, 1).contiguous()

def gen_binary_program(I, O, nsteps):
    steps = []

    for _ in range(nsteps):
        instr = random.choice(["XOR", "OR", "AND", "NOT", "REVERSE"])

        if instr in ["XOR", "OR", "AND"]:
            l = random.randint(2, floor(I / 3))
            i, j, k = [random.randint(0, I - l) for _ in range(3)]

            step = instr, i, j, k, l

        if instr in ["NOT", "REVERSE"]:
            l = random.randint(2, floor(I / 3))
            i = random.randint(0, I - l)

            step = instr, i, l

        steps.append(step)

    return BinaryProgram(I, O, steps=steps)
