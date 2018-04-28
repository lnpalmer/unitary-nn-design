import random
import torch
import torch.nn as nn
import torch.optim as optim

from dagnn import DAGNN

primary_network = DAGNN(80, 40, 1)
for _ in range(20):
    primary_network.addunit(random.randint(0, 40), 79, 0, random.random(), random.random())

primary_network.gen_parameters()
optimizer = optim.SGD(primary_network.parameters(), lr=2e-1)

B = 1000

x = torch.randn(B, 40)
y = x.mean(1).unsqueeze(1)

for i in range(1000):
    primary_network.zero_grad()

    y_hat = primary_network(x)
    loss = (((y_hat - y) ** 2) * .5).sum() / B

    loss.backward()
    optimizer.step()
    print(loss.detach().numpy())

def a(n):
    print(n)

for _ in range(5):
    print(a())
