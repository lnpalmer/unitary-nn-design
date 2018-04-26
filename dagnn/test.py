from dagnn import DAGNNFunction
import torch

a = torch.ones(4, requires_grad=True)
scale = 5

W_i = torch.LongTensor([[2, 2, 3, 3],
                        [0, 1, 2, 1]])
W_v = torch.FloatTensor([.3, .4, .5, .6])
W = torch.sparse.FloatTensor(W_i, W_v, torch.Size([4, 4]))
W.requires_grad_()

b = torch.FloatTensor([0, 1, 0, 0])
b.requires_grad_(True)

i = torch.IntTensor([2])
o = torch.IntTensor([1])

x = torch.FloatTensor([[1, 1]])
x.requires_grad_(True)

a = DAGNNFunction.apply(x, W, b, i, o)

y = a[:, -1].unsqueeze(1)

print(a)
print(y)

y.backward(torch.FloatTensor([[1]]).transpose(0, 1))

print(b.grad)
print(W.grad)
