import torch
from torch.autograd import Function

import dagnn_cpp

class DAGNNFunction(Function):

    @staticmethod
    def forward(ctx, x, W, b, i, o):
        a = dagnn_cpp.forward(x, W, b, i, o)
        ctx.save_for_backward(W, b, i, o, a)
        return a

    @staticmethod
    def backward(ctx, da):
        W, b, i, o, a = ctx.saved_variables
        dx, dW, db = dagnn_cpp.backward(W, b, i, o, a, da)
        return dx, dW, db, torch.zeros(1), torch.zeros(1)
