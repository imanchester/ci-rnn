import torch
from torch import nn
import torch.nn.functional as F
import torch.jit as jit
from torch.nn import Parameter

from typing import List
from torch import Tensor

import cvxpy as cp
import numpy as np


class vRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batches=1, learn_init_hidden=True, criterion=None):
        super(vRNN, self).__init__()

        self.nu = input_size
        self.nx = hidden_size

        # Learnable Initial state?
        if learn_init_hidden:
            self.h0 = Parameter(torch.rand(1, batches, self.nx))
            self.c0 = Parameter(torch.rand(1, batches, self.nx))
        else:
            self.h0 = torch.rand(1, batches, self.nx)
            self.c0 = torch.rand(1, batches, self.nx)

        if criterion is None:
            self.criterion = torch.nn.MSELoss()
        else:
            self.criterion = criterion

        self.recurrent_unit = torch.nn.RNN(input_size, hidden_size, nonlinearity="relu", batch_first=True)

        U, S, V = self.recurrent_unit.weight_ih_l0.svd()
        self.recurrent_unit.weight_ih_l0.data = U @ V.T

        self.output = torch.nn.Linear(hidden_size, output_size)

    def forward(self, u, h0=None):
        # Add a new dimension corresponding to the number of hidden layers
        b = u.size(0)
        if h0 is None:
            h0 = self.h0[:, 0:b]

        init_state = h0
        states, hn = self.recurrent_unit(u, init_state)
        y = self.output(states)
        return y, states

    # returns a flattened tensor of all parameters
    def flatten_params(self):
        views = []
        for p in self.parameters():
            if p is None:
                view = p.new(p.numel()).zero_()
            elif p.is_sparse:
                view = p.to_dense().view(-1)
            else:
                view = p.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def flatten_grad(self):
        views = []
        for p in self.decVars:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    # Used for testing a model. Data should be a dictionary containing keys 
    #   "inputs" and "outputs" in order (batches x seq_len x size)
    def test(self, data, h0=None):

        self.eval()
        with torch.no_grad():
            u = data["inputs"]
            y = data["outputs"]

            if h0 is None:
                h0 = self.h0

            yest, states = self.forward(u, h0=h0)

            ys = y - y.mean(1).unsqueeze(2)
            error = yest - y
            NSE = error.norm() / ys.norm()
            results = {"SE": float(self.criterion(y, yest)),
                       "NSE": float(NSE),
                       "estimated": yest.detach().numpy(),
                       "inputs": u.detach().numpy(),
                       "true_outputs": y.detach().numpy(),
                       "hidden_layers": self.nx,
                       "model": "L2_reg"}
        return results
