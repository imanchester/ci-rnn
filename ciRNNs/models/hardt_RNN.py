import torch
from torch import nn
from torch.nn import Parameter

from typing import List
from torch import Tensor

import cvxpy as cp
import numpy as np


class hRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nl=None, batches=1, learn_init_hidden=True, criterion=None):
        super(hRNN, self).__init__()

        self.nx = hidden_size
        self.nu = input_size
        self.ny = output_size
        self.nBatches = batches

        # Initial state.
        if learn_init_hidden:
            self.h0 = Parameter(torch.rand(batches, self.nx))
        else:
            self.h0 = torch.rand(batches, self.nx)

        if criterion is None:
            self.criterion = torch.nn.MSELoss()
        else:
            self.criterion = criterion

        #  nonlinearity
        if nl is None:
            self.nl = torch.nn.ReLU
        else:
            self.nl = nl

        self.output_layer = torch.nn.Linear(hidden_size, output_size)

        # dynamic layers

        # This is similar to the sRNN except we remove E as a parameter and set it to identity
        self.E = torch.eye(hidden_size, requires_grad=False)
        # self.E = nn.Parameter(E0)
        self.K = nn.Linear(input_size, hidden_size, bias=False)
        self.H = nn.Linear(hidden_size, hidden_size)

        #  This initialization seems to work well.
        #  I.e. set the singular values to 1
        # U, S, V = self.H.weight.svd()
        # self.H.weight.data = U @ V.T

        self.H.weight.data = torch.eye(hidden_size)
        # self.H.bias.data = 1 * torch.zeros(hidden_size)

    # @jit.script_method
    def forward(self, inputs, h0=None):

        b = inputs.size(0)
        #  Initial state
        if h0 is None:
            ht = self.h0[0:b]
        else:
            ht = h0

        E_inv = self.E.inverse()
        seq_len = inputs.size(1)
        outputs = torch.jit.annotate(List[Tensor], [ht])
        for tt in range(seq_len - 1):
            xt = self.H(ht) + self.K(inputs[:, tt, :])
            eh = self.nl(xt)
            ht = eh.matmul(E_inv)
            outputs += [ht]

        states = torch.stack(outputs, 1)
        yest = self.output_layer(states)

        return yest, states

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
                       "model": "lstm"}
        return results

    # penalty function for equality constraints
    def equality_penalty(self, states):
        return torch.norm(states[:, -1, :] - self.h0, "fro")**2

    def penalty_l2(self):
        n = self.nx

        E = self.E
        H = self.H.weight
        P = torch.diag(self.P)

        Id1 = torch.eye(n)

        M1x = torch.cat((E + E.T - P, H.T), 1)
        M2x = torch.cat((H, P), 1)
        M = torch.cat((M1x, M2x), 0)

        def pd_penalty(M, epsilon):
            n = M.size(0)
            eigs = M.eig()

            # min_eig = eigs[0][:, 0].min()

            # if min_eig < epsilon:
            #     return min_eig.abs()

            if (eigs[0][:, 0] <= 0).any():
                # print("penalty")
                return (M - torch.eye(n) * epsilon).det()**2

            return 0

        return pd_penalty(E + E.T, 0.5) + pd_penalty(M, 1E-3) +\
            pd_penalty(P, 1E-2) + pd_penalty(500 * Id1 - E - E.T, 0)

    def barrier_l2(self):
        n = self.nx

        E = self.E
        H = self.H.weight
        P = torch.diag(self.P)

        Id1 = torch.eye(n)

        M1x = torch.cat((E + E.T - P, H.T), 1)
        M2x = torch.cat((H, P), 1)
        M = torch.cat((M1x, M2x), 0)

        def pd_barrier(M, epsilon):
            n = M.size(0)
            eigs = M.eig()

            if (eigs[0][:, 0] <= 0).any():
                return torch.tensor(float("inf"))

            return -torch.logdet(M - torch.eye(n) * epsilon)

        return pd_barrier(E + E.T, 0.5) + pd_barrier(M, 1E-3) +\
            pd_barrier(P, 1E-2) + pd_barrier(500 * Id1 - E - E.T, 0)

    def project_l2(self):

        n = self.nx
        # Current values
        H_star = self.H.weight.detach().numpy()

        H = cp.Variable((n, n), 'H')

        M = cp.bmat([[np.eye(n), H.T], [H, np.eye(n)]])

        Id2 = np.eye(2 * n)

        constraints = [M >> 1E-1 * Id2]
        objective = cp.norm(H - H_star, "fro")

        prob = cp.Problem(cp.Minimize(objective), constraints)
        # prob.solve(verbose=False, solver=cp.MOSEK)
        prob.solve(verbose=False)

        if prob.status in ["infeasible", "unbounded"]:
            print("Unable to solve problem")

        print('projecting onto stable set')

        # Reassign the values after projecting
        self.H.weight.data = torch.Tensor(H.value)
