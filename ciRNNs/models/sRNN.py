import torch
from torch import nn
from torch.nn import Parameter

from typing import List
from torch import Tensor

import cvxpy as cp
import numpy as np


class sRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, nl=None, batches=1, learn_init_hidden=True, criterion=None):
        super(sRNN, self).__init__()

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

        # metric
        E0 = torch.eye(hidden_size)
        self.E = nn.Parameter(E0)
        P0 = torch.ones(hidden_size)
        self.P = nn.Parameter(P0)

        # dynamic layers
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

        seq_len = inputs.size(1)
        E_inv = self.E.inverse()
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

    def project_dl2(self, gamma):

        n = self.nx
        # Current values
        E_star = self.E.detach().numpy()
        H_star = self.H.weight.detach().numpy()
        B_star = self.K.weight.detach().numpy()
        C_star = self.output_layer.weight.detach().numpy()

        E = cp.Variable((n, n), 'E')
        H = cp.Variable((n, n), 'H')
        B = cp.Variable((n, 1), 'B')
        C = cp.Variable((1, n), 'C')
        P = cp.diag(cp.Variable((n, 1), 'P'))

        M = cp.bmat([[E + E.T - P, np.zeros((n, 1)), H.T, C.T], [np.zeros((1, n)), [[gamma]], B.T, [[0.0]]],
                     [H, B, P, np.zeros((n, 1))], [C, [[0.0]], np.zeros((1, n)), [[gamma]]]])

        # M = cp.bmat([[E + E.T - P - np.eye(n), np.zeros((n, 1)), H.T], [np.zeros((1, n)), [[gamma ** 2]], B.T], [H, B, P]])

        Id2 = np.eye(M.shape[0])

        constraints = [M >> 1E-6 * Id2]
        # objective = cp.norm(E - E_star, "fro") + cp.norm(H - H_star, "fro") + cp.norm(B - B_star, "fro") + cp.norm(C - C_star, "fro")
        objective = cp.norm(E - E_star, "fro") + cp.norm(H - H_star, "fro") + cp.norm(B - B_star, "fro") + cp.norm(C - C_star, "fro")
        # objective = cp.norm(E @ A_star - H, "fro") ** 2

        prob = cp.Problem(cp.Minimize(objective), constraints)
        # prob.solve(verbose=False, solver=cp.MOSEK)
        prob.solve(verbose=False)

        if prob.status in ["infeasible", "unbounded"]:
            print("Unable to solve problem")

        print('projecting onto stable set')

        # Reassign the values after projecting
        self.E.data = torch.Tensor(E.value)
        self.P.data = torch.Tensor(P.value.diagonal())
        self.H.weight.data = torch.Tensor(H.value)
        self.K.weight.data = torch.Tensor(B.value)
        self.output_layer.weight.data = torch.Tensor(C.value)

    def project_passive(self):

        n = self.nx
        # Current values
        E_star = self.E.detach().numpy()
        H_star = self.H.weight.detach().numpy()
        B_star = self.K.weight.detach().numpy()
        C_star = self.output_layer.weight.detach().numpy()

        E = cp.Variable((n, n), 'E')
        H = cp.Variable((n, n), 'H')
        B = cp.Variable((n, 1), 'B')
        C = cp.Variable((1, n), 'C')
        P = cp.diag(cp.Variable((n, 1), 'P'))

        M = cp.bmat([[E + E.T - P, 0.5 * C.T, H.T], [0.5 * C, np.zeros((1, 1)), B.T],
                    [H, B, P]])

        Id2 = np.eye(M.shape[0])

        constraints = [M >> 1E-6 * Id2]
        objective = cp.norm(E - E_star, "fro") + cp.norm(H - H_star, "fro") + cp.norm(B - B_star, "fro") + cp.norm(C - C_star, "fro")

        prob = cp.Problem(cp.Minimize(objective), constraints)
        prob.solve(verbose=False)

        if prob.status in ["infeasible", "unbounded"]:
            print("Unable to solve problem")

        print('projecting onto stable set')

        # Reassign the values after projecting
        self.E.data = torch.Tensor(E.value)
        self.P.data = torch.Tensor(P.value.diagonal())
        self.H.weight.data = torch.Tensor(H.value)
        self.K.weight.data = torch.Tensor(B.value)
        self.output_layer.weight.data = torch.Tensor(C.value)

    def project_l2(self):

        n = self.nx
        # Current values
        E_star = self.E.detach().numpy()
        H_star = self.H.weight.detach().numpy()

        E = cp.Variable((n, n), 'E')
        H = cp.Variable((n, n), 'H')
        P = cp.diag(cp.Variable((n, 1), 'P'))
        # P = cp.Variable((n,n), 'P')

        M = cp.bmat([[E + E.T - P - 0.2 * np.eye(n), H.T], [H, P]])

        Id1 = np.eye(n)
        Id2 = np.eye(2 * n)

        constraints = [M >> 1E-1 * Id2, E >> 0.1 * Id1, P >> 2E-2 * Id1]
        objective = cp.norm(E - E_star, "fro") + cp.norm(H - H_star, "fro")
        # objective = cp.norm(E @ A_star - H, "fro") ** 2

        prob = cp.Problem(cp.Minimize(objective), constraints)
        # prob.solve(verbose=False, solver=cp.MOSEK)
        prob.solve(verbose=False)

        if prob.status in ["infeasible", "unbounded"]:
            print("Unable to solve problem")

        print('projecting onto stable set')

        # Reassign the values after projecting
        self.E.data = torch.Tensor(E.value)
        self.P.data = torch.Tensor(P.value.diagonal())
        self.H.weight.data = torch.Tensor(H.value)
