import torch
import torch.nn as nn
import torch.functional as F
# import torch.optim as opt
import lbfgs_local as opt

import matplotlib.pyplot as plt
import numpy as np


def new_var(x):
    return nn.Parameter(torch.tensor(x))


# Call after solving inner iteration with LBFGS to print solution statistics to screen
def print_lbfgs_stats(optimizer):
    state = optimizer.state[optimizer._params[0]]
    iters = state.get('n_iter')
    g_max = state.get('prev_flat_grad').abs().max()

    print('\t\t    inner_iter:{0:3d}, max_g {1:1.3e}'.format(iters, g_max))


class problem():
    # decVars should be a list of parameter vectors and ceq cineq should be lists of functions
    def __init__(self, objective, decVars, ceq=[], cineq=[]):
        self.objective = objective
        self.equConstraints = ceq
        self.cineq = cineq

        #  if decVars is a list of tensors, convert to tensor
        self.decVars = decVars
        self.innerLoopCallBack = None

    # evaluates equalit constraints as a vector
    def ceq(self):
        if self.equConstraints.__len__() == 0:
            return torch.tensor([0.0])

        views = []
        for c in self.equConstraints:
            views.append(c())

        return torch.cat(views, 0)

    def flatten_params(self):
        views = []
        for p in self.decVars:
            if p is None:
                view = p.new(p.numel()).zero_()
            elif p.is_sparse:
                view = p.grad.to_dense().view(-1)
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

    # Adds the SDP constraint Q > 0
    def addSDPconstraint(self, Qf):
        Q = Qf()
        n = Q.size(1)
        p = n * (n + 1) // 2  # number of decision variables

        params = torch.nn.Parameter(torch.rand((p)))
        self.decVars += [params]

        # equality Constraints
        def c():

            lower_indices = torch.tril(torch.ones(n, n))
            L = torch.zeros(n, n)
            L[lower_indices == 1] = params

            return (L @ L.T - Qf()).view((-1, 1))

        self.equConstraints += [c]

    def solve(self, method='AL', plot_prog=True, eta_tol=1E-6, omega_tol=1E-4):

        # Initial Parameters
        muk = 10.0  # penalty parameter
        etak = 1 / muk**0.1  # solver accuracy
        omegak = 1 / muk

        fig, (ax1, ax2) = plt.subplots(2, 1)
        plt.ion()

        Jk = []
        Ck = []
        Lk = []

        # Create Lagrange Multiplier
        c0 = self.ceq()
        multipliers = torch.rand(c0.size())
        multipliers.requires_grad = False

        def AugmentedLagrangian():
            optimizer.zero_grad()
            L = self.objective() - multipliers.T @ self.ceq() + 0.5 * muk * self.ceq().T @ self.ceq()

            L.backward()

            return L

        # KKT for equality constrained optimization
        def KKT():
            L = AugmentedLagrangian()
            grad = self.flatten_grad()

            ceqk = self.ceq()
            kktCondition = multipliers.T @ ceqk

            # Check KKT conditions for equality constrained problem
            if grad.abs().max() < omega_tol and ceqk.abs().max() <= eta_tol and \
                    kktCondition.abs().max() <= eta_tol:
                return True

            else:
                False

        #  Main Loop of Optimizer
        for kk in range(1000):

            # Create an optimizer to solve the inner loop. Adjust the required accuracy based on omega_k.
            # optimizer = opt.LBFGS(self.decVars, lr=1E-1, line_search_fn='strong_wolfe', history_size=100,
            #                       max_iter=10000, tolerance_grad=omegak, tolerance_change=1E-10)

            optimizer = opt.LBFGS(self.decVars, lr=1E-1, history_size=100,
                                  max_iter=10000, tolerance_grad=omegak, tolerance_change=1E-10)

            # step optimizer
            ob_val = optimizer.step(AugmentedLagrangian)

            if self.innerLoopCallBack is not None:
                self.innerLoopCallBack()
            # print some statistics for the inner iteration
            # print_lbfgs_stats(optimizer)

            #  check for sufficient decrease in constraint violation
            if self.ceq().abs().max() <= etak:
                if KKT():  # is the problem solved?
                    print()
                    print('Problem optimal within tolerance: |dLdx| <= {0:1.3e}  |c(x)| <= {1:1.3e}'.
                          format(omega_tol, eta_tol))
                    break

                # Update multipliers and tighten tolerences
                with torch.no_grad():
                    multipliers = multipliers - muk * self.ceq()
                    etak = etak / muk**0.9
                    omegak = omegak / muk

                    if (etak < eta_tol):
                        etak = eta_tol
                    if (omegak < omega_tol):
                        omegak = omega_tol

                print('Iter: {0:2d}, J = {1:1.3e} reducing tolerance \t ------>  eta = {2:1.3e},  omega = {3:1.3e}'.format(kk, float(ob_val), etak, omegak))

            #  If insufficient decrease in constraint violation, increase the penalty
            else:
                muk = 50 * muk
                etak = 1 / muk**0.1
                omegak = 1 / muk

                if muk < 1E8:
                    print('Iter: {0:2d}, J = {1:1.3e} Increasing penalty \t ------>  eta = {2:1.3e},  omega = {3:1.3e}, mu ={4:3.0e}'.format(kk, float(ob_val), etak, omegak, muk))
                else:
                    muk = 1E8
                    print('Iter: {0:2d}, J = {1:1.3e} Increasing penalty \t ------>  eta = {2:1.3e},  omega = {3:1.3e}, mu capped'.format(kk, float(ob_val), etak, omegak))

            Lk += [float(ob_val)]
            Jk += [float(self.objective())]
            Ck += [float(self.ceq().abs().max())]

            ax1.plot(Lk)
            ax1.plot(Jk)
            ax1.legend(['Augmented Objective', 'True Objective'])
            ax2.plot(np.log10(Ck))
            ax2.legend(['Constraint Satisfaction'])
            plt.show()
            plt.pause(0.001)



# #  initialize two new parameters
# x1 = new_var([2.0])
# x2 = new_var([1.0])
# varList = [x1, x2]

# #  solve the problem min f s.t. c=0
# def objective():
#     return x1 - x2

# def ceq():
#     c = x1**2 + x2**2 - 2
#     return c.view([-1, 1])


# p = problem(objective, varList, ceq=ceq)

# p.solve()
# print(x1)
# print(x2)