import torch
import torch.optim as opt
import matplotlib.pyplot as plt

from functools import reduce
from torch.optim.LBFGS import FullBatchLBFGS
from torch.optim.optimizer import Optimizer
import data_logger


def is_legal(v):
    legal = not torch.isnan(v).any() and not torch.isinf(v)
    return legal


class fminunc():

    def __init__(self, objective, params, debug=True, lr=1,
                 maxIter=2000, grad_tol=1E-6, change_tol=1E-8, patience=25, max_ls=20, print_progress=0, logger_func=None, state_dict=None):

        self.debug = debug
        self.print_progress = print_progress

        # functions for evaluating objective, validation and test performance
        self.objective = objective
        self.params = params

        self.state_dict = state_dict

        # Tolerances and number of iterations
        self.grad_tol = grad_tol
        self.change_tol = change_tol
        self.maxIter = maxIter
        self.patience = patience
        self.max_ls = max_ls
        self.lr = lr

        self.callback = None

        self.data_log = data_logger.data_logger()
        if logger_func is not None:
            self.logger_func = logger_func
        else:
            self.logger_func = None

    def flatten_grad(self):
        views = []
        for p in self.params:
            if p.grad is None:
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                view = p.grad.to_dense().view(-1)
            else:
                view = p.grad.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    def optimize(self):

        using_pytorch_optimizer = True

        optimizer = torch.optim.Adam(self.params, self.lr)

        if self.state_dict is not None:
            # Update learning rate and load state dict
            self.state_dict["param_groups"][0]["lr"] = self.lr
            # optimizer.load_state_dict(self.state_dict)  # uncomment this to avoid resetting ADAM's state each time this function is called

        insufficient_progress = 0
        current_loss = self.objective()

        # print("Starting Optimizer")
        status = "Max Iterations"
        for kk in range(self.maxIter):

            def closure():
                optimizer.zero_grad()
                J = self.objective()
                J.backward(retain_graph=True)
                return J

            # Minimization options - damping and Armijo do not appear to work
            options = {'closure': closure, 'max_ls': self.max_ls,
                       'ls_debug': self.debug, 'interpolate': False, 'eta': 1.8, 'damping': False}

            prev_loss = current_loss

            if not using_pytorch_optimizer:
                current_loss, t, ls_steps, _, _, fail = optimizer.step(options)
                if ls_steps == 0:
                    optimizer.param_groups[0]["lr"] *= 1.2
                    print("\t\tNot enough ls steps. increasing learning rate (ls = ", ls_steps, "), lr = ", optimizer.param_groups[0]["lr"])

                if ls_steps > 15:
                    optimizer.param_groups[0]["lr"] /= 1.2
                    print("\t\ttoo many ls steps. decreasing learning rate (ls = ", ls_steps, "), lr = ", optimizer.param_groups[0]["lr"])

            else:
                current_loss = optimizer.step(closure)

            # Logs the data using whatever self contained callback the user defines
            if self.logger_func is not None:
                entry = self.logger_func()
                self.data_log.add_entry(entry)


            if not is_legal(current_loss):
                print("Optimization failed. Restart optim")
                status = "illegal"
                return current_loss, status

            #  count how many times we fail to progress in a row
            if (current_loss - prev_loss).abs() < self.change_tol:
                insufficient_progress += 1
            else:
                insufficient_progress = 0

            if insufficient_progress >= self.patience:
                print("\tUnable to make progress.")
                status = "Unable To Make any more Progress"
                break

            if self.print_progress != 0:
                if kk % self.print_progress == 0:
                    l = self.objective()
                    # print("\tIteration: {0:4d} \t J={1:1.2E}, \t |g|={2:1.2E}".format(kk + 1, float(l), float(grad.abs().max())))
                    print("\tIteration: {0:4d} \t J={1:1.6E}".format(kk + 1, float(l)))

                    if self.callback is not None:
                        self.callback()

        print("\tlbfgs: iterations:", kk + 1, "/", self.maxIter)

        state_dict = optimizer.state_dict()
        return current_loss, status, state_dict
