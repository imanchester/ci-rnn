import torch
import numpy as np
import matplotlib.pyplot as plt


def is_legal(v):
    legal = not torch.isnan(v).any() and not torch.isinf(v)
    return legal


def plot_response(y, yest):
    plt.cla()
    for batch in range(y.size(0)):
        dt = y.size(2)
        t = np.arange(batch * dt, (batch + 1) * dt)
        plt.plot(t, y[batch].T.detach().numpy(), 'k')
        plt.plot(t, yest[batch].T.detach().numpy(), 'r')
    plt.show()
    plt.pause(0.01)


def make_stochastic_nlsdp_options(max_epochs=100, lr=1E-3, lr_decay=0.95, mu0=10, patience=20):
    options = {"max_epochs": max_epochs, "lr": lr, "tolerance_constraint": 1E-6, "debug": False,
               "patience": patience, "omega0": 1E-2, "eta0": 1E-2, "mu0": mu0, "lr_decay": lr_decay
               }
    return options


class stochastic_nlsdp():
    # decVars should be a list of parameter vectors and ceq cineq should be lists of functions
    def __init__(self, model, train_loader, val_loader, criterion=None, equ=None, max_epochs=1000, lr=1.0, max_ls=50,
                 tolerance_grad=1E-6, tolerance_change=1E-6, tolerance_constraint=1E-6, debug=False,
                 patience=10, omega0=1E-2, eta0=1E-1, mu0=10, lr_decay=0.95):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Add model parameters to list of decision variables
        self.decVars = list(self.model.parameters())

        self.criterion = criterion
        self.patience = patience
        self.lr = lr
        self.lr_decay = lr_decay
        self.max_ls = max_ls

        self.omega0 = omega0
        self.eta0 = eta0
        self.mu0 = mu0

        if equ is None:
            self.equConstraints = []
        else:
            self.equConstraints = equ

        self.max_epochs = max_epochs
        self.tolerance_constraint = tolerance_constraint

        self.tolerance_change = tolerance_change
        self.tolerance_grad = tolerance_grad

        self.LMIs = []
        self.regularizers = []

    # Evaluates the equality constraints c_i(x) as a vector c(x)
    def ceq(self):
        if self.equConstraints.__len__() == 0:
            return None

        views = []
        for c in self.equConstraints:
            views.append(c())

        return torch.cat(views, 0)

    # returns a flattened tensor of all parameters
    def flatten_params(self):
        views = []
        for p in self.decVars:
            if p is None:
                view = p.new(p.numel()).zero_()
            elif p.is_sparse:
                view = p.to_dense().view(-1)
            else:
                view = p.view(-1)
            views.append(view)
        return torch.cat(views, 0)

    # Returns the gradients as a flattened tensor
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

    # Adds the SDP constraint Qf > 0. Qf should be a function that returns the LMI
    # so that we want Qf()>0
    def addSDPconstraint(self, Qf):
        self.LMIs += [Qf]

        Q = Qf()
        n = Q.size(1)
        p = n * (n + 1) // 2  # number of decision variables

        # calculate the projection of Q onto pd matrices
        with torch.no_grad():

            # Project onto P.D. matrices
            e, V = Q.symeig(eigenvectors=True)
            e[e <= 1E-6] = 1E-6
            Qp = V @ e.diag() @ V.T

            #  initial value of L - if this fails, increase the minimum eigenvalues in e
            L_init = Qp.cholesky()

        # Dummy parameters to enforce Qf = L L'
        lower_indices = torch.tril(torch.ones(n, n))
        params = torch.nn.Parameter(torch.rand(p))
        params.data = L_init[lower_indices == 1]
        self.decVars += [params]

        # Add the nonlinear equality constraint
        def c(params=params, n=n, Qf=Qf):

            lower_indices = torch.tril(torch.ones(n, n))
            L = torch.zeros(n, n)
            L[lower_indices == 1] = params

            cons = (L @ L.T - Qf())
            return cons[lower_indices == 1]

        self.equConstraints += [c]

    def checkLMIs(self):
        lbs = []
        for lmi in self.LMIs:
            min_eval = lmi().eig()[0]
            lbs += [min_eval[0].min()]

        return lbs

    # Adds a regualizers reg where reg returns term that we woule like to regularize by
    def add_regularizer(self, reg):
        self.regularizers += [reg]

    # Evaluates the regularizers
    def eval_regularizers(self):

        if self.regularizers.__len__() == 0:
            return None

        res = 0
        for reg in self.regularizers:
            res += reg()

        return res

    #  eta tol is the desired tolerance in the constraint satisfaction
    #  omega_tol is the tolerance for first order optimality
    def solve(self):

        def validate(loader):
            total_loss = 0.0
            total_batches = 0

            self.model.eval()
            with torch.no_grad():
                for u, y in loader:
                    yest = self.model(u)
                    total_loss += self.criterion(yest, y) * u.size(0)
                    total_batches += u.size(0)

            return float(np.sqrt(total_loss / total_batches))

        # Initial Parameters
        muk = self.mu0  # penalty parameter
        eta0 = self.eta0  # solver accuracy for first subproblem

        no_decrease_counter = 0

        # Create Lagrange Multiplier - initialize as 0
        c0 = self.ceq()
        if c0 is not None:
            multipliers = 0.0 * torch.randn(c0.size()) / c0.size(0)
            multipliers.requires_grad = False
            satisfaction = c0.abs().max()

        else:
            satisfaction = 0.0

        with torch.no_grad():
            vloss = validate(self.val_loader)
            tloss = validate(self.train_loader)

            best_loss = vloss
            best_model = self.model.clone()

        log = {"val": [vloss], "training": [tloss], "satisfaction": [float(satisfaction)], "epoch": [0]}
        optimizer = torch.optim.Adam(params=self.decVars, lr=self.lr)

        #  Main Loop of Optimizer
        for epoch in range(self.max_epochs):

            #  --------------- Training Step ---------------
            train_loss = 0.0
            total_batches = 0
            self.model.train()
            for idx, (u, y) in enumerate(self.train_loader):

                def AugmentedLagrangian():
                    optimizer.zero_grad()

                    h0 = self.model.h0[idx: idx + 1]
                    yest = self.model(u, h0=h0)
                    L = self.criterion(y, yest)

                    train_loss = float(L) * u.size(0)

                    reg = self.eval_regularizers()
                    if reg is not None:
                        L += reg

                    c = self.ceq()
                    if c is not None:
                        L += -0*multipliers.T @ c + 0.5 * muk * c.T @ c
                        satisfaction = float(c.abs().max())
                    else:
                        satisfaction = 0.0

                    if not is_legal(train_loss + L):
                        print("illegal value encountered")

                    L.backward()

                    # for p in self.model.parameters():
                    #     p.grad[torch.isnan(p.grad.data)].data = torch.tensor(0.0)

                    # torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                    return L, train_loss, satisfaction

                Lag, t_loss, c = optimizer.step(AugmentedLagrangian)

                train_loss += t_loss
                total_batches += u.size(0)

                print("Epoch {:4d}: \t[{:04d}/{:04d}],\tlr = {:1.2e},\t avg loss: {:.5f},\t satisfaction {:.4f}".format(epoch,
                      idx + 1, len(self.train_loader), optimizer.param_groups[0]["lr"], train_loss / total_batches, c))

            # Reduce learning rate slightly after each epoch
            for param_group in optimizer.param_groups:
                param_group['lr'] *= self.lr_decay

            #  Check Constraints
            c = self.ceq()
            if c is not None:
                with torch.no_grad():
                    # If the constraints are still satisfied sufficiently
                    satisfaction = float(c.abs().max())
                    if satisfaction < eta0:
                        multipliers = multipliers + muk * c
                        print("multiplier update = {:1.3e}".format(muk * float(c.abs().max())))
                    #  If constraints are sufficiently violated, Increase penalty
                    else:
                        muk = 10 * muk if muk < 1E6 else 1E6

            else:
                satisfaction = 0.0

            # ---------------- Validation Step ---------------
            vloss = validate(self.val_loader)
            tloss = validate(self.train_loader)

            if vloss < best_loss:
                no_decrease_counter = 0
                best_loss = vloss
                best_model = self.model.clone()

            else:
                no_decrease_counter += 1

            log["val"] += [vloss]
            log["training"] += [tloss]
            log["epoch"] += [epoch]
            log["satisfaction"] += [satisfaction]

            print("-" * 120)
            print("Epoch {:4d}\t train_loss {:.4f},\tval_loss: {:.4f},\tsatisfaction: {:1.2e}".format(epoch, tloss, vloss, satisfaction))
            print("-" * 120)

            if no_decrease_counter > self.patience:
                break

        return log, best_model
