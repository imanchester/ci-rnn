import torch
import opt.fminunc as fminunc
import data_logger

#  Implementation of burer-Monteiro/cholesky method for semidefinite programming.
#  Reformulates the SDP as a nonlinear program.
#  It is not clear whether or not the nonlinear constraint Q = L L' introduce new local minima.

#  The resulting nonlinear program is solved using a Augmented Lagrangian method.
#  The inner optimization problem is solved using the file fminunc and defaults to LBFGS


def is_legal(v):
    legal = not torch.isnan(v).any() and not torch.isinf(v)
    return legal


# Call after solving inner iteration with LBFGS to print solution statistics to screen
def print_lbfgs_stats(optimizer):
    state = optimizer.state[optimizer._params[0]]
    iters = state.get('n_iter')
    g_max = state.get('prev_flat_grad').abs().max()

    print('\t\t    inner_iter:{0:3d}, max_g {1:1.3e}'.format(iters, g_max))


class nlsdp():
    # decVars should be a list of parameter vectors and ceq cineq should be lists of functions
    def __init__(self, objective, decVars, equ=None, outer_iter=10, inner_iter=2500, lr=1.0, max_ls=50,
                 tolerance_grad=1E-6, tolerance_change=1E-6, tolerance_constraint=1E-6, debug=False,
                 patience=50, callback=None, logger_func=None, omega0=1E-2, eta0=1E-1, mu0=10):

        self.debug = debug
        self.objective = objective
        self.decVars = list(decVars)
        self.patience = patience
        self.lr = lr
        self.max_ls = max_ls

        self.omega0 = omega0
        self.eta0 = eta0
        self.mu0 = mu0

        if equ is None:
            self.equConstraints = []
        else:
            self.equConstraints = equ

        if callback is not None:
            self.callback = callback

        self.logger_func = logger_func
        self.data_log = data_logger.data_logger()

        self.outer_iter = outer_iter
        self.inner_iter = inner_iter
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
    def solve(self, plot_prog=True):

        # Initial Parameters
        muk = self.mu0  # penalty parameter
        eta0 = self.eta0  # solver accuracy for first subproblem
        omega0 = self.omega0

        omegaf = self.tolerance_change
        etaf = self.tolerance_constraint

        omegar = (omegaf / omega0) ** (1.0 / self.outer_iter)
        etar = (etaf / eta0) ** (1.0 / self.outer_iter)

        state_dict = None

        # Create Lagrange Multiplier
        c0 = self.ceq()
        if c0 is not None:
            multipliers = torch.randn(c0.size()) / c0.size(0)
            multipliers.requires_grad = False

        #  Main Loop of Optimizer
        n_iter = 0
        while n_iter <= self.outer_iter:

            etak = eta0 * etar ** n_iter  # tolerance for constraints at outer iteration kk
            omegak = omega0 * omegar ** n_iter  # tolerance for objectiveat outer iteration kk

            def AugmentedLagrangian():
                L = self.objective()

                reg = self.eval_regularizers()
                if reg is not None:
                    L += reg

                c = self.ceq()
                if c is not None:
                    L += - multipliers.T @ c + 0.5 * muk * c.T @ c

                return L

            # Solve inner subproblem with accuracy of omegak
            problem = fminunc.fminunc(AugmentedLagrangian, self.decVars, maxIter=self.inner_iter, lr=self.lr * omegak / omega0, max_ls=self.max_ls,
                                      debug=self.debug, change_tol=omegak, patience=self.patience,
                                      print_progress=25, logger_func=self.logger_func, state_dict=state_dict)

            problem.callback = self.callback
            _, status, state_dict = problem.optimize()

            self.data_log.append_log(problem.data_log)

            if status == "illegal":
                return "illegal"

            if self.callback is not None:
                self.callback()

            #  check for sufficient decrease in constraint violation
            c = self.ceq()
            if c is not None:
                satisfaction = self.ceq().abs().max()
            else:
                satisfaction = 0.0

            if satisfaction <= etak:
                # update multipliers
                with torch.no_grad():
                    if c is not None:
                        multipliers = multipliers - muk * self.ceq()

                    n_iter += 1
                    J = AugmentedLagrangian()

                print("nlsdp:", n_iter, " - tightening tolerances")

                print("     eta = {0:1.3e} / [{1:1.3e}], J = {2:1.3e}, omega = {3:1.3e}"
                      .format(float(satisfaction), etak, float(J), omegak))
            #  If insufficient decrease in constraint violation, increase the penalty
            #  and solve again
            else:
                muk = 10 * muk
                print("nlsdp: increasing penalty. Satisfaction = {0:1.3e} / [{1:1.3e}]".format(float(satisfaction), etak))


def make_stochastic_nlsdp_options():
    options = {"outer_iter": 10, "inner_iter": 2500, "lr": 1.0, "max_ls": 50,
               "tolerance_grad": 1E-6, "tolerance_change": 1E-6, "tolerance_constraint": 1E-6, "debug": False,
               "patience": 10, "omega0": 1E-2, "eta0": 1E-1, "mu0": 10
               }
    return options


class stochastic_nlsdp():
    # decVars should be a list of parameter vectors and ceq cineq should be lists of functions
    def __init__(self, model, train_loader, val_loader, criterion=None, equ=None, outer_iter=10, inner_iter=2500, lr=1.0, max_ls=50,
                 tolerance_grad=1E-6, tolerance_change=1E-6, tolerance_constraint=1E-6, debug=False,
                 patience=10, omega0=1E-2, eta0=1E-1, mu0=10):

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = criterion
        self.patience = patience
        self.lr = lr
        self.max_ls = max_ls

        self.omega0 = omega0
        self.eta0 = eta0
        self.mu0 = mu0

        if equ is None:
            self.equConstraints = []
        else:
            self.equConstraints = equ

        self.outer_iter = outer_iter
        self.inner_iter = inner_iter
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

        # Initial Parameters
        muk = self.mu0  # penalty parameter
        eta0 = self.eta0  # solver accuracy for first subproblem

        # Create Lagrange Multiplier
        c0 = self.ceq()
        if c0 is not None:
            multipliers = torch.randn(c0.size()) / c0.size(0)
            multipliers.requires_grad = False

        # Create optimizer
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)

        #  Main Loop of Optimizer
        for epoch in range(self.maxEpochs):

            #  Training Step
            self.model.train()
            for idx, (u, y) in enumerate(self.train_loader):

                def AugmentedLagrangian():
                    optimizer.zero_grad()

                    yest = self.model(u)
                    L = self.criterion(y, yest)

                    reg = self.eval_regularizers()
                    if reg is not None:
                        L += reg

                    c = self.ceq()
                    if c is not None:
                        L += - multipliers.T @ c + 0.5 * muk * c.T @ c

                    L.backward()
                    return L

                optimizer.step(AugmentedLagrangian)

            #  Check Constraints
            c = self.ceq()
            if c is not None:

                # If the constraints are still satisfied sufficiently
                if c.abs().max() < eta0:
                    multipliers = multipliers - muk * c

                #  If constraints are sufficiently violated, Increase penalty
                else:
                    muk *= 10

