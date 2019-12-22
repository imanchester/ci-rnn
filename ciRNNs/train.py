import torch
import time
# import opt.nlsdp as nlsdp
import opt.stochastic_nlsdp as nlsdp

default_log_dir = "./logs"
solve_tol = 1E-5
patience = 5


def train_model(model, train_loader, test_loader, val_loader, obj=None, LMIs=None, regularizers=None, log_dir=default_log_dir, options=None):

    print("Beginnning training at {}".format(time.ctime()))

    if obj is None:
        obj = torch.nn.MSELoss()

    if options is None:
        options = nlsdp.make_stochastic_nlsdp_options()

    problem = nlsdp.stochastic_nlsdp(model=model, criterion=obj, train_loader=train_loader, val_loader=val_loader, **options)

    if LMIs is not None:
        for lmi in LMIs:
            problem.addSDPconstraint(lmi)

    if regularizers is not None:
        for reg in regularizers:
            problem.add_regularizer(reg)

    log = problem.solve()

    print("Training Complete")
    return log
