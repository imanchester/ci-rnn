import numpy as np
import torch
import scipy.io as io
import sys as sys
import os

import opt.stochastic_nlsdp as nlsdp
import models.diRNN as diRNN
import models.dnb as dnb
import train as train
import data.load_data as load_data


import multiprocessing
multiprocessing.set_start_method('spawn', True)


# Returns the results of running model on the data in loader.
def test(model, loader):
    model.eval()

    length = loader.__len__()
    inputs = np.zeros((length,), dtype=np.object)
    outputs = np.zeros((length,), dtype=np.object)
    measured = np.zeros((length,), dtype=np.object)

    SE = np.zeros((length, model.ny))
    NSE = np.zeros((length, model.ny))

    with torch.no_grad():
        for idx, (u, y) in enumerate(loader):
            yest = model(u)
            inputs[idx] = u.numpy()
            outputs[idx] = yest.numpy()
            measured[idx] = y.numpy()

            error = yest[0].numpy() - y[0].numpy()
            mu = y[0].mean(1).numpy()
            N = error.shape[1]
            norm_factor = ((y[0].numpy() - mu[0, None])**2).sum(1)

            SE[idx] = (error ** 2 / N).sum(1) ** (0.5)
            NSE[idx] = ((error ** 2).sum(1) / norm_factor) ** (0.5)

    res = {"inputs": inputs, "outputs": outputs, "measured": measured, "SE": SE, "NSE": NSE}
    return res


def test_and_save_model(name, model, train_loader, val_loader, test_loader, log, params=None):

    nx = model.nx
    layers = model.layers
    path = "./experimental_results/gait_prediction/w{}_l{}/".format(nx, layers)
    file_name = name + '.mat'

    train_stats = test(model, train_loader)
    val_stats = test(model, val_loader)
    test_stats = test(model, test_loader)

    data = {"validation": val_stats, "training": train_stats, "test": test_stats, "nx": model.nx, "nu": model.nu, "ny": model.ny,
            "layers": model.layers, "training_log": log}

    if params is not None:
        data = {**data, **params}

    # Create target Directory if doesn't exist
    if not os.path.exists(path):
        os.mkdir(path)
        print("Directory ", path, " Created ")

    io.savemat(path + file_name, data)


if __name__ == '__main__':

    mu = 0.05
    eps = 1E-3
    ar = False
    max_epochs = 500
    patience = 20

    # layers = int(sys.argv[1])
    layers = 2
    print("Training models with {} layers".format(layers))
    width = 64

    for subject in range(2, 10):
        for val_set in range(0, 9):

            # Load the data set
            dataset_options = load_data.make_default_options(train_bs=1, train_sl=2048, val_bs=10, ar=ar, val_set=val_set)
            dataset_options["subject"] = subject
            train_loader, val_loader, test_loader = load_data.load_dataset(dataset="gait_prediction_stairs", dataset_options=dataset_options)

            nu = train_loader.nu
            ny = train_loader.ny

            # Options for the solver
            # solver_options = nlsdp.make_stochastic_nlsdp_options(max_epochs=max_epochs, lr=5.0E-4, mu0=100, lr_decay=0.98)
            solver_options = nlsdp.make_stochastic_nlsdp_options(max_epochs=max_epochs, lr=5.0E-4, mu0=500, lr_decay=0.987, patience=20)

            # Train Unconstrained model - still project onto stable models for stable initial point
            name = "uncon_sub{:d}_val{:d}".format(subject, val_set)
            model = dnb.dnbRNN(nu, width, ny, layers, nBatches=9, init_var=1.3)

            log, best_model = train.train_model(model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                                                options=solver_options)

            test_and_save_model(name, best_model, train_loader, val_loader, test_loader, log)

            # Optimize over models with weights confined to spectral norm ball (I.e. what Miller and Hardt did)
            name = "spectral_norm_sub{:d}_val{:d}".format(subject, val_set)
            model = dnb.dnbRNN(nu, width, ny, layers, nBatches=9, init_var=1.3)
            model.project_norm_ball(eps)
            log, best_model = train.train_model(model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                                                options=solver_options, LMIs=model.norm_ball_lmi(eps))

            test_and_save_model(name, best_model, train_loader, val_loader, test_loader, log)

            # Train Contracting model
            name = "contracting_sub{:d}_val{:d}".format(subject, val_set)
            model = diRNN.diRNN(nu, width, ny, layers, nBatches=9)
            # model.project_l2(mu=mu, epsilon=eps)
            model.init_l2(mu=mu, epsilon=eps, init_var=1.3)

            log, best_model = train.train_model(model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                                                options=solver_options, LMIs=model.contraction_lmi(mu=mu, epsilon=eps))

            test_and_save_model(name, best_model, train_loader, val_loader, test_loader, log)
