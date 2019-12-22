import numpy as np
import torch
import scipy.io as io
import sys as sys
import os
import matplotlib.pyplot as plt

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

def compare_exp_imp(eModel, iModel, loader):
    with torch.no_grad():
        for (u, y) in loader:
            yest1 = eModel(u)
            yest2 = iModel(u)
            plt.plot(y[0].T, 'k')
            plt.plot(yest1[0].T, 'b')
            plt.plot(yest2[0].T, 'r')
            plt.show()

def test_and_save_model(name, model, train_loader, val_loader, test_loader, log, params=None):

    nx = model.nx
    layers = model.layers
    path = "./experimental_results/chen_2/w{}_l{}/".format(nx, layers)
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

    mu = 0.01
    eps = 1E-2
    ar = False
    max_epochs = 500
    patience = 20

    layers = int(sys.argv[1])
    # layers = 3
    print("Training models with {} layers".format(layers))
    width = 200

    for subject in range(1, 50):
        for val_set in range(0, 20):

            # Load the data set
            dataset_options = load_data.make_default_options(train_bs=40, train_sl=200, val_set=val_set, test_sl=2000)
            dataset_options["gain"] = 1.4
            train_loader, val_loader, test_loader = load_data.load_dataset(dataset="chen", dataset_options=dataset_options)

            nu = train_loader.nu
            ny = train_loader.ny

            # Options for the solver
            solver_options = nlsdp.make_stochastic_nlsdp_options(max_epochs=max_epochs, lr=0.1E-4, mu0=2000, lr_decay=0.96, patience=10)

            # Train Contracting implicit model ------------------------------------------------------------------------
            name = "contracting_sub{:d}_val{:d}".format(subject, val_set)
            model = diRNN.diRNN(nu, width, ny, layers, nBatches=dataset_options["train_batch_size"] - 1)
            model.init_l2(mu=mu, epsilon=eps, init_var=1.2)

            log, best_model = train.train_model(model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                                                options=solver_options, LMIs=model.contraction_lmi(mu=mu, epsilon=eps))

            # eModel = best_model.make_explicit()
            # compare_exp_imp(eModel, best_model, train_loader)
            test_and_save_model(name, best_model, train_loader, val_loader, test_loader, log)

            # Train implicit model - Critical Initialization-----------------------------------------------
            name = "contracting_init_sub{:d}_val{:d}".format(subject, val_set)
            model = diRNN.diRNN(nu, width, ny, layers, nBatches=dataset_options["train_batch_size"] - 1)
            model.init_l2(mu=mu, epsilon=eps, init_var=1.2)

            log, best_model = train.train_model(model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                                                options=solver_options)

            test_and_save_model(name, best_model, train_loader, val_loader, test_loader, log)

            # Train implicit unconstrained model - normal Init---------------------------------
            name = "implicit_uncon_sub{:d}_val{:d}".format(subject, val_set)
            model = diRNN.diRNN(nu, width, ny, layers, nBatches=dataset_options["train_batch_size"] - 1)

            log, best_model = train.train_model(model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                                                options=solver_options)

            test_and_save_model(name, best_model, train_loader, val_loader, test_loader, log)

            # # Train Unconstrained model with identity initialization---------------------------------------
            # name = "identity_sub{:d}_val{:d}".format(subject, val_set)
            # model = dnb.dnbRNN(nu, width, ny, layers, nBatches=dataset_options["train_batch_size"], init_var=1.0)
            # model.identity_init()

            # log, best_model = train.train_model(model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
            #                                     options=solver_options)

            # test_and_save_model(name, best_model, train_loader, val_loader, test_loader, log)

            # Optimize over models with weights confined to spectral norm ball (I.e. what Miller and Hardt did)
            name = "spectral_norm_sub{:d}_val{:d}".format(subject, val_set)
            model = dnb.dnbRNN(nu, width, ny, layers, nBatches=dataset_options["train_batch_size"]-1, init_var=1.0)
            model.project_norm_ball(eps)
            log, best_model = train.train_model(model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                                                options=solver_options, LMIs=model.norm_ball_lmi(eps))

            test_and_save_model(name, best_model, train_loader, val_loader, test_loader, log)

            # Train Unconstrained model - still project onto stable models for stable initial point
            name = "uncon_sub{:d}_val{:d}".format(subject, val_set)
            model = dnb.dnbRNN(nu, width, ny, layers, nBatches=dataset_options["train_batch_size"], init_var=1.0)

            log, best_model = train.train_model(model, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                                                options=solver_options)

            test_and_save_model(name, best_model, train_loader, val_loader, test_loader, log)
