import torch
import scipy.io
import urllib
import urllib.request
import zipfile
import shutil
import os
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


class IO_data(Dataset):

    # Inputs and Outputs should be of size (seq_len) x (feature size)
    def __init__(self, inputs, outputs, seq_len=None):

        self.nu = inputs[0].shape[0]
        self.ny = outputs[0].shape[0]

        self.nBatches = inputs.__len__()

        as_float = lambda x: x.astype(np.float32)
        self.u = list(map(as_float, inputs))
        self.y = list(map(as_float, outputs))

    def __len__(self):
        return self.nBatches

    def __getitem__(self, index):
        # return self.u[index][None, ...], self.y[index][None, ...]
        return self.u[index], self.y[index]


def dynamics(y1, y2, u1, u2):
    return (0.8 - 0.5 * np.exp(-y1 ** 2)) * y1 - (0.3 + 0.9 * np.exp(-y1 ** 2)) * y2 + u1 + 0.2 * u2 + 0.1 * u1 * u2


def gen_data(N=1020, b=20, w_sd=0.5, v_sd=0.3, batches=20, gain=1.8, u_sd=1.0):
    period = 5
    y = np.zeros((batches, N, 1))

    u = u_sd * np.random.randn(batches, N // period, 1)  # test input
    u = np.tile(u, (1, 1, period)).reshape((batches, -1, 1))

    for kk in range(2, N):
        uk = u[:, (kk - 1)]
        u_last = u[:, (kk - 2)]
        yk = y[:, kk - 1]
        y_last = y[:, kk - 2]
        y[:, kk] = gain * dynamics(yk, y_last, uk, u_last) + w_sd * np.random.randn(batches, 1)

    yo = y + v_sd * np.random.randn(batches, N, 1)
    return u.swapaxes(1, 2), yo.swapaxes(1, 2)


def gen_test(N=5000, b=20, w_sd=0.5, v_sd=2.0, batches=1, gain=1.8, u_sd=1.0):

    # Test sets different input size
    u1 = []
    y1 = []

    for u_sd in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
        ut, yt = gen_data(N=N, batches=1, u_sd=u_sd, gain=gain)
        u1.append(ut[0])
        y1.append(yt[0])

    period = 5
    y = np.zeros((batches, N, 1))

    # test set with ranmp input
    u = u_sd * np.random.randn(batches, N // period, 1)
    u = np.tile(u, (1, 1, period)).reshape((batches, -1, 1))

    u_gain = np.linspace(0.1, 2.0, N)
    u = u * u_gain[None, :, None]

    for kk in range(2, N):
        uk = u[:, (kk - 1)]
        u_last = u[:, (kk - 2)]
        yk = y[:, kk - 1]
        y_last = y[:, kk - 2]
        y[:, kk] = gain * dynamics(yk, y_last, uk, u_last) + w_sd * np.random.randn(batches, 1)

    yo = y + v_sd * np.random.randn(batches, N, 1)

    u1.append(u[0].T)
    y1.append(yo[0].T)
    return u1, y1


def load_data(options, shuffle_training=True, workers=1, subject=1, batches=20):
    #  check to see if the data set has already been downloaded.

    gain = options["gain"]
    # Test Set
    test_u, test_y = gen_test(N=options["test_seq_len"])

    # Generate Validation and training sets
    inputs, outputs = gen_data(N=options["train_seq_len"], batches=options["train_batch_size"])

    # split data into training, validation
    L = inputs.__len__()
    val_set = options["val_set"]

    # val sets
    val_u = [x for i, x in enumerate(inputs) if i == val_set]
    val_y = [x for i, x in enumerate(outputs) if i == val_set]

    # training sets
    train_u = [x for i, x in enumerate(inputs) if i != val_set and i < L]
    train_y = [x for i, x in enumerate(outputs) if i != val_set and i < L]

    training = IO_data(train_u, train_y)
    validation = IO_data(val_u, val_y)
    test = IO_data(test_u, test_y)

    # training = IO_data(inputs[0:-2], outputs[0:-2])
    # validation = IO_data(inputs[-2:], outputs[-2:-1])
    # test = IO_data(inputs[-2:], outputs[-2:])

    # Merge the list of arrays into a 3D tensor - just kdding, the trials have different lengths
    # merge_arrays = lambda x, y: np.concatenate((x[None, ...], y[None, ...]), 0)
    # inputs = reduce(merge_arrays, inputs)
    # outputs = reduce(merge_arrays, outputs)

    train_loader = DataLoader(training, batch_size=1, shuffle=shuffle_training, num_workers=workers)
    val_loader = DataLoader(validation, batch_size=1, num_workers=workers)
    test_loader = DataLoader(test, batch_size=1, num_workers=workers)

    train_loader.nu = training.nu
    train_loader.ny = training.ny

    val_loader.nu = training.nu
    val_loader.ny = training.ny

    test_loader.nu = training.nu
    test_loader.ny = training.ny

    return train_loader, val_loader, test_loader
