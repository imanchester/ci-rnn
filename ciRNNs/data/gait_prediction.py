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


def load_data(options, dataset="walk", shuffle_training=True, workers=1, subject=1):
    #  check to see if the data set has already been downloaded.

    folder = "./data/gait_prediction/python_data/".format(subject)
    if dataset == "stairs":
        file_name = "sub{:d}_Upstairs_canes_all.mat".format(subject)
    elif dataset == "walk":
        file_name = "sub{:d}_Walk_canes_all.mat".format(subject)

    data = scipy.io.loadmat(folder + file_name)

    # What even is this data format ....
    transpose = lambda x: x.T
    inputs = list(map(transpose, data["p_data"][0, 0][0][0]))
    outputs = list(map(transpose, data["p_data"][0, 0][1][0]))

    # split data into training, validation and test
    L = inputs.__len__()
    val_set = options["val_set"]

    # test sets
    test_u = inputs[-2:]
    test_y = outputs[-2:]

    # val sets
    val_u = [x for i, x in enumerate(inputs) if i == val_set]
    val_y = [x for i, x in enumerate(outputs) if i == val_set]

    # training sets
    train_u = [x for i, x in enumerate(inputs) if i != val_set and i < L - 2]
    train_y = [x for i, x in enumerate(outputs) if i != val_set and i < L - 2]

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
