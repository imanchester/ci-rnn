import torch
import scipy.io
import urllib
import urllib.request
import zipfile
import shutil
import os
import numpy as np

from torch.utils.data import Dataset, DataLoader


class IO_data(Dataset):

    # Inputs and Outputs should be of size (seq_len) x (feature size)
    def __init__(self, inputs, outputs, seq_len=None):

        self.nu = inputs.shape[1]
        self.ny = outputs.shape[1]

        self.nBatches = inputs.shape[0] // seq_len
        self.batch_len = seq_len

        self.u = inputs[0:self.nBatches * self.batch_len].reshape((self.batch_len, self.nBatches, -1), order='F').astype(np.float32)
        self.y = outputs[0:self.nBatches * self.batch_len].reshape((self.batch_len, self.nBatches, -1), order='F').astype(np.float32)

        self.u = self.u.transpose(1, 2, 0)
        self.y = self.y.transpose(1, 2, 0)

    def __len__(self):
        return self.nBatches
        # return 1

    def __getitem__(self, index):
        return self.u[index, ...], self.y[index, ...]


def format_sine_sweep(sineSweep, batch_len=None):

    if batch_len is None:
        batch_len = sineSweep["Voltage"].shape[1]

    nBatches = sineSweep['Voltage'].shape[1] // batch_len

    data = {}
    data["inputs"] = torch.tensor(sineSweep["Force"][:, 0:batch_len * nBatches].reshape((nBatches, batch_len, 1), order='C')).float()
    data["outputs"] = torch.tensor(sineSweep["Acceleration"][:, 0:batch_len * nBatches].reshape((nBatches, batch_len, 3), order='C')).float()
    return data


# Loads the multi sine data set and formats into the correct form
def fetch_multi_sine(options, f16_dir):

    if options["autoregressive"]:
        nu = 1
        ny = 3

        train = scipy.io.loadmat(f16_dir + '/F16Data_SpecialOddMSine_Level2.mat')
        test = scipy.io.loadmat(f16_dir + '/F16Data_SpecialOddMSine_Level2_Validation.mat')

        u = train['Force'].reshape(-1, nu)
        y = train["Acceleration"].transpose((1, 2, 0)).reshape(-1, ny)

        u_test = test["Force"].reshape(-1, nu)
        y_test = test["Acceleration"].transpose((1, 2, 0)).reshape(-1, ny)

        # Shift Regressors measurement for the regressors one step into the future
        y_regressors = np.pad(y[:-1, :], ((1, 0), (0, 0)))
        u_ar = np.concatenate((u, y_regressors), 1)
        # u_ar = np.concatenate((u, y), 1)

        y_reg_test = np.pad(y_test[:-1, :], ((1, 0), (0, 0)))
        u_ar_test = np.concatenate((u_test, y_reg_test), 1)
        # u_ar_test = np.concatenate((u_test, y_test), 1)

        train = IO_data(u_ar, y, seq_len=options["train_seq_len"])
        test = IO_data(u_ar_test, y_test, seq_len=options["test_seq_len"])

        val = test

    else:
        nu = 1
        ny = 3

        train = scipy.io.loadmat(f16_dir + '/F16Data_SpecialOddMSine_Level2.mat')
        test = scipy.io.loadmat(f16_dir + '/F16Data_SpecialOddMSine_Level2_Validation.mat')

        u = train['Force'].reshape(-1, nu)
        y = train["Acceleration"].transpose((1, 2, 0)).reshape(-1, ny)

        u_test = test["Force"].reshape(-1, nu)
        y_test = test["Acceleration"].transpose((1, 2, 0)).reshape(-1, ny)

        train = IO_data(u, y, seq_len=options["train_seq_len"])
        test = IO_data(u_test, y_test, seq_len=options["test_seq_len"])

        val = test

    return train, val, test


# Loads the multi sine data set and formats into the correct form
def fetch_multi_sine_full(options, f16_dir):

    ms1 = scipy.io.loadmat(f16_dir + '/F16Data_FullMSine_Level1.mat')
    ms2 = scipy.io.loadmat(f16_dir + '/F16Data_FullMSine_Level2_Validation.mat')
    ms3 = scipy.io.loadmat(f16_dir + '/F16Data_FullMSine_Level3.mat')
    ms4 = scipy.io.loadmat(f16_dir + '/F16Data_FullMSine_Level4_Validation.mat')
    ms5 = scipy.io.loadmat(f16_dir + '/F16Data_FullMSine_Level5.mat')
    ms6 = scipy.io.loadmat(f16_dir + '/F16Data_FullMSine_Level6_Validation.mat')
    ms7 = scipy.io.loadmat(f16_dir + '/F16Data_FullMSine_Level7.mat')

    u = np.concatenate((ms2["Force"], ms4["Force"], ms6["Force"]), 1).T
    y = np.concatenate((ms2["Acceleration"], ms4["Acceleration"], ms6["Acceleration"]), 1).T

    u_test = np.concatenate((ms1["Force"], ms3["Force"], ms5["Force"], ms7["Force"]), 1).T
    y_test = np.concatenate((ms1["Acceleration"], ms3["Acceleration"], ms5["Acceleration"], ms7["Acceleration"]), 1).T

    if options["autoregressive"]:

        # Shift Regressors measurement for the regressors one step into the future
        y_regressors = np.pad(y[:-1, :], ((1, 0), (0, 0)))
        u = np.concatenate((u, y_regressors), 1)
        # u = np.concatenate((u_train, y_train), 1)

        y_reg_test = np.pad(y_test[:-1, :], ((1, 0), (0, 0)))
        u_test = np.concatenate((u_test, y_reg_test), 1)
        # u_test = np.concatenate((u_test, y_test), 1)

    train = IO_data(u, y, seq_len=options["train_seq_len"])
    val = IO_data(u_test, y_test, seq_len=options["test_seq_len"])

    test = val
    # test_data = np.concatenate((u, u_test), 0)
    # test_out = np.concatenate((y, y_test), 0)

    # test = IO_data(test_data, test_out, seq_len=test_data.shape(0))

    return train, val, test


# Loads the sine sweep data and puts it into the correct format
def fetch_sine_sweep(options, f16_dir):
    sineSweep1 = scipy.io.loadmat(f16_dir + '/F16Data_SineSw_Level1.mat')
    sineSweep2 = scipy.io.loadmat(f16_dir + '/F16Data_SineSw_Level2_Validation.mat')
    sineSweep3 = scipy.io.loadmat(f16_dir + '/F16Data_SineSw_Level3.mat')
    sineSweep4 = scipy.io.loadmat(f16_dir + '/F16Data_SineSw_Level4_Validation.mat')
    sineSweep5 = scipy.io.loadmat(f16_dir + '/F16Data_SineSw_Level5.mat')
    sineSweep6 = scipy.io.loadmat(f16_dir + '/F16Data_SineSw_Level6_Validation.mat')
    sineSweep7 = scipy.io.loadmat(f16_dir + '/F16Data_SineSw_Level7.mat')

    # Training Set
    u_train = np.concatenate((sineSweep1["Force"], sineSweep3["Force"], sineSweep5["Force"], sineSweep7["Force"]), 1).T
    y_train = np.concatenate((sineSweep1["Acceleration"], sineSweep3["Acceleration"], sineSweep5["Acceleration"], sineSweep7["Acceleration"]), 1).T

    # test sets
    u_test = np.concatenate((sineSweep2["Force"], sineSweep4["Force"], sineSweep6["Force"]), 1).T
    y_test = np.concatenate((sineSweep2["Acceleration"], sineSweep4["Acceleration"], sineSweep6["Acceleration"]), 1).T

    if options["autoregressive"]:

        # Shift Regressors measurement for the regressors one step into the future
        y_regressors = np.pad(y_train[:-1, :], ((1, 0), (0, 0)))
        u = np.concatenate((u_train, y_regressors), 1)
        # u = np.concatenate((u_train, y_train), 1)
        y = y_train

        y_reg_test = np.pad(y_test[:-1, :], ((1, 0), (0, 0)))
        u_test = np.concatenate((u_test, y_reg_test), 1)
        # u_test = np.concatenate((u_test, y_test), 1)

    train = IO_data(u, y, seq_len=options["train_seq_len"])
    test = IO_data(u_test, y_test, seq_len=options["train_seq_len"])

    val = test

    return train, val, test


def load_f16_data(options, dataset='multiSine', shuffle_training=True, workers=4):

    #  check to see if the data set has already been downloaded.
    working_dir = os.getcwd()
    data_dir = working_dir + '/data'

    # If the directory for the silverbox files does not exist
    if not os.path.exists(data_dir + '/F16GVT_Files'):
        print('Downloading f16 vibration test data to directory: ' + data_dir)
        #  url and path to download zip to
        url = 'http://www.nonlinearbenchmark.org/FILES/BENCHMARKS/F16/F16GVT_Files.zip'
        path_to_zip = working_dir + '/data/F16GVT_Files.zip'

        #  download and extract
        with urllib.request.urlopen(url) as response, open(path_to_zip, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
            with zipfile.ZipFile(path_to_zip) as zf:
                zf.extractall(data_dir)

            # delete zip file
            os.remove(path_to_zip)

    else:
        print('f16 vibration test data dir already exists at: ' + data_dir)

    f16_dir = data_dir + '/F16GVT_Files/BenchmarkData'

    # load specific dataset
    if dataset == "multiSine":
        train, val, test = fetch_multi_sine(options, f16_dir)

    if dataset == "multiSine_full":
        train, val, test = fetch_multi_sine_full(options, f16_dir)

    if dataset == 'sineSweep':
        train, val, test = fetch_sine_sweep(options, f16_dir)

    # Construct data loader for training set
    train_loader = DataLoader(train, batch_size=options["train_batch_size"], shuffle=shuffle_training, num_workers=workers)

    # construct loader for validation set
    val_loader = DataLoader(val, batch_size=options["val_batch_size"], shuffle=False, num_workers=workers)

    # construct loader for test set
    test_loader = DataLoader(test, batch_size=options["test_batch_size"], shuffle=False, num_workers=workers)

    return train_loader, val_loader, test_loader
