import torch
import scipy.io
import urllib
import urllib.request
import zipfile
import shutil
import os
import matplotlib.pyplot as plt


def load_silverbox(batch_len, trainSamples=None):

    #  check to see if the data set has already been downloaded.
    working_dir = os.getcwd()
    data_dir = working_dir + '/data'

    # If the directory for the silverbox files does not exist
    if not os.path.exists(data_dir + '/SilverboxFiles'):
        print('Downloading Silverbox data to directory: ' + data_dir)
        #  url and path to download zip to
        url = 'http://www.nonlinearbenchmark.org/FILES/BENCHMARKS/SILVERBOX/SilverboxFiles.zip'
        path_to_zip = working_dir + '/data/SilverboxFiles.zip'

        #  download and extract
        with urllib.request.urlopen(url) as response, open(path_to_zip, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
            with zipfile.ZipFile(path_to_zip) as zf:
                zf.extractall(data_dir)

            # delete zip file
            os.remove(path_to_zip)

    else:
        print('Silver box data dir already exists at: ' + data_dir)

    silverbox_dir = data_dir + '/SilverboxFiles'
    data = scipy.io.loadmat(silverbox_dir + '/SNLS80mV.mat')

    inputs = data['V1'].T
    outputs = data['V2'].T

    train_in, val_in, test_in = separate_data(inputs, trainSamples)
    train_out, val_out, test_out = separate_data(outputs, trainSamples)

    #  calculate the number of batches that can be made from
    nBatches = train_in.shape[0] // batch_len
    if train_in.shape[0] % batch_len != 0:
        print('Data does not divide nicely. Discarding ', train_in.shape[0] - batch_len * nBatches, 'Samples')

    # truncate training and validation data so that we can form batches of constant size
    train_in = train_in[0: nBatches * batch_len].reshape(nBatches, batch_len, -1)
    train_out = train_out[0: nBatches * batch_len].reshape(nBatches, batch_len, -1)
    val_in = val_in[0: nBatches * batch_len].reshape(nBatches, batch_len, -1)
    val_out = val_out[0: nBatches * batch_len].reshape(nBatches, batch_len, -1)

    train_in = torch.tensor(train_in).float()
    train_out = torch.tensor(train_out).float()

    val_in = torch.tensor(val_in).float()
    val_out = torch.tensor(val_out).float()

    test_in = torch.tensor(test_in).unsqueeze(0).float()
    test_out = torch.tensor(test_out).unsqueeze(0).float()

    train = {"inputs": train_in, "outputs": train_out}
    val = {"inputs": val_in, "outputs": val_out}
    test = {"inputs": test_in, "outputs": test_out}
    return train, val, test


#  separates dat into test, validation and training
def separate_data(dat, trainSamples):
    n_test = 40400
    test = dat[0:n_test]
    val = dat[n_test + 1:]

    if trainSamples is None:
        train = dat[n_test + 1:]
    else:
        train = dat[n_test + 1:n_test + trainSamples + 1]

    return train, val, test


def plot_data(inputs, outputs, pltrange=(0, 1000)):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(inputs[pltrange[0]: pltrange[1]])
    ax1.grid()
    plt.ylabel('Inputs')
    ax2.plot(outputs[pltrange[0]: pltrange[1]], '.')
    ax2.grid()
    plt.ylabel('Outputs')
    plt.show()


if __name__ == "__main__":
    print()
    print('Test script in silverbox.py.')
    dat = load_silverbox(100)
