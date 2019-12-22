from torch.utils.data import Dataset
import numpy as np


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
