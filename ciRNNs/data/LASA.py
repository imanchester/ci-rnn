import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import CubicSpline as spline


class LASA_data():
    def __init__(self, pos, vel, acc, t, dt):
        self.pos = pos
        self.vel = vel
        self.acc = acc
        self.t = t
        self.dt = dt

        self.examples = pos.__len__()


class IO_data(Dataset):

    # Inputs and Outputs should be of size (seq_len) x (feature size)
    def __init__(self, inputs, outputs, seq_len=None):

        self.nu = inputs[0].shape[0]
        self.ny = outputs[0].shape[0]

        self.nBatches = inputs.__len__()

        self.u = inputs.astype(np.float32)
        self.y = outputs.astype(np.float32)

    def __len__(self):
        return self.nBatches

    def __getitem__(self, index):
        # return self.u[index][None, ...], self.y[index][None, ...]
        return self.u[index, ...], self.y[index, ...]


def cubic_interp(time, data, notch_points):
    return 0


def read_LASA_data(shape='Angle', workers=4, shuffle_training=False):
    path = "./data/LASADataset/DataSet/"
    data = io.loadmat(path + shape + '.mat')
    dt = data["dt"]

    vel = []
    pos = []
    acc = []
    time = []

    for examples in range(data["demos"][0].__len__()):
        sample = data["demos"][0][examples]

        pos += [sample["pos"][0, 0]]
        vel += [sample["vel"][0, 0]]
        acc += [sample["acc"][0, 0]]
        time += [sample["t"][0, 0]]

    pos = np.stack(pos, 0)
    vel = np.stack(vel, 0)
    acc = np.stack(acc, 0)
    time = np.stack(time, 0)

    u_train = pos[0:1]
    y_train = np.concatenate((pos[0:1], vel[0:1], acc[0:1]), 1)

    u_test = pos[1:-1]
    y_test = np.concatenate((pos[1:-1], vel[1:-1], acc[1:-1]), 1)

    train = IO_data(u_train, y_train)
    test = IO_data(u_test, y_test)

    train_loader = DataLoader(train, batch_size=1, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test, batch_size=1, shuffle=False, num_workers=workers)

    return train_loader, test_loader


if __name__ == '__main__':

    train_loader, test_loader = read_LASA_data(shape="Worm")

    for u, y in train_loader:
        plt.plot(u[0, 0], u[0, 1])

    for u, y in test_loader:
        plt.plot(u[0, 0], u[0, 1], 'r')

    plt.show()

    print("fin")
