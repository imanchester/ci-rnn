
from functools import wraps  # This convenience func preserves name and docstring
from matplotlib import pyplot as plt


def add_method(cls):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(*args, **kwargs)
        setattr(cls, func.__name__, wrapper)
        # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
        return func  # returning func means func can still be used normally
    return decorator


class data_logger():
    def __init__(self):
        self.data_dict = {}

    def add_entry(self, entry):
        for key, value in entry.items():
            if key not in self.data_dict.keys():
                self.data_dict[key] = []

            self.data_dict[key] += [float(value)]

    # Appends the entries in log to this log
    def append_log(self, log):
        for key, value in log.data_dict.items():
            if key not in self.data_dict.keys():
                self.data_dict[key] = []

            self.data_dict[key] += value

    def plot_data(self, name):
        plt.plot(self.data_dict[name])
        plt.show()
        plt.pause(1E-6)
