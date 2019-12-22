import data.f16 as f16Dataset
import data.gait_prediction as gait_pred
import data.chen_sim as chen


def make_default_options(train_sl=2000, test_sl=5000, val_sl=2000, train_bs=1, test_bs=1, val_bs=1, ar=False, val_set=1):
    options = {"train_seq_len": train_sl, "test_seq_len": test_sl, "val_seq_len": val_sl,
               "train_batch_size": train_bs, "test_batch_size": test_bs, "val_batch_size": val_bs, "autoregressive": ar, "val_set":val_set}
    return options


def load_dataset(dataset, dataset_options=None):

    if dataset_options is None:
        dataset_options = make_default_options()

    # The f16 dataset contains multiple different datasets
    if dataset == "f16_sineSweep":
        train, val, test = f16Dataset.load_f16_data(dataset="sineSweep", options=dataset_options)

    elif dataset == "f16_multiSine":
        train, val, test = f16Dataset.load_f16_data(dataset="multiSine", options=dataset_options)

    elif dataset == "f16_multiSine_full":
        train, val, test = f16Dataset.load_f16_data(dataset="multiSine_full", options=dataset_options)

    elif dataset == "gait_prediction_stairs":
        train, val, test = gait_pred.load_data(dataset="stairs", options=dataset_options, subject=dataset_options["subject"])

    elif dataset == "gait_prediction":
        train, val, test = gait_pred.load_data(dataset="walk", options=dataset_options, subject=dataset_options["subject"])

    elif dataset == "chen":
        train, val, test = chen.load_data(options=dataset_options)

    else:
        raise Exception("Dataset: {} is not an option".format(dataset))

    return train, val, test
