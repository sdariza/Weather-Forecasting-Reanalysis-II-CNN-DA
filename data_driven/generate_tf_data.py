import numpy as np
import tensorflow as tf
from netCDF4 import Dataset

tf.random.set_seed(123)
BATCH_SIZE = 32


def get_data(current_state, variable2train):
    """
    :param current_state:
    :param variable2train:
    """
    Xi_nc = Dataset(f'./data/data-training/{current_state}/{variable2train}{current_state}.nc')

    if current_state == 18:
        Xi1_nc = Dataset(f'./data/data-training/0/{variable2train}0.nc')
    else:
        Xi1_nc = Dataset(f'./data/data-training/{current_state + 6}/{variable2train}{current_state + 6}.nc')

    X = Xi_nc.variables[f'{variable2train}'][:].data[..., np.newaxis]
    Y = Xi1_nc.variables[f'{variable2train}'][:].data[..., np.newaxis]

    Xi_nc.close()
    Xi1_nc.close()

    if variable2train == 'air':
        X = X - 273.15
        Y = Y - 273.15

    if current_state == 18:
        X = X[:-1, ...]
        Y = Y[1:, ...]

    split_ratio_train = 0.7
    split_ratio_val = 0.2

    num_samples = len(X)
    num_train = int(split_ratio_train * num_samples)
    num_val = int(split_ratio_val * num_samples)

    train_data = X[:num_train]
    val_data = X[num_train:num_train + num_val]
    test_data = X[num_train + num_val:]

    train_labels = Y[:num_train]
    val_labels = Y[num_train:num_train + num_val]
    test_labels = Y[num_train + num_val:]

    train_ds = tf.data.Dataset.from_tensor_slices(tensors=(train_data, train_labels))
    train_ds = train_ds.shuffle(buffer_size=num_train, seed=123).batch(batch_size=BATCH_SIZE)

    valid_ds = tf.data.Dataset.from_tensor_slices(tensors=(val_data, val_labels))
    valid_ds = valid_ds.shuffle(buffer_size=num_val, seed=123).batch(batch_size=BATCH_SIZE)

    test_ds = tf.data.Dataset.from_tensor_slices(tensors=(test_data, test_labels))
    test_ds = test_ds.shuffle(buffer_size=num_samples - num_train - num_val, seed=123).batch(batch_size=BATCH_SIZE)
    del X, Y

    train_ds.save(f'./data/tf_data_training/{current_state}/train_{variable2train}_{current_state}')
    valid_ds.save(f'./data/tf_data_training/{current_state}/valid_{variable2train}_{current_state}')
    test_ds.save(f'./data/tf_data_training/{current_state}/test_{variable2train}_{current_state}')


if __name__ == "__main__":
    print('Generating train dataset...')
    for state in [0, 6, 12, 18]:
        for variable in ['air', 'vwnd', 'uwnd']:
            get_data(state, variable)
