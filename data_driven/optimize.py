import argparse

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

tf.random.set_seed(123)
import plotly.graph_objects as go
import netCDF4 as nc
import numpy as np
import optuna

parser = argparse.ArgumentParser(prefix_chars='--')
parser.add_argument('--state', type=int, help='State to predict')
parser.add_argument('--variable', type=str, help='Variable to predict')

args = parser.parse_args()
state = args.state
variable = args.variable
EPOCHS = 100
BATCH_SIZE = 32


def create_optimizer(trial):
    kwargs = {}
    kwargs["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    optimizer = getattr(tf.optimizers, 'Adam')(**kwargs)
    return optimizer


def get_data():
    Xi_nc = nc.Dataset(f'./data/data-training/{state}/{variable}{state}.nc')

    if state == 18:
        Xi1_nc = nc.Dataset(f'./data/data-training/0/{variable}0.nc')
    else:
        Xi1_nc = nc.Dataset(f'./data/data-training/{state + 6}/{variable}{state + 6}.nc')

    X = Xi_nc.variables[f'{variable}'][:].data
    Y = Xi1_nc.variables[f'{variable}'][:].data

    Xi_nc.close()
    Xi1_nc.close()

    if variable == 'air':
        X = X - 273.15
        Y = Y - 273.15

    if state == 18:
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
    return train_ds, valid_ds, test_ds


def create_model(trial):
    kernel_sizes = [(2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2), (3, 4), (4, 3), (4, 4)]
    kz_selected = trial.suggest_categorical("kernel_size", kernel_sizes)
    alpha_optiones = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    alpha_selected = trial.suggest_categorical("alpha", alpha_optiones)
    inputs = tf.keras.layers.Input(shape=(73, 144, 1), name='input')

    # Encoder
    conv1 = tf.keras.layers.Conv2D(32, kz_selected, padding='same', name='conv2D_1')(inputs)
    act1 = tf.keras.layers.LeakyReLU(alpha=alpha_selected, name='act_1')(conv1)
    conv2 = tf.keras.layers.Conv2D(32, kz_selected, padding='same', name='conv2D_2')(act1)
    act2 = tf.keras.layers.LeakyReLU(alpha=alpha_selected, name='act_2')(conv2)
    pool1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same', name='AvP_1')(act2)

    conv3 = tf.keras.layers.Conv2D(64, kz_selected, padding='same', name='conv2D_3')(pool1)
    act3 = tf.keras.layers.LeakyReLU(alpha=alpha_selected, name='act_3')(conv3)
    conv4 = tf.keras.layers.Conv2D(64, kz_selected, padding='same', name='conv2D_4')(act3)
    act4 = tf.keras.layers.LeakyReLU(alpha=alpha_selected, name='act_4')(conv4)
    pool2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same', name='AvP_2')(act4)

    # Decoder
    deconv_1 = tf.keras.layers.Conv2DTranspose(128, kz_selected, padding='same', name='conv2DT_1')(pool2)
    act5 = tf.keras.layers.LeakyReLU(alpha=alpha_selected, name='act_5')(deconv_1)
    deconv2 = tf.keras.layers.Conv2DTranspose(64, kz_selected, padding='same', name='conv2DT_2')(act5)
    act6 = tf.keras.layers.LeakyReLU(alpha=alpha_selected, name='act_6')(deconv2)
    upsample1 = tf.keras.layers.UpSampling2D(size=(2, 2), name='UpS2D_1')(act6)
    crop1 = tf.keras.layers.Cropping2D(cropping=((1, 0), (0, 0)), name='Cropping_1')(upsample1)
    concat1 = tf.keras.layers.concatenate([act4, crop1], axis=-1, name='concat_1')  # Conexión de salto

    deconv3 = tf.keras.layers.Conv2DTranspose(64, kz_selected, padding='same', name='conv2DT_3')(concat1)
    act7 = tf.keras.layers.LeakyReLU(alpha=alpha_selected, name='act_7')(deconv3)
    deconv4 = tf.keras.layers.Conv2DTranspose(32, kz_selected, padding='same', name='conv2DT_4')(act7)
    act8 = tf.keras.layers.LeakyReLU(alpha=alpha_selected, name='act_8')(deconv4)
    upsample2 = tf.keras.layers.UpSampling2D(size=(2, 2), name='UpS2D_3')(act8)
    crop2 = tf.keras.layers.Cropping2D(cropping=((1, 0), (0, 0)), name='Cropping_3')(upsample2)
    concat2 = tf.keras.layers.concatenate([act2, crop2], axis=-1, name='concat_2')  # Conexión de salto

    deconv5 = tf.keras.layers.Conv2DTranspose(32, kz_selected, padding='same', name='conv2DT_5')(concat2)
    act9 = tf.keras.layers.LeakyReLU(alpha=alpha_selected, name='act_9')(deconv5)
    deconv6 = tf.keras.layers.Conv2DTranspose(32, kz_selected, padding='same', name='conv2DT_6')(act9)
    act10 = tf.keras.layers.LeakyReLU(alpha=alpha_selected, name='act_10')(deconv6)
    output = tf.keras.layers.Conv2D(1, kz_selected, padding='same', activation='linear', name='output')(act10)

    model = tf.keras.models.Model(inputs=inputs, outputs=output, name='CNN-Weather-Forecasting-Optimizing-Parameters')
    return model


def plot_loss_metric(history, params, _trial_id):
    kz, lr, a = params['kernel_size'], params['learning_rate'], params['alpha']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_mae = history.history['mean_absolute_error']
    val_mae = history.history['val_mean_absolute_error']

    fig = go.Figure()
    epochs = list(range(1, len(train_loss) + 1))
    fig.add_trace(go.Scatter(x=epochs, y=np.log(train_loss), mode='lines+markers', name='Loss in training'))
    fig.add_trace(go.Scatter(x=epochs, y=np.log(val_loss), mode='lines+markers', name='Loss in validation'))

    fig.update_layout(
        xaxis_title='Epoch',
        yaxis_title='ln(loss)',
        width=700,
        height=600,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    fig.update_layout(
        legend=dict(x=0.2, y=0.99, font=dict(size=22)),
        yaxis=dict(tickfont=dict(size=20), title=dict(font=dict(size=28))),
        xaxis=dict(tickfont=dict(size=20), title=dict(font=dict(size=28))),
    )

    fig.write_image(
        f'data_driven/optimize/loss_imgs/{variable}/{state}/loss_{variable}_{state}_id_{_trial_id - 1}_lr_{lr}_a_{a}_kz_{kz}.pdf',
        scale=2)

    fig_1 = go.Figure()
    fig_1.add_trace(go.Scatter(x=epochs, y=np.log(train_mae), mode='lines+markers', name='MAE in training'))
    fig_1.add_trace(go.Scatter(x=epochs, y=np.log(val_mae), mode='lines+markers', name='MAE in validation'))

    fig_1.update_layout(
        xaxis_title='Epoch',
        yaxis_title='ln(MAE)',
        width=700,
        height=600,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    fig_1.update_layout(
        legend=dict(x=0.2, y=0.99, font=dict(size=22)),
        yaxis=dict(tickfont=dict(size=20), title=dict(font=dict(size=28))),
        xaxis=dict(tickfont=dict(size=20), title=dict(font=dict(size=28))),
    )

    fig_1.write_image(
        f'data_driven/optimize/mae_imgs/{variable}/{state}/mae_{variable}_{state}_id_{_trial_id - 1}_lr_{lr}_a_{a}_kz_{kz}.pdf',
        scale=2)


def objective(trial):
    train_ds, valid_ds, test_ds = get_data()
    model = create_model(trial)
    optimizer = create_optimizer(trial)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='auto', verbose=0)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.losses.MeanAbsoluteError(), tf.keras.losses.MeanSquaredError()])
    history = model.fit(train_ds, validation_data=valid_ds, epochs=EPOCHS, callbacks=[early_stop], verbose=0)
    plot_loss_metric(history, trial.__dict__['_cached_frozen_trial'].params, trial.__dict__['_trial_id'])
    _, mae, mse = model.evaluate(test_ds)
    if np.isnan(mae) or np.isnan(mse):
        return np.inf, np.inf
    return mae, mse


if __name__ == "__main__":
    study = optuna.create_study(study_name=f'optimizing_parameters_{variable}_{state}',
                                storage=f'sqlite:///data_driven/optimize/db/optimizing_parameters_state_{state}.db',
                                load_if_exists=True, directions=['minimize'] * 2)
    study.optimize(objective, n_trials=1)
