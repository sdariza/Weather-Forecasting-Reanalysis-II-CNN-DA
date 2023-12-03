import argparse

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import plotly.graph_objects as go

tf.random.set_seed(123)
import numpy as np
import optuna

parser = argparse.ArgumentParser(prefix_chars='--')
parser.add_argument('--state', type=int, help='State to predict')
parser.add_argument('--variable', type=str, help='Variable to predict')

args = parser.parse_args()
state = args.state
variable = args.variable
EPOCHS = 100


def create_optimizer(trial):
    kwargs = {}
    kwargs["learning_rate"] = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    optimizer = getattr(tf.optimizers, 'Adam')(**kwargs)
    return optimizer


def get_data():
    train_ds = tf.data.Dataset.load(f'./data/tf_data_training/{state}/train_{variable}_{state}')
    valid_ds = tf.data.Dataset.load(f'./data/tf_data_training/{state}/valid_{variable}_{state}')
    test_ds = tf.data.Dataset.load(f'./data/tf_data_training/{state}/test_{variable}_{state}')
    return train_ds, valid_ds, test_ds


def create_model(trial):
    kernel_sizes = [(2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2), (3, 4), (4, 3), (4, 4)]
    kz_selected = trial.suggest_categorical("kernel_size", kernel_sizes)
    alpha_selected = trial.suggest_float("alpha", 0.01, 0.9, log=True)
    model = tf.keras.models.Sequential(
        layers=[
            # input
            tf.keras.layers.Input(shape=(73, 144, 1), name='input'),
            # encoder
            tf.keras.layers.Conv2D(filters=16, kernel_size=kz_selected, padding='same', name='conv2D_1'),
            tf.keras.layers.LeakyReLU(alpha=alpha_selected, name='act_1'),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same', name='AvP_1'),
            tf.keras.layers.Conv2D(filters=32, kernel_size=kz_selected, padding='same', name='conv2D_2'),
            tf.keras.layers.LeakyReLU(alpha=alpha_selected, name='act_2'),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same', name='AvP_2'),
            tf.keras.layers.Conv2D(filters=64, kernel_size=kz_selected, padding='same', name='conv2D_3'),
            tf.keras.layers.LeakyReLU(alpha=alpha_selected, name='act_3'),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same', name='AvP_3'),
            tf.keras.layers.Conv2D(filters=128, kernel_size=kz_selected, padding='same', name='conv2D_4'),
            tf.keras.layers.LeakyReLU(alpha=alpha_selected, name='act_4'),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same', name='AvP_4'),
            # decoder
            tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=kz_selected, padding='same', name='conv2DT_1'),
            tf.keras.layers.LeakyReLU(alpha=alpha_selected, name='act_5'),
            tf.keras.layers.UpSampling2D(size=(2, 2), name='UpS2D_1'),
            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=kz_selected, padding='same', name='conv2DT_2'),
            tf.keras.layers.LeakyReLU(alpha=alpha_selected, name='act_6'),
            tf.keras.layers.UpSampling2D(size=(2, 2), name='UpS2D_2'),
            tf.keras.layers.Cropping2D(cropping=((1, 0), (0, 0)), name='Cropping_1'),
            tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=kz_selected, padding='same', name='conv2DT_3'),
            tf.keras.layers.LeakyReLU(alpha=alpha_selected, name='act_7'),
            tf.keras.layers.UpSampling2D(size=(2, 2), name='UpS2D_3'),
            tf.keras.layers.Cropping2D(cropping=((1, 0), (0, 0)), name='Cropping_3'),
            tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=kz_selected, padding='same', name='conv2DT_4'),
            tf.keras.layers.LeakyReLU(alpha=alpha_selected, name='act_8'),
            tf.keras.layers.UpSampling2D(size=(2, 2), name='UpS2D_4'),
            tf.keras.layers.Cropping2D(cropping=((1, 0), (0, 0)), name='Cropping_4'),

            # output
            tf.keras.layers.Conv2D(filters=1, kernel_size=kz_selected, padding='same', activation='linear',
                                   name='output')
        ]
        , name=f'CNN-Weather-Forecasting-Optimizing-Parameters_{variable}_{state}')
    return model


def create_layout(train_data, val_data, fig_type, name_train, name_val, xaxis_title, yaxis_title, _trial_id, lr, a, kz):
    fig = go.Figure()
    epochs = list(range(1, len(train_data) + 1))
    fig.add_trace(go.Scatter(x=epochs, y=np.log(train_data), mode='lines+markers', name=name_train))
    fig.add_trace(go.Scatter(x=epochs, y=np.log(val_data), mode='lines+markers', name=name_val))

    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
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
        f'data_driven/optimize/{fig_type}_imgs/{variable}/{state}/{fig_type}_{variable}_{state}_id_{_trial_id}_lr_{lr}_a_{a}_kz_{kz}.pdf',
        scale=2)


def plot_loss_metric(history, params, _trial_id):
    kz, lr, a = params['kernel_size'], params['learning_rate'], params['alpha']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_mae = history.history['mean_absolute_error']
    val_mae = history.history['val_mean_absolute_error']
    create_layout(train_data=train_loss, val_data=val_loss, fig_type='loss', name_train='Loss in training',
                  name_val='Loss in validation', xaxis_title='Epoch', yaxis_title='ln(MSE)', _trial_id=_trial_id, lr=lr,
                  a=a, kz=kz)
    create_layout(train_data=train_mae, val_data=val_mae, fig_type='mae', name_train='MAE in training',
                  name_val='MAE in validation', xaxis_title='Epoch', yaxis_title='ln(MAE)', _trial_id=_trial_id, lr=lr,
                  a=a, kz=kz)


def objective(trial):
    train_ds, valid_ds, test_ds = get_data()
    model = create_model(trial)
    optimizer = create_optimizer(trial)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='auto', verbose=1)
    model.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError(),
                  metrics=[tf.keras.losses.MeanAbsoluteError(), tf.keras.losses.MeanSquaredError()])
    history = model.fit(train_ds, validation_data=valid_ds, epochs=EPOCHS, callbacks=[early_stop], verbose=0)
    plot_loss_metric(history, trial.__dict__['_cached_frozen_trial'].params, trial.__dict__['_trial_id'])
    _, mae, mse = model.evaluate(test_ds)
    if np.isnan(mae) or np.isnan(mse):
        return np.inf, np.inf
    return mae, mse


if __name__ == "__main__":
    study = optuna.create_study(study_name=f'optimizing_parameters_{variable}_{state}_pareto',
                                storage=f'sqlite:///data_driven/optimize/db/optimizing_parameters_state_{state}_pareto.db',
                                load_if_exists=True, directions=['minimize'] * 2)
    study.optimize(objective, n_trials=1)
