import argparse

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
tf.random.set_seed(123)
import netCDF4 as nc
import matplotlib.pyplot as plt
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

    Xi_nc = nc.Dataset(f'data-training/{state}/{variable}{state}.nc')
    
    if state == 18:
        Xi1_nc = nc.Dataset(f'data-training/0/{variable}0.nc')
    else:
        Xi1_nc = nc.Dataset(f'data-training/{state+6}/{variable}{state+6}.nc')

    X = Xi_nc.variables[f'{variable}'][:].data
    Y =  Xi1_nc.variables[f'{variable}'][:].data

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
    val_data = X[num_train:num_train+num_val]
    test_data = X[num_train+num_val:]

    train_labels = Y[:num_train]
    val_labels = Y[num_train:num_train+num_val]
    test_labels = Y[num_train+num_val:]

    train_ds = tf.data.Dataset.from_tensor_slices(tensors=(train_data, train_labels))
    train_ds = train_ds.shuffle(buffer_size = num_train, seed=123).batch(batch_size=BATCH_SIZE)

    valid_ds = tf.data.Dataset.from_tensor_slices(tensors=(val_data, val_labels))
    valid_ds = valid_ds.shuffle(buffer_size = num_val, seed=123).batch(batch_size=BATCH_SIZE)

    test_ds = tf.data.Dataset.from_tensor_slices(tensors=(test_data, test_labels))
    test_ds = test_ds.shuffle(buffer_size = num_samples-num_train-num_val, seed=123).batch(batch_size=BATCH_SIZE)
    del X, Y
    return train_ds, valid_ds, test_ds


def create_model(trial):
  kernel_sizes = [(2,2),(2,3),(3,2),(3,3),(2,4),(4,2),(3,4),(4,3),(4,4)]
  kz_selected = trial.suggest_categorical("kernel_size", kernel_sizes)
  alpha_optiones = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

  alpha_selected = trial.suggest_categorical("alpha", alpha_optiones)
  model = tf.keras.models.Sequential(name=f'CNN-Weather-Forecasting-Optimizing-Parameters')

  model.add(tf.keras.layers.Input(shape=(73,144,1), name='input')) #input
  
#encoder
  model.add(tf.keras.layers.Conv2D(filters=16, kernel_size= kz_selected, padding='same', name='conv2D_1'))
  model.add(tf.keras.layers.LeakyReLU(alpha=alpha_selected, name='act_1'))
  model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2),padding='same', name='AvP_1'))

  model.add(tf.keras.layers.Conv2D(filters=32, kernel_size= kz_selected, padding='same', name='conv2D_2'))
  model.add(tf.keras.layers.LeakyReLU(alpha=alpha_selected, name='act_2'))
  model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2),padding='same', name='AvP_2'))

  model.add(tf.keras.layers.Conv2D(filters=64, kernel_size= kz_selected, padding='same', name='conv2D_3'))
  model.add(tf.keras.layers.LeakyReLU(alpha=alpha_selected, name='act_3'))
  model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2),padding='same', name='AvP_3'))

  model.add(tf.keras.layers.Conv2D(filters=128, kernel_size= kz_selected, padding='same', name='conv2D_4'))
  model.add(tf.keras.layers.LeakyReLU(alpha=alpha_selected, name='act_4'))
  model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2),padding='same', name='AvP_4'))

#decoder 
  model.add(tf.keras.layers.Conv2DTranspose(filters=128, kernel_size= kz_selected, padding='same', name='conv2DT_1'))
  model.add(tf.keras.layers.LeakyReLU(alpha=alpha_selected, name='act_5'))
  model.add(tf.keras.layers.UpSampling2D(size=(2,2), name='UpS2D_1'))


  model.add(tf.keras.layers.Conv2DTranspose(filters=64, kernel_size= kz_selected, padding='same', name='conv2DT_2'))
  model.add(tf.keras.layers.LeakyReLU(alpha=alpha_selected, name='act_6'))
  model.add(tf.keras.layers.UpSampling2D(size=(2,2), name='UpS2D_2'))
  model.add(tf.keras.layers.Cropping2D(cropping=((1,0),(0,0)), name='Cropping_1'))

  model.add(tf.keras.layers.Conv2DTranspose(filters=32, kernel_size= kz_selected, padding='same', name='conv2DT_3'))
  model.add(tf.keras.layers.LeakyReLU(alpha=alpha_selected, name='act_7'))
  model.add(tf.keras.layers.UpSampling2D(size=(2,2), name='UpS2D_3'))
  model.add(tf.keras.layers.Cropping2D(cropping=((1,0),(0,0)), name='Cropping_3'))

  model.add(tf.keras.layers.Conv2DTranspose(filters=16, kernel_size= kz_selected, padding='same', name='conv2DT_4'))
  model.add(tf.keras.layers.LeakyReLU(alpha=alpha_selected, name='act_8'))
  model.add(tf.keras.layers.UpSampling2D(size=(2,2), name='UpS2D_4'))
  model.add(tf.keras.layers.Cropping2D(cropping=((1,0),(0,0)), name='Cropping_4'))

  model.add(tf.keras.layers.Conv2D(filters=1, kernel_size= kz_selected, padding='same', activation='linear', name='output')) #output
  return model


def plot_loss_metric(history, params, _trial_id):
    # Obtener las métricas del entrenamiento y la validación
    train_loss = history.history['loss']
    train_mae = history.history['mean_absolute_error']
    val_loss = history.history['val_loss']
    val_mae = history.history['val_mean_absolute_error']

    # Graficar las métricas
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(25, 15))
    kz, lr, a = params['kernel_size'], params['learning_rate'], params['alpha']
    plt.suptitle(f'trialId:{_trial_id} | kz:{kz} | lr:{lr} | alpha:{a}', fontsize=20, y=0.98)

    plt.subplot(1, 2, 1)
    plt.plot(epochs, np.log(train_loss), label='Training Loss')
    plt.plot(epochs, np.log(val_loss), label='Validation Loss')
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('$log(loss)$', fontsize=20)
    plt.title('Training and Validation Loss', fontsize=20)
    plt.legend(fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, np.log(train_mae), label='Training MAE')
    plt.plot(epochs, np.log(val_mae), label='Validation MAE')
    plt.xlabel('Epoch', fontsize=20)
    plt.ylabel('$log(MAE)$', fontsize=20)
    plt.title(f'Training and Validation MAE', fontsize=20)
    plt.legend(fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()
    plt.savefig(f'optimize/loss_val_imgs/{variable}/{state}/{variable}_{state}_id_{_trial_id-1}.png')        


def objective(trial):
    train_ds, valid_ds, test_ds = get_data()
    model = create_model(trial)
    optimizer = create_optimizer(trial)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, mode='auto', verbose=1)
    model.compile(optimizer=optimizer,loss=tf.keras.losses.MeanSquaredError(), metrics=tf.keras.losses.MeanAbsoluteError())
    history = model.fit(train_ds, validation_data=valid_ds, epochs=EPOCHS, callbacks=[early_stop], verbose=0)
    plot_loss_metric(history, trial.__dict__['_cached_frozen_trial'].params,trial.__dict__['_trial_id'])
    mae = model.evaluate(test_ds)[1]
    if np.isnan(mae):
        return np.inf
    return mae



if __name__ == "__main__":
    study = optuna.create_study(study_name=f'optimizing_parameters_{variable}_{state}', storage=f'sqlite:///optimize/db/optimizing_parameters_state_{state}.db', load_if_exists=True)
    study.optimize(objective, n_trials=1)