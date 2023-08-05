import warnings
import argparse
import tensorflow as tf
tf.random.set_seed(123)
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(prefix_chars='--')
parser.add_argument('--state', type=int, help='State to predict')
parser.add_argument('--variable', type=str, help='Variable to predict')


args = parser.parse_args()
state = args.state
variable = args.variable
EPOCHS = 100
BATCH_SIZE = 32


def create_model(alpha, kernel_size):
    model = tf.keras.models.Sequential(name=f'CNN-Weather-Forecasting-training-{variable}{state}')
    model.add(tf.keras.layers.Input(shape=(73,144,1), name='input')) #input
    #encoder
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size= kernel_size, padding='same', name='conv2D_1'))
    model.add(tf.keras.layers.LeakyReLU(alpha=alpha, name='act_1'))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2),padding='same', name='AvP_1'))

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size= kernel_size, padding='same', name='conv2D_2'))
    model.add(tf.keras.layers.LeakyReLU(alpha=alpha, name='act_2'))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2),padding='same', name='AvP_2'))

    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size= kernel_size, padding='same', name='conv2D_3'))
    model.add(tf.keras.layers.LeakyReLU(alpha=alpha, name='act_3'))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2),padding='same', name='AvP_3'))

    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size= kernel_size, padding='same', name='conv2D_4'))
    model.add(tf.keras.layers.LeakyReLU(alpha=alpha, name='act_4'))
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2),padding='same', name='AvP_4'))

    #decoder 
    model.add(tf.keras.layers.Conv2DTranspose(filters=128, kernel_size= kernel_size, padding='same', name='conv2DT_1'))
    model.add(tf.keras.layers.LeakyReLU(alpha=alpha, name='act_5'))
    model.add(tf.keras.layers.UpSampling2D(size=(2,2), name='UpS2D_1'))


    model.add(tf.keras.layers.Conv2DTranspose(filters=64, kernel_size= kernel_size, padding='same', name='conv2DT_2'))
    model.add(tf.keras.layers.LeakyReLU(alpha=alpha, name='act_6'))
    model.add(tf.keras.layers.UpSampling2D(size=(2,2), name='UpS2D_2'))
    model.add(tf.keras.layers.Cropping2D(cropping=((1,0),(0,0)), name='Cropping_1'))

    model.add(tf.keras.layers.Conv2DTranspose(filters=32, kernel_size= kernel_size, padding='same', name='conv2DT_3'))
    model.add(tf.keras.layers.LeakyReLU(alpha=alpha, name='act_7'))
    model.add(tf.keras.layers.UpSampling2D(size=(2,2), name='UpS2D_3'))
    model.add(tf.keras.layers.Cropping2D(cropping=((1,0),(0,0)), name='Cropping_3'))

    model.add(tf.keras.layers.Conv2DTranspose(filters=16, kernel_size= kernel_size, padding='same', name='conv2DT_4'))
    model.add(tf.keras.layers.LeakyReLU(alpha=alpha, name='act_8'))
    model.add(tf.keras.layers.UpSampling2D(size=(2,2), name='UpS2D_4'))
    model.add(tf.keras.layers.Cropping2D(cropping=((1,0),(0,0)), name='Cropping_4'))

    model.add(tf.keras.layers.Conv2D(filters=1, kernel_size= kernel_size, padding='same', activation='linear', name='output')) #output
    
    return model



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

    split_ratio_train = 0.8
    split_ratio_val = 0.2

    num_samples = len(X)
    num_train = int(split_ratio_train * num_samples)
    num_val = int(split_ratio_val * num_samples)

    train_data = X[:num_train]
    val_data = X[num_train:num_train+num_val]

    train_labels = Y[:num_train]
    val_labels = Y[num_train:num_train+num_val]

    train_ds = tf.data.Dataset.from_tensor_slices(tensors=(train_data, train_labels))
    train_ds = train_ds.shuffle(60000).batch(batch_size=BATCH_SIZE)

    valid_ds = tf.data.Dataset.from_tensor_slices(tensors=(val_data, val_labels))
    valid_ds = valid_ds.shuffle(10000).batch(batch_size=BATCH_SIZE)

    del X, Y
    return train_ds, valid_ds


def plot_loss_metric(history):
    # Obtener las métricas del entrenamiento y la validación
    train_loss = history.history['loss']
    train_mae = history.history['mean_absolute_error']
    val_loss = history.history['val_loss']
    val_mae = history.history['val_mean_absolute_error']

    # Graficar las métricas
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(25, 15))
    plt.suptitle(f'Train {variable}{state}', fontsize=20, y=0.98)

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
    plt.savefig(f'./cnn-models/loss_val_imgs/{state}/{variable}_{state}.png')    

if __name__ == "__main__":
    df_best_parameters = pd.read_csv('./optimize/best_cnn_params.csv')
    best_parameters = df_best_parameters[(df_best_parameters['state'] == state) & (df_best_parameters['variable'] == variable)]
    alpha, kernel_size, learning_rate = (best_parameters[['params_alpha', 'params_kernel_size',  'params_learning_rate']].values)[0]
    train_ds, valid_ds = get_data()
    model = create_model(alpha=alpha, kernel_size=eval(kernel_size))
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) ,loss = tf.keras.losses.MeanSquaredError(), metrics = tf.keras.losses.MeanAbsoluteError())
    history = model.fit(train_ds, validation_data=valid_ds, epochs=EPOCHS, verbose=1)
    model.save(f'./cnn-models/{state}/{variable}{state}.h5')
    plot_loss_metric(history=history)
