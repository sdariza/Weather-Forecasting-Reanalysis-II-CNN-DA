import pandas as pd
import numpy as np
import netCDF4 as nc
import warnings
import argparse
import plotly.graph_objects as go
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
tf.random.set_seed(123)
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(prefix_chars='--')
parser.add_argument('--state', type=int, help='State to predict')
parser.add_argument('--variable', type=str, help='Variable to predict')


args = parser.parse_args()
state = args.state
variable = args.variable
EPOCHS = 100
BATCH_SIZE = 32

state, variable = 6, 'air'

def create_model(alpha, kernel_size):
    inputs = tf.keras.layers.Input(shape=(73, 144, 1), name='input')
    # Encoder
    conv1 = tf.keras.layers.Conv2D(32, kernel_size=kernel_size, padding='same', name='conv2D_1')(inputs)
    act1 = tf.keras.layers.LeakyReLU(alpha=alpha, name='act_1')(conv1)
    conv2 = tf.keras.layers.Conv2D(32, kernel_size=kernel_size, padding='same', name='conv2D_2')(act1)
    act2 = tf.keras.layers.LeakyReLU(alpha=alpha, name='act_2')(conv2)
    pool1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same', name='AvP_1')(act2)

    conv3 = tf.keras.layers.Conv2D(64, kernel_size=kernel_size, padding='same', name='conv2D_3')(pool1)
    act3 = tf.keras.layers.LeakyReLU(alpha=alpha, name='act_3')(conv3)
    conv4 = tf.keras.layers.Conv2D(64, kernel_size=kernel_size, padding='same', name='conv2D_4')(act3)
    act4 = tf.keras.layers.LeakyReLU(alpha=alpha, name='act_4')(conv4)
    pool2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same', name='AvP_2')(act4)

    # Decoder
    deconv_1 = tf.keras.layers.Conv2DTranspose(128, kernel_size=kernel_size, padding='same', name='conv2DT_1')(pool2)
    act5 = tf.keras.layers.LeakyReLU(alpha=alpha, name='act_5')(deconv_1)
    deconv2 = tf.keras.layers.Conv2DTranspose(64, kernel_size=kernel_size, padding='same', name='conv2DT_2')(act5)
    act6 = tf.keras.layers.LeakyReLU(alpha=alpha, name='act_6')(deconv2)
    upsample1 = tf.keras.layers.UpSampling2D(size=(2, 2), name='UpS2D_1')(act6)
    crop1 = tf.keras.layers.Cropping2D(cropping=((1, 0), (0, 0)), name='Cropping_1')(upsample1)
    concat1 = tf.keras.layers.concatenate([act4, crop1], axis=-1, name='concat_1')  # Conexión de salto

    deconv3 = tf.keras.layers.Conv2DTranspose(64, kernel_size=kernel_size, padding='same', name='conv2DT_3')(concat1)
    act7 = tf.keras.layers.LeakyReLU(alpha=alpha, name='act_7')(deconv3)
    deconv4 = tf.keras.layers.Conv2DTranspose(32, kernel_size=kernel_size, padding='same', name='conv2DT_4')(act7)
    act8 = tf.keras.layers.LeakyReLU(alpha=alpha, name='act_8')(deconv4)
    upsample2 = tf.keras.layers.UpSampling2D(size=(2, 2), name='UpS2D_3')(act8)
    crop2 = tf.keras.layers.Cropping2D(cropping=((1, 0), (0, 0)), name='Cropping_3')(upsample2)
    concat2 = tf.keras.layers.concatenate([act2, crop2], axis=-1, name='concat_2')  # Conexión de salto

    deconv5 = tf.keras.layers.Conv2DTranspose(32, kernel_size=kernel_size, padding='same', name='conv2DT_5')(concat2)
    act9 = tf.keras.layers.LeakyReLU(alpha=alpha, name='act_9')(deconv5)
    deconv6 = tf.keras.layers.Conv2DTranspose(32, kernel_size=kernel_size, padding='same', name='conv2DT_6')(act9)
    act10 = tf.keras.layers.LeakyReLU(alpha=alpha, name='act_10')(deconv6)
    output = tf.keras.layers.Conv2D(1, kernel_size=kernel_size, padding='same', activation='linear', name='output')(act10)

    model = tf.keras.models.Model(inputs=inputs, outputs=output, name='CNN-Weather-Forecasting-Training')
    return model


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


    train_labels = Y[:num_train]
    val_labels = Y[num_train:num_train + num_val]

    train_ds = tf.data.Dataset.from_tensor_slices(tensors=(train_data, train_labels))
    train_ds = train_ds.shuffle(buffer_size=num_train, seed=123).batch(batch_size=BATCH_SIZE)

    valid_ds = tf.data.Dataset.from_tensor_slices(tensors=(val_data, val_labels))
    valid_ds = valid_ds.shuffle(buffer_size=num_val, seed=123).batch(batch_size=BATCH_SIZE)

    del X, Y
    return train_ds, valid_ds



def plot_loss_metric(history):
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
        f'data_driven/cnn-models/loss_val_imgs/{state}/loss_{variable}_{state}.pdf',
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
        f'data_driven/cnn-models/loss_val_imgs/{state}/val_{variable}_{state}.pdf',
        scale=2)




if __name__ == "__main__":
    # df_best_parameters = pd.read_csv('./data_driven/optimize/best_cnn_params.csv')
    # best_parameters = df_best_parameters[(df_best_parameters['state'] == state) & (
    #     df_best_parameters['variable'] == variable)]
    # alpha, kernel_size, learning_rate = (best_parameters[[
    #                                      'alpha', 'kernel_size', 'learning_rate']].values)[0]

    # train_ds, valid_ds = get_data()
    # model = create_model(alpha=alpha, kernel_size=eval(kernel_size))
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    #               loss=tf.keras.losses.MeanSquaredError(), metrics=tf.keras.losses.MeanAbsoluteError())
    # early_stop = EarlyStopping(
    #     monitor='val_loss', patience=5, mode='auto', verbose=1)
    # history = model.fit(train_ds, validation_data=valid_ds,
    #                     epochs=EPOCHS, callbacks=[early_stop], verbose=1)
    # model.save(f'./data_driven/cnn-models/{state}/{variable}{state}.h5')
    # plot_loss_metric(history=history)


    alpha, kernel_size, learning_rate = 0.04, (4,4), 0.0002802720366666156
    train_ds, valid_ds = get_data()
    model = create_model(alpha=alpha, kernel_size=kernel_size)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.MeanSquaredError(), metrics=tf.keras.losses.MeanAbsoluteError())
    early_stop = EarlyStopping(
        monitor='val_loss', patience=5, mode='auto', verbose=1)
    history = model.fit(train_ds, validation_data=valid_ds,
                        epochs=EPOCHS, callbacks=[early_stop], verbose=1)
    model.save(f'./data_driven/cnn-models/{state}/{variable}{state}.h5')
    plot_loss_metric(history=history)
