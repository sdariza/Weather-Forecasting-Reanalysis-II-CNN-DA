import pandas as pd
from optimize import get_data
import warnings
import argparse
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


def create_model(kz, alpha):
    cnn_model = tf.keras.models.Sequential(
        layers=[
            # input
            tf.keras.layers.Input(shape=(73, 144, 1), name='input'),
            # encoder
            tf.keras.layers.Conv2D(filters=16, kernel_size=kz, padding='same', name='conv2D_1'),
            tf.keras.layers.LeakyReLU(alpha=alpha, name='act_1'),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same', name='AvP_1'),

            tf.keras.layers.Conv2D(filters=32, kernel_size=kz, padding='same', name='conv2D_2'),
            tf.keras.layers.LeakyReLU(alpha=alpha, name='act_2'),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same', name='AvP_2'),

            tf.keras.layers.Conv2D(filters=64, kernel_size=kz, padding='same', name='conv2D_3'),
            tf.keras.layers.LeakyReLU(alpha=alpha, name='act_3'),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same', name='AvP_3'),

            tf.keras.layers.Conv2D(filters=128, kernel_size=kz, padding='same', name='conv2D_4'),
            tf.keras.layers.LeakyReLU(alpha=alpha, name='act_4'),
            tf.keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same', name='AvP_4'),

            # decoder
            tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=kz, padding='same', name='conv2DT_1'),
            tf.keras.layers.LeakyReLU(alpha=alpha, name='act_5'),
            tf.keras.layers.UpSampling2D(size=(2, 2), name='UpS2D_1'),

            tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=kz, padding='same', name='conv2DT_2'),
            tf.keras.layers.LeakyReLU(alpha=alpha, name='act_6'),
            tf.keras.layers.UpSampling2D(size=(2, 2), name='UpS2D_2'),
            tf.keras.layers.Cropping2D(cropping=((1, 0), (0, 0)), name='Cropping_1'),

            tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=kz, padding='same', name='conv2DT_3'),
            tf.keras.layers.LeakyReLU(alpha=alpha, name='act_7'),
            tf.keras.layers.UpSampling2D(size=(2, 2), name='UpS2D_3'),
            tf.keras.layers.Cropping2D(cropping=((1, 0), (0, 0)), name='Cropping_3'),

            tf.keras.layers.Conv2DTranspose(filters=16, kernel_size=kz, padding='same', name='conv2DT_4'),
            tf.keras.layers.LeakyReLU(alpha=alpha, name='act_8'),
            tf.keras.layers.UpSampling2D(size=(2, 2), name='UpS2D_4'),
            tf.keras.layers.Cropping2D(cropping=((1, 0), (0, 0)), name='Cropping_4'),

            # output
            tf.keras.layers.Conv2D(filters=1, kernel_size=kz, padding='same', activation='linear',
                                   name='output')
        ]
        , name=f'CNN-Weather-Forecasting-Training-{variable}_{state}')
    return cnn_model


if __name__ == "__main__":
    df_best_parameters = pd.read_csv('./data_driven/optimize/best_cnn_params.csv')
    best_parameters = df_best_parameters[(df_best_parameters['state'] == state) & (
            df_best_parameters['variable'] == variable)]
    a, kernel_size_w, kernel_size_h, learning_rate = (best_parameters[[
        'alpha', 'kernel_size_w', 'kernel_size_h', 'learning_rate']].values)[0]
    kernel_size = (int(kernel_size_w), int(kernel_size_h))
    train_ds, valid_ds, _ = get_data(variable, state)
    model = create_model(kz=kernel_size, alpha=a)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=tf.keras.losses.MeanSquaredError(), metrics=tf.keras.losses.MeanAbsoluteError())
    early_stop = EarlyStopping(
        monitor='val_loss', patience=5, mode='auto', verbose=1)
    history = model.fit(train_ds, validation_data=valid_ds,
                        epochs=EPOCHS, callbacks=[early_stop], verbose=1)
    model.save(f'./data_driven/cnn-models/{state}/{variable}{state}.h5')
