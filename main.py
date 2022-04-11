"""
Title: main.py

Created on: 4/11/2022

Author: 187-Shogun

Encoding: UTF-8

Description: <Some description>
"""


import tensorflow as tf
import pandas as pd
import os


# Global vars:
EPOCHS = 100
PATIENCE = 12
RANDOM_SEED = 420
LOGS_DIR = os.path.join(os.getcwd(), 'logs')
MODELS_DIR = os.path.join(os.getcwd(), 'models')
AUTOTUNE = tf.data.AUTOTUNE


def fetch_dataset(seq_size: int, batch_size: int, targets: int, train_split: float) -> tuple:
    """ Read raw data from local file system and return a TF Dataset. """
    # Read raw data and preprocess it:
    df = pd.read_csv(r"data/energy_data.csv")
    df.date = pd.to_datetime(df.date)
    df = df[['date', 'T_out', 'lights']].sort_values('date')
    df.lights = df.lights.shift(-targets)
    df = df.dropna()

    # Configure a TF dataset:
    X_dataset = tf.data.Dataset.from_tensor_slices(df[['T_out']].values)
    y_dataset = tf.data.Dataset.from_tensor_slices(df[['lights']].values)
    X_windowed = X_dataset.window(size=seq_size, shift=1, drop_remainder=True)
    y_windowed = y_dataset.window(size=seq_size, shift=1, drop_remainder=True)
    X_batched = X_windowed.flat_map(lambda x: x.batch(seq_size))
    y_batched = y_windowed.flat_map(lambda x: x.batch(seq_size))
    dataset = tf.data.Dataset.zip((X_batched, y_batched))
    dataset = dataset.map(lambda x, y: (x, y[-1])).shuffle(10_000).batch(batch_size).prefetch(AUTOTUNE)

    # Return datasets
    total_records = len(y_windowed)
    training_size = int(total_records * train_split) // batch_size
    return dataset.take(training_size), dataset.skip(training_size)


def get_baseline_regressor() -> tf.keras.models.Model:
    """ Build a baseline network. """
    # Assemble the model:
    model = tf.keras.models.Sequential(
        name='Baseline-NN',
        layers=[
            tf.keras.layers.Flatten(input_shape=[36, 1]),
            tf.keras.layers.Dense(36),
            tf.keras.layers.Dense(1)
        ]
    )

    # Compile it and return it:
    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam()
    )
    return model


def main():
    """ Run script. """
    X_train, X_val = fetch_dataset(36, 32, 1, 0.9)
    model = get_baseline_regressor()
    model.fit(X_train, validation_data=X_val, epochs=10)
    return {}


if __name__ == "__main__":
    pd.set_option("expand_frame_repr", False)
    main()
