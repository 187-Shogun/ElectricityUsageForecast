"""
Title: main.py

Created on: 4/11/2022

Author: 187-Shogun

Encoding: UTF-8

Description: <Some description>
"""


from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import os


# Global vars:
EPOCHS = 100
PATIENCE = 12
RANDOM_SEED = 420
SEQ_LEN = 36
BATCH_SIZE = 32
TARGETS = 1
FEATURES = 2
LOGS_DIR = os.path.join(os.getcwd(), 'logs')
MODELS_DIR = os.path.join(os.getcwd(), 'models')
AUTOTUNE = tf.data.AUTOTUNE


def get_scaler() -> MinMaxScaler:
    """ Get scaler for transforming data. """
    df = fetch_raw_data(TARGETS)
    scaler = MinMaxScaler()
    scaler.fit(df.T_out.values.reshape(-1, 1))
    return scaler


def fetch_raw_data(targets: int) -> pd.DataFrame:
    """ Read raw data into a pandas df. """
    # Read raw data and preprocess it:
    df = pd.read_csv(r"data/energy_data.csv")
    df.date = pd.to_datetime(df.date)
    df = df[['date', 'Appliances', 'T_out']].sort_values('date')
    df['ampm'] = df.date.apply(lambda x: 0 if x.hour in range(6, 18) else 1).astype(float)
    df['doy'] = df.date.apply(lambda x: x.strftime('%j')).astype(int)
    df.Appliances = df.Appliances.shift(-targets)
    df = df.dropna()
    df = df.join(pd.get_dummies(df.doy).astype(float))
    df = df.drop(columns=['date', 'doy'])
    return df


def fetch_preprocessed_data(df: pd.DataFrame) -> pd.DataFrame:
    """ Get pandas df after applying preprocessing steps. """
    scaler = get_scaler()
    df.T_out = scaler.transform(df.T_out.values.reshape(-1, 1)).reshape(-1)
    return df


def fetch_dataset(seq_size: int, batch_size: int, targets: int, features_count: int, train_split: float) -> tuple:
    """ Cast pandas dataframe into TF Datasets. """
    # Fetch raw data:
    df = fetch_raw_data(targets)
    df = fetch_preprocessed_data(df)
    labels = df.columns[:1]
    features = df.columns[1:]

    # Configure a TF dataset:
    X_dataset = tf.data.Dataset.from_tensor_slices(df[features[:features_count]].values)
    y_dataset = tf.data.Dataset.from_tensor_slices(df[labels[0]].values)
    X_windowed = X_dataset.window(size=seq_size, shift=1, drop_remainder=True)
    y_windowed = y_dataset.window(size=seq_size, shift=1, drop_remainder=True)
    X_batched = X_windowed.flat_map(lambda x: x.batch(seq_size))
    y_batched = y_windowed.flat_map(lambda x: x.batch(seq_size))
    dataset = tf.data.Dataset.zip((X_batched, y_batched))
    dataset = dataset.map(lambda x, y: (x, y[-1])).shuffle(10_000).batch(batch_size).prefetch(AUTOTUNE)

    # Return split datasets:
    total_records = len(y_windowed)
    training_size = int(total_records * train_split) // batch_size
    return dataset.take(training_size), dataset.skip(training_size)


def get_baseline_regressor(input_len: int, features: int) -> tf.keras.models.Model:
    """ Build a baseline network. """
    # Assemble the model:
    model = tf.keras.models.Sequential(
        name='Baseline-NN',
        layers=[

            tf.keras.layers.Flatten(input_shape=[input_len, features]),
            tf.keras.layers.Dense(input_len),
            tf.keras.layers.Dense(1)
        ]
    )

    # Compile it and return it:
    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam()
    )
    return model


def sample_random_seq(df: pd.DataFrame, time_steps: int = 3) -> tuple:
    """ Sample a given dataframe and return a sequence of a specified length. """
    # Prepare valid samples:
    df = df.loc[list(df.index)[:-SEQ_LEN + time_steps]]
    sample = np.random.choice(df.index)
    x_samples = []
    y_samples = []

    for i in range(time_steps):
        # Pick samples and apply format:
        sample_index = list(range(sample + i, sample + SEQ_LEN + i))
        results = fetch_preprocessed_data(df.loc[sample_index, ['Appliances', 'T_out', 'ampm']])
        x_samples.append(np.reshape(results[['T_out', 'ampm']].values, (1, SEQ_LEN, FEATURES)))
        y_samples.append(np.reshape(results[['Appliances']].values, (SEQ_LEN, 1)))

    return x_samples, y_samples


def dev():
    """ Test some shit. """
    # Train baseline model:
    X_train, X_val = fetch_dataset(SEQ_LEN, BATCH_SIZE, TARGETS, FEATURES, 0.9)
    model = get_baseline_regressor(SEQ_LEN, FEATURES)
    model.fit(X_train, validation_data=X_val, epochs=10)

    # Get samples to predict on:
    df = fetch_raw_data(TARGETS)
    x_samples, y_samples = sample_random_seq(df)
    return {}


def main():
    """ Run script. """
    X_train, X_val, scaler = fetch_dataset(SEQ_LEN, BATCH_SIZE, TARGETS, FEATURES, 0.9)
    model = get_baseline_regressor(SEQ_LEN, FEATURES)
    model.fit(X_train, validation_data=X_val, epochs=10)
    return {}


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    pd.set_option("expand_frame_repr", False)
    dev()
