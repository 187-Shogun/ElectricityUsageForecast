"""
Title: main.py

Created on: 4/11/2022

Author: 187-Shogun

Encoding: UTF-8

Description: <Some description>
"""


from matplotlib import pyplot as plt
from datetime import datetime
from pytz import timezone
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import time


# Global vars:
EPOCHS = 25
PATIENCE = 5
RANDOM_SEED = 420
SEQ_LEN = 288
BATCH_SIZE = 32
TARGETS = 36
FEATURES = 2
LOGS_DIR = os.path.join(os.getcwd(), 'logs')
MODELS_DIR = os.path.join(os.getcwd(), 'models')
AUTOTUNE = tf.data.AUTOTUNE
plt.style.use('dark_background')


def get_model_version_name(model_name: str) -> str:
    """ Generate a unique name using timestamps. """
    ts = datetime.now(timezone('America/Costa_Rica'))
    run_id = ts.strftime("%Y%m%d-%H%M%S")
    return f"{model_name}_v.{run_id}"


def fetch_raw_data() -> pd.DataFrame:
    """ Read raw data into a pandas df. """
    # Read raw data and preprocess it:
    df = pd.read_csv(r"data/energy_data.csv")
    df.date = pd.to_datetime(df.date)
    df = df.sort_values('date')
    df['ampm'] = df.date.apply(lambda x: 0 if x.hour in range(6, 18) else 1).astype(float)
    df['doy'] = df.date.apply(lambda x: x.strftime('%j')).astype(int)
    df = df.join(pd.get_dummies(df.doy).astype(float))
    df = df.drop(columns=['doy', 'lights'])
    return df


def fetch_dataset(train_split: float = 0.8) -> tuple:
    """ Cast pandas dataframe into TF Datasets. Returns Test, Validation and Test Datasets. """
    # Fetch raw data:
    df = fetch_raw_data()
    X = df.iloc[:, 1:FEATURES + 1].values
    y = df[['Appliances']].values
    features_list = []
    labels_list = []

    for i in range(df.shape[0] - SEQ_LEN - TARGETS):
        features = X[i: i + SEQ_LEN + TARGETS]
        labels = y[i: i + SEQ_LEN + TARGETS]
        features_list.append(features[:SEQ_LEN])
        labels_list.append(labels[SEQ_LEN:])

    # Configure a TF dataset:
    X_ds = tf.data.Dataset.from_tensor_slices(features_list)
    y_ds = tf.data.Dataset.from_tensor_slices(labels_list)
    dataset = tf.data.Dataset.zip((X_ds, y_ds)).batch(BATCH_SIZE)
    dataset = dataset.shuffle(10_000).prefetch(AUTOTUNE)

    # Return split datasets:
    total_records = len(dataset)
    training_size = int(total_records * train_split)
    val_size = int((total_records - training_size) / 2)
    return (
        dataset.take(training_size),
        dataset.skip(training_size).take(val_size),
        dataset.skip(training_size).skip(val_size)
    )


def get_baseline_regressor() -> tf.keras.models.Model:
    """ Build a baseline network. """
    # Assemble the model:
    model = tf.keras.models.Sequential(
        name='Baseline-NN',
        layers=[

            tf.keras.layers.Flatten(input_shape=[SEQ_LEN, FEATURES]),
            tf.keras.layers.Dense(SEQ_LEN),
            tf.keras.layers.Dense(TARGETS)
        ]
    )

    # Compile it and return it:
    model.compile(
        loss='mse',
        optimizer=tf.keras.optimizers.Adam()
    )
    return model


def train_model(model: tf.keras.models.Model, train_ds, val_ds) -> tf.keras.models.Model:
    """ Pass a model and train it on a given dataset. """
    # Start training:
    version_name = get_model_version_name(model.name)
    tb_logs = tf.keras.callbacks.TensorBoard(os.path.join(LOGS_DIR, version_name))
    early_stop = tf.keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=.5, patience=int(PATIENCE/2))
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[tb_logs, early_stop, lr_scheduler])
    model.save(os.path.join(MODELS_DIR, f"{version_name}.h5"))
    return model


def plot_results(x_values: np.array, y_true: np.array, predictions: np.array):
    """ Pass in some predictions and plot results against ground truth values. """
    features = dict(zip(list(range(0, len(x_values))), x_values))
    true_labels = dict(zip(list(range(len(x_values), len(x_values) + len(y_true))), y_true))
    predicted_labels = dict(zip(list(range(len(x_values), len(x_values) + len(y_true))), predictions))

    plt.plot(list(features.keys()), list(features.values()))
    plt.plot(list(true_labels.keys()), list(true_labels.values()), 'y--')
    plt.plot(list(predicted_labels.keys()), list(predicted_labels.values()), 'r--')
    return plt.show()


def dev():
    """ Test some shit. """
    # Train baseline model:
    X_train, X_val, X_test = fetch_dataset()
    # model = get_baseline_regressor()
    # model = train_model(model, X_train, X_val)
    model = tf.keras.models.load_model(r'models/Baseline-NN_v.20220412-194255.h5')

    for x, y in X_test.shuffle(1000).take(3):
        features = [z[0].numpy() for z in x[0]]
        labels = np.reshape(y[0], TARGETS)
        predictions = model.predict(np.reshape(x[0], (1, SEQ_LEN, FEATURES)))[0]
        plot_results(features, labels, predictions)
        time.sleep(3)

    return {}


def main():
    """ Run script. """
    X_train, X_val, scaler = fetch_dataset(0.9)
    model = get_baseline_regressor()
    model.fit(X_train, validation_data=X_val, epochs=10)
    return {}


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    pd.set_option("expand_frame_repr", False)
    dev()
