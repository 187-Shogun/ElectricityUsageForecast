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
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import tensorflow as tf
import pandas as pd
import numpy as np
import os


# Global vars:
EPOCHS = 100
PATIENCE = 10
RANDOM_SEED = 420
BATCH_SIZE = 32
SEQ_LEN = 256  # 432 = 3 Days worth of data
TARGETS = 36  # 36 = 3 Hrs Ahead Predictions
FEATURES = 50  # Max -> 383, Min -> 1
CONV_SIZE = 16
LOGS_DIR = os.path.join(os.getcwd(), 'logs')
MODELS_DIR = os.path.join(os.getcwd(), 'models')
RAW_DATA = os.path.join(os.getcwd(), 'data', 'energy_data.csv')
SCALER = StandardScaler()
AUTOTUNE = tf.data.AUTOTUNE
plt.style.use('dark_background')


def get_model_version_name(model_name: str) -> str:
    """ Generate a unique name using timestamps. """
    ts = datetime.now(timezone('America/Costa_Rica'))
    run_id = ts.strftime("%Y%m%d-%H%M%S")
    return f"{model_name}_v.{run_id}"


def extract_features(df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    """ Perform feature engineering on a vector and return multiple derivatives back. """
    # Compute lag values:
    f_index = list(df.columns).index(feature_name) + 1
    features = dict()

    for i in (3, 6, 12, 36):
        cname = f"{feature_name}Lag{i}"
        features[cname] = df[feature_name].shift(i).values

    # Compute moving averages values:
    for i in (3, 6, 12, 36):
        cname = f"{feature_name}MA{i}"
        features[cname] = df[feature_name].rolling(i).mean()

    # Compute diff values:
    df_features = pd.DataFrame(features)
    df = df.iloc[:, :f_index].join(df_features).join(df.iloc[:, f_index:])
    features = dict()
    for i in (3, 6, 12, 36):
        cname = f"{feature_name}Diff{i}"
        features[cname] = df[feature_name] / df[f"{feature_name}Lag{i}"]

    # Compute MA Diff values:
    for i in (3, 6):
        cname = f"{feature_name}MADiff{i}"
        features[cname] = df[f"{feature_name}MA{i}"] / df[f"{feature_name}MA{i*2}"]

    df_features = pd.DataFrame(features)
    df = df.iloc[:, :f_index].join(df_features).join(df.iloc[:, f_index:])
    return df


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """ Extract categorical features derived from timestamps. """
    # Some extra preprocessing:
    df['ampm'] = df.date.apply(lambda x: 0 if x.hour in range(6, 18) else 1).astype(float)
    df['dow'] = df.date.apply(lambda x: x.strftime('%w')).astype(int)
    df = df.join(pd.get_dummies(df.dow).astype(float))
    df = df.drop(columns=['dow'])
    return df


def fetch_raw_data(file_path: str) -> pd.DataFrame:
    """ Read raw data into a pandas df. """
    # Read raw data from csv file:
    df = pd.read_csv(file_path)
    df.date = pd.to_datetime(df.date)
    df = extract_time_features(df)
    df = df.sort_values('date').drop(columns=['date', 'lights', 'rv1', 'rv2'])
    return df


def fetch_preprocessed_data(df: pd.DataFrame, targets: list) -> pd.DataFrame:
    """ Preprocess raw data to drop and create new features. """
    # Perform feature engineering:
    for col in df.columns[:-8]:  # The last 8 values are the time-derived features.
        df = extract_features(df, col)

    # Move prediction columns to the front of the dataset:
    for i in targets:
        df.insert(0, i, df.pop(i))

    # Scale features using standard scaler:
    df = df.replace([np.nan, np.inf, -np.inf], 0.0)
    SCALER.fit(df.values)
    df = pd.DataFrame(SCALER.transform(df.values), columns=list(df.columns))
    return df


def get_windows(df: pd.DataFrame, targets: list) -> tuple:
    """ Compute sliding windows over a dataframe. """
    X = df.values
    y = df[targets].values
    features_list = []
    labels_list = []
    total_available_windows = int(df.shape[0] - SEQ_LEN - TARGETS)

    for i in tqdm(list(range(total_available_windows)), desc="Slicing sequence into windows"):
        # Loop over the entire sequence and create the windows:
        features = X[i: i + SEQ_LEN + TARGETS]
        labels = y[i: i + SEQ_LEN + TARGETS]
        features_list.append(features[:SEQ_LEN])
        labels_list.append(labels[SEQ_LEN:])

    return features_list, labels_list


def fetch_dataset(targets: list, train_split: float = 0.9) -> tuple:
    """ Cast pandas dataframe into TF Datasets. Returns Test, Validation and Test Datasets. """
    # Fetch raw data:
    df = fetch_raw_data(RAW_DATA)
    df = fetch_preprocessed_data(df, targets=targets)
    df = df.iloc[:, :FEATURES]
    X, y = get_windows(df, targets)

    # Configure a TF dataset:
    X_ds = tf.data.Dataset.from_tensor_slices(X)
    y_ds = tf.data.Dataset.from_tensor_slices(y)
    dataset = tf.data.Dataset.zip((X_ds, y_ds)).batch(BATCH_SIZE)
    dataset = dataset.shuffle(df.shape[0]).prefetch(AUTOTUNE)

    # Return split datasets:
    total_records = len(dataset)
    training_size = int(total_records * train_split)
    val_size = int((total_records - training_size) / 2)
    return (
        dataset.take(training_size),
        dataset.skip(training_size).take(val_size),
        dataset.skip(training_size).skip(val_size),
        df
    )


def viz_features():
    """ """
    # Check out some data:
    # features_cols = list(df.columns)[1:10]
    # f, ax = plt.subplots(len(features_cols), 1)
    # for i, x in enumerate(ax):
    #     x.plot(df[features_cols[i]].values[9000:10_000])
    #     x.set_title(features_cols[i])


def get_baseline_regressor() -> tf.keras.models.Model:
    """ Build a baseline network. """
    # Assemble the model:
    model = tf.keras.models.Sequential(
        name='Baseline-NN',
        layers=[
            tf.keras.layers.Flatten(input_shape=[SEQ_LEN, FEATURES]),
            tf.keras.layers.Dense(SEQ_LEN, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(TARGETS)
        ]
    )

    # Compile it and return it:
    model.compile(
        loss='mae',
        optimizer=tf.keras.optimizers.Adam()
    )
    return model


def get_custom_network() -> tf.keras.models.Model:
    """ Build a custom network. """
    # Assemble the model:
    model = tf.keras.models.Sequential(
        name='Custom-DNN',
        layers=[
            tf.keras.layers.Flatten(input_shape=[SEQ_LEN, FEATURES]),
            tf.keras.layers.Dense(SEQ_LEN/6, activation='relu'),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(SEQ_LEN/3, activation='relu'),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(SEQ_LEN/6, activation='relu'),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(TARGETS)
        ]
    )

    # Compile it and return it:
    model.compile(
        loss='mae',
        optimizer=tf.keras.optimizers.SGD(lr=0.03, momentum=0.9, nesterov=True)
    )
    return model


def get_wavenet() -> tf.keras.models.Model:
    """ Build a custom network. """
    # Assemble the model:
    model = tf.keras.models.Sequential(name='Wavenet-CNN')
    model.add(tf.keras.layers.InputLayer(input_shape=[SEQ_LEN, FEATURES]))
    for rate in (1, 2, 4, 8, 16, 32, 64, 128) * 2:
        model.add(
            tf.keras.layers.Conv1D(
                filters=CONV_SIZE,
                kernel_size=2,
                padding='causal',
                activation='relu',
                dilation_rate=rate
            )
        )

    model.add(tf.keras.layers.Conv1D(filters=TARGETS, kernel_size=1))
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dense(TARGETS))

    # Compile it and return it:
    model.compile(
        loss='mae',
        optimizer=tf.keras.optimizers.Adam()
    )
    return model


def get_recurrent_network() -> tf.keras.models.Model:
    """ Build a recurrent network using LSTM cells. """
    # Assemble the model:
    model = tf.keras.models.Sequential(
        name='LSTM-RNN',
        layers=[
            tf.keras.layers.LSTM(SEQ_LEN, input_shape=[SEQ_LEN, FEATURES], dropout=0.2, return_sequences=True),
            tf.keras.layers.LSTM(SEQ_LEN, dropout=0.2),
            tf.keras.layers.Dense(TARGETS)
        ]
    )

    # Compile it and return it:
    model.compile(
        loss='mae',
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


def plot_prediction_results(axis, x_values: np.array, y_true: np.array, predictions: np.array):
    """ Pass in some predictions and plot results against ground truth values. """
    features = dict(zip(list(range(0, len(x_values))), x_values))
    true_labels = dict(zip(list(range(len(x_values), len(x_values) + len(y_true))), y_true))
    predicted_labels = dict(zip(list(range(len(x_values), len(x_values) + len(y_true))), predictions))

    axis.plot(list(features.keys()), list(features.values()))
    axis.plot(list(true_labels.keys()), list(true_labels.values()), 'y--')
    axis.plot(list(predicted_labels.keys()), list(predicted_labels.values()), 'r--')
    return axis


def plot_multiple_predictions(model: tf.keras.models.Model, n_samples: int, samples: list):
    """ Plot a single figure with multiple predictions picked at random. """
    f, ax = plt.subplots(n_samples, 1)
    for axz, sample in zip(ax, samples):
        features = [z[0] for z in sample[0]]
        labels = list(np.reshape(sample[1], TARGETS))
        predictions = list(model.predict(np.reshape(sample[0], (1, SEQ_LEN, FEATURES)))[0])
        plot_prediction_results(axz, features, labels, predictions)
    return plt.show()


def dev():
    """ Test some shit. """
    # Train baseline model:
    X_train, X_val, X_test, df = fetch_dataset(['Appliances'])
    model = get_baseline_regressor()
    model = train_model(model, X_train, X_val)
    # model = tf.keras.models.load_model(r'models/Baseline-NN_v.20220412-194255.h5')

    # Test the model:
    n_samples = 4
    samples = [(r[0], s[0]) for r, s in X_test.shuffle(1000).take(n_samples).as_numpy_iterator()]
    f, ax = plt.subplots(n_samples, 1)
    for axz, sample in zip(ax, samples):
        features = [z[0] for z in sample[0]]
        labels = list(np.reshape(sample[1], TARGETS))
        predictions = list(model.predict(np.reshape(sample[0], (1, SEQ_LEN, FEATURES)))[0])
        plot_prediction_results(axz, features, labels, predictions)
    return plt.show()


def main():
    """ Run script. """
    # Train baseline model:
    X_train, X_val, X_test, df = fetch_dataset(['Appliances'])
    model = get_baseline_regressor()
    model = train_model(model, X_train, X_val)
    # model = tf.keras.models.load_model(r'models/Baseline-NN_v.20220414-183217.h5')

    # Test the model:
    n_samples = 4
    samples = [(r[0], s[0]) for r, s in X_test.shuffle(1000).take(n_samples).as_numpy_iterator()]
    plot_multiple_predictions(model, n_samples, samples)
    return {}


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    plt.rcParams.update({'font.size': 8})
    pd.set_option("expand_frame_repr", False)
    dev()
