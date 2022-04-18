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
PATIENCE = 16
RANDOM_SEED = 420
BATCH_SIZE = 32
SEQ_LEN = 432  # 432 = 3 Days worth of data
TARGETS = 36  # 36 = 3 Hrs Ahead Predictions
FEATURES = 1  # Max -> 383, Min -> 1
CONV_SIZE = 8
LOGS_DIR = os.path.join(os.getcwd(), 'logs')
MODELS_DIR = os.path.join(os.getcwd(), 'models')
RAW_DATA = os.path.join(os.getcwd(), 'data', 'energy_data.csv')
PREDICTION_LABELS = ['Appliances']
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
    offsets = [1, 3, 12, 36, 144, 432]
    f_index = list(df.columns).index(feature_name) + 1
    features = dict()

    for i in offsets:
        # Compute lag values:
        cname = f"{feature_name}Lag{i}"
        features[cname] = df[feature_name].shift(i).values
        # Compute moving averages values:
        cname = f"{feature_name}MA{i}"
        features[cname] = df[feature_name].rolling(i).mean()

    # Compute diff values:
    df_features = pd.DataFrame(features)
    df = df.iloc[:, :f_index].join(df_features).join(df.iloc[:, f_index:])
    features = dict()

    for i in offsets:
        # Compute Diff values:
        cname = f"{feature_name}Diff{i}"
        features[cname] = df[feature_name] / df[f"{feature_name}Lag{i}"]
        # Compute MA Diff values:
        cname = f"{feature_name}MADiff{i}"
        try:
            features[cname] = df[f"{feature_name}MA{i}"] / df[f"{feature_name}MA{offsets[offsets.index(i)+1]}"]
        except IndexError:
            pass
        # Compute Lag Diff values:
        cname = f"{feature_name}LagDiff{i}"
        try:
            features[cname] = df[f"{feature_name}Lag{i}"] / df[f"{feature_name}Lag{offsets[offsets.index(i)+1]}"]
        except IndexError:
            pass

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
    df = df.replace([np.nan, np.inf, -np.inf], np.nan)
    df = df.dropna()
    SCALER.fit(df.values)
    df = pd.DataFrame(SCALER.transform(df.values), columns=list(df.columns))
    return df


def get_windows(df: pd.DataFrame, targets: list, seq_len: int) -> tuple:
    """ Compute sliding windows over a dataframe. """
    X = df.values
    y = df[targets].values
    features_list = []
    labels_list = []
    total_available_windows = int(df.shape[0] - seq_len - len(targets))

    for i in tqdm(list(range(total_available_windows)), desc="Slicing sequence into windows"):
        # Loop over the entire sequence and create the windows:
        features = X[i: i + seq_len + len(targets)]
        labels = y[i: i + seq_len + len(targets)]
        features_list.append(features[:seq_len])
        labels_list.append(labels[seq_len:])

    return features_list, labels_list


def fetch_dataset(features: int, targets: list, seq_len: int, train_split: float = 0.8) -> tuple:
    """ Cast pandas dataframe into TF Datasets. Returns Test, Validation and Test Datasets. """
    # Fetch raw data:
    df = fetch_raw_data(RAW_DATA)
    df = fetch_preprocessed_data(df, targets=targets)
    df = df.iloc[:, :features]
    X, y = get_windows(df, targets, seq_len)

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


def viz_features(df: pd.DataFrame, subset: tuple, index: int = 0):
    """ Visualize a set of features in a dataframe. """
    features_cols = list(df.columns)[subset[0]:subset[1]]
    f, ax = plt.subplots(len(features_cols), 1)

    for i, x in enumerate(ax):
        x.plot(df[features_cols[i]].values[index*1000:index*1000 + 1000])
        x.set_title(features_cols[i])
    return plt.show()


def get_baseline_regressor(sl: int, features: int, targets: int) -> tf.keras.models.Model:
    """ Build a baseline network. """
    # Assemble the model:
    model = tf.keras.models.Sequential(
        name=get_model_version_name(f"Baseline-NN-sl{sl}f{features}t{targets}"),
        layers=[
            tf.keras.layers.Flatten(input_shape=[sl, features]),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Dense(targets)
        ]
    )

    # Compile it and return it:
    model.compile(
        loss='mae',
        optimizer=tf.keras.optimizers.SGD(lr=0.03, momentum=0.9, nesterov=True)
    )
    return model


def get_custom_network(sl: int, features: int, targets: int) -> tf.keras.models.Model:
    """ Build a custom network. """
    # Assemble the model:
    model = tf.keras.models.Sequential(
        name=get_model_version_name(f"Custom-DNN-sl{sl}f{features}t{targets}"),
        layers=[
            tf.keras.layers.Conv1D(filters=sl // 4, kernel_size=CONV_SIZE, input_shape=[sl, features]),
            tf.keras.layers.LSTM(sl // 2, dropout=0.2, return_sequences=True),
            tf.keras.layers.LSTM(sl // 2, dropout=0.2),
            tf.keras.layers.Dense(targets)
        ]
    )

    # Compile it and return it:
    model.compile(
        loss='mae',
        optimizer=tf.keras.optimizers.Adam()
    )
    return model


def get_wavenet(sl: int, features: int, targets: int, filter_size: int) -> tf.keras.models.Model:
    """ Build a custom network. """
    # Assemble the model:
    model = tf.keras.models.Sequential(name=get_model_version_name(f"Wavenet-CNN-sl{sl}f{features}t{targets}"))
    model.add(tf.keras.layers.InputLayer(input_shape=[sl, features]))
    for rate in (1, 2, 4, 8, 16, 32, 64, 128, 256) * 2:
        model.add(
            tf.keras.layers.Conv1D(
                filters=filter_size,
                kernel_size=2,
                padding='causal',
                activation='relu',
                dilation_rate=rate
            )
        )

    model.add(tf.keras.layers.Conv1D(filters=targets, kernel_size=1))
    model.add(tf.keras.layers.GlobalAveragePooling1D())
    model.add(tf.keras.layers.Dense(targets))

    # Compile it and return it:
    model.compile(
        loss='mae',
        optimizer=tf.keras.optimizers.Adam()
    )
    return model


def get_recurrent_network(sl: int, features: int, targets: int) -> tf.keras.models.Model:
    """ Build a recurrent network using LSTM cells. """
    # Assemble the model:
    model = tf.keras.models.Sequential(
        name=get_model_version_name(f"LSTM-RNN-sl{sl}f{features}t{targets}"),
        layers=[
            tf.keras.layers.LSTM(sl // 2, input_shape=[sl, features], dropout=0.25, return_sequences=True),
            tf.keras.layers.LSTM(sl // 2, dropout=0.25),
            tf.keras.layers.Dense(targets)
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
    tb_logs = tf.keras.callbacks.TensorBoard(os.path.join(LOGS_DIR, model.name))
    early_stop = tf.keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True)
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=.5, patience=int(PATIENCE/2))
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[tb_logs, early_stop, lr_scheduler])
    model.save(os.path.join(MODELS_DIR, f"{model.name}.h5"))
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


def plot_multiple_predictions(model: tf.keras.models.Model, n_samples: int, samples: list, seq_len: int):
    """ Plot a single figure with multiple predictions picked at random. """
    f, ax = plt.subplots(n_samples, 1)
    for axz, sample in zip(ax, samples):
        features = [z[0] for z in sample[0]]
        labels = list(np.reshape(sample[1], TARGETS))
        predictions = list(model.predict(np.reshape(sample[0], (1, seq_len, FEATURES)))[0])
        plot_prediction_results(axz, features, labels, predictions)
    return plt.show()


# noinspection PyShadowingNames
def train_univariate_models() -> pd.DataFrame:
    """ Train different models on the same univariate dataset and compare results. """
    # Run a loop a training:
    results = []

    for sl in {64, 128, 256, 512}:
        SEQ_LEN = sl
        TARGETS = sl // 8
        models = [
            get_baseline_regressor(SEQ_LEN, FEATURES, TARGETS),
            get_wavenet(SEQ_LEN, FEATURES, TARGETS, CONV_SIZE),
            get_recurrent_network(SEQ_LEN, FEATURES, TARGETS),
            get_custom_network(SEQ_LEN, FEATURES, TARGETS)
        ]
        for model in models:
            X_train, X_val, X_test, _ = fetch_dataset(FEATURES, PREDICTION_LABELS, SEQ_LEN)
            print(f"Model: {model.name} -> Starting training...")
            model = train_model(model, X_train, X_val)
            results.append({
                "model_name": model.name,
                "model_type": "Univariate",
                "training_params": f"SL: {SEQ_LEN}, Targets: {TARGETS}, TargetName: {PREDICTION_LABELS}",
                "val": model.evaluate(X_val),
                "test": model.evaluate(X_test)
            })

    # Write training results in a CSV file:
    run_ts = datetime.now(timezone('America/Costa_Rica')).strftime("%Y%m%d-%H%M%S")
    df = pd.DataFrame(results)
    df.to_csv(f"results/univariate-training-results-{run_ts}.csv")
    return df


def dev():
    """ Test some shit. """
    # Train models:
    X_train, X_val, X_test, df = fetch_dataset(FEATURES, PREDICTION_LABELS, SEQ_LEN)
    results = train_univariate_models()
    model = tf.keras.models.load_model(f"models/{results.sort_values('test').iloc[0]['model_name']}.h5")

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
    X_train, X_val, X_test, df = fetch_dataset(FEATURES, PREDICTION_LABELS, SEQ_LEN)
    model = get_baseline_regressor(SEQ_LEN, FEATURES, TARGETS)
    model = train_model(model, X_train, X_val)
    # model = tf.keras.models.load_model(r'models/Baseline-NN_v.20220414-183217.h5')

    # Test the model:
    n_samples = 4
    samples = [(r[0], s[0]) for r, s in X_test.shuffle(1000).take(n_samples).as_numpy_iterator()]
    plot_multiple_predictions(model, n_samples, samples, SEQ_LEN)
    return {}


if __name__ == "__main__":
    np.random.seed(RANDOM_SEED)
    plt.rcParams.update({'font.size': 8})
    pd.set_option("expand_frame_repr", False)
    dev()
