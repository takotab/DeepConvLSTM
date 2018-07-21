import os
import pickle as cp

import numpy as np
import wget

from .preprocess_data import generate_data
from .sliding_window import sliding_window


def get_data(config, _3_dim = False):
    Opportunity_UCIDataset_zip_dir = "OpportunityUCIDataset.zip"
    if not os.path.isfile(Opportunity_UCIDataset_zip_dir):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00226" \
              "/OpportunityUCIDataset" \
              ".zip"
        print("downloading data")
        Opportunity_UCIDataset_zip_dir = wget.download(url)
        print("data downloaded")

    data_dir = os.path.join("dataset", 'Opportunity' + ".pckl")
    if not os.path.isfile(data_dir):
        print("now preparing..")
        generate_data(Opportunity_UCIDataset_zip_dir,
                      data_dir
                      )
        print("data prepared")

    print("Loading data...")
    x_train, y_train, x_test, y_test = load_dataset(data_dir)
    assert config.NB_SENSOR_CHANNELS == x_train.shape[1]

    # Sensor data is segmented using a sliding window mechanism
    x_test, y_test = opp_sliding_window(x_test, y_test,
                                        config.SLIDING_WINDOW_LENGTH,
                                        config.SLIDING_WINDOW_STEP)
    print(" ..after sliding window (testing): inputs {0}, targets {1}".format(x_test.shape,
                                                                              y_test.shape))
    x_train, y_train = opp_sliding_window(x_train, y_train,
                                          config.SLIDING_WINDOW_LENGTH,
                                          config.SLIDING_WINDOW_STEP)
    print(" ..after sliding window (training): inputs {0}, targets {1}".format(x_train.shape,
                                                                               y_train.shape))

    # Data is reshaped
    if _3_dim:
        # for starters flattened
        x_test = x_test.reshape((-1, config.SLIDING_WINDOW_LENGTH * config.NB_SENSOR_CHANNELS))
        x_train = x_train.reshape((-1, config.SLIDING_WINDOW_LENGTH * config.NB_SENSOR_CHANNELS))
    else:
        x_test = x_test.reshape((-1, 1, config.SLIDING_WINDOW_LENGTH, config.NB_SENSOR_CHANNELS))
        x_train = x_train.reshape((-1, 1, config.SLIDING_WINDOW_LENGTH, config.NB_SENSOR_CHANNELS))

    features_train = {"windows": x_train}
    features_test = {"windows": x_test}

    return features_train, y_train, features_test, y_test


def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.int)


def load_dataset(filename):
    f = open(filename, 'rb')
    data = cp.load(f)
    f.close()

    X_train, y_train = data[0]
    X_test, y_test = data[1]

    print(" ..from file {}".format(filename))
    print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.int)
    y_test = y_test.astype(np.int)

    return X_train, y_train, X_test, y_test
