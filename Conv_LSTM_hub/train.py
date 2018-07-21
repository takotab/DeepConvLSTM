import os

import tensorflow as tf
import tensorflow_hub as hub
import wget

from . import model
from .preprocess_data import generate_data
from .sliding_window import sliding_window


def train(config, dataset = 'Opportunity'):
    if dataset is not 'Opportunity':
        raise NotImplementedError()
    Opportunity_UCIDataset_zip_dir = "OpportunityUCIDataset.zip"
    if not os.path.isfile(Opportunity_UCIDataset_zip_dir):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00226" \
              "/OpportunityUCIDataset" \
              ".zip"
        print("downloading data")
        Opportunity_UCIDataset_zip_dir = wget.download(url)
        print("data downloaded")

    data_dir = os.path.join("dataset", dataset + ".pckl")
    if not os.path.isfile(data_dir):
        print("now preparing..")
        generate_data(Opportunity_UCIDataset_zip_dir,
                      data_dir
                      )
        print("data prepared")

    print("Loading data...")
    X_train, y_train, X_test, y_test = load_dataset(data_dir)
    assert config.NB_SENSOR_CHANNELS == X_train.shape[1]

    # Sensor data is segmented using a sliding window mechanism
    X_test, y_test = opp_sliding_window(X_test, y_test,
                                        config.SLIDING_WINDOW_LENGTH,
                                        config.SLIDING_WINDOW_STEP)
    print(" ..after sliding window (testing): inputs {0}, targets {1}".format(X_test.shape,
                                                                              y_test.shape))

    # Data is reshaped since the input of the network is a 4 dimension tensor
    X_test = X_test.reshape((-1, 1, config.SLIDING_WINDOW_LENGTH, config.NB_SENSOR_CHANNELS))
    spec = model.make_model()
    _train(spec)
    # TODO: use estimator https://www.tensorflow.org/hub/api_docs/python/hub/LatestModuleExporter


def _train(spec):
    with tf.Graph().as_default():
        m = hub.Module(spec)

        for name in m.variable_map:
            print(name)

            # p_embeddings = tf.placeholder(tf.float32)

        with tf.Session() as sess:
            sess.run([load_embeddings], feed_dict = {p_embeddings: embeddings})
        m.export(export_path, sess)


def opp_sliding_window(data_x, data_y, ws, ss):
    data_x = sliding_window(data_x, (ws, data_x.shape[1]), (ss, 1))
    data_y = np.asarray([[i[-1]] for i in sliding_window(data_y, ws, ss)])
    return data_x.astype(np.float32), data_y.reshape(len(data_y)).astype(np.uint8)


def load_dataset(filename):
    f = file(filename, 'rb')
    data = cp.load(f)
    f.close()

    X_train, y_train = data[0]
    X_test, y_test = data[1]

    print(" ..from file {}".format(filename))
    print(" ..reading instances: train {0}, test {1}".format(X_train.shape, X_test.shape))

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # The targets are casted to int8 for GPU compatibility.
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)

    return X_train, y_train, X_test, y_test
