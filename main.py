import numpy as np
import tensorflow as tf

import config
import oppertunity_data
from metric import extra_metrics

if __name__ == "__main__":

    features_train, y_train, features_test, y_test = oppertunity_data.get_data(config,
                                                                               _3_dim = True)

    classifier = tf.estimator.DNNClassifier(hidden_units = [1000, 300],
                                            feature_columns = [
                                                tf.feature_column.numeric_column(key = 'windows',
                                                                                 shape =
                                                                                 int(features_train[
                                                                                         "windows"].shape[
                                                                                         -1]))
                                                ],

                                            model_dir =
                                            ".trained_models/simple_model_1000x300",
                                            n_classes = int(np.max(y_train)
                                                            ) + 1
                                            )
    tf.logging.set_verbosity('INFO')
    classifier = tf.contrib.estimator.add_metrics(
            classifier,
            extra_metrics
            )
    input_fn = tf.estimator.inputs.numpy_input_fn(x = features_train,
                                                  y = y_train,
                                                  shuffle = True,
                                                  )
    input_fn_test = tf.estimator.inputs.numpy_input_fn(x = features_test,
                                                       y = y_test,
                                                       shuffle = False,
                                                       )
    train_spec = tf.estimator.TrainSpec(input_fn = input_fn,
                                        max_steps = 10000)

    eval_spec = tf.estimator.EvalSpec(input_fn = input_fn_test,
                                      throttle_secs = 60 * 10,
                                      start_delay_secs = 60 * 5)
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
