import argparse

import tensorflow as tf

import config
import conv_model
import oppertunity_data
from metric import extra_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    params = parser.parse_args()

    features_train, y_train, features_test, y_test = oppertunity_data.get_data(config)
    params.learning_rate = 0.01

    classifier = tf.estimator.Estimator(model_fn = conv_model.model_fn,
                                        model_dir =
                                        ".trained_models/conv_model_0.0.1",
                                        params = params
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
                                      throttle_secs = 60 * 3,
                                      start_delay_secs = 60 * 5)
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
