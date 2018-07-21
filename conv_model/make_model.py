import tensorflow as tf

from . import utils  # pylint : ignore


def make_model(deep_t, params, mode):
    # Makes the custom model.
    # Design is the same as showcased on https://www.tensorflow.org/tutorials/layers
    X = utils.batch_norm(deep_t, mode = mode)

    X = utils.conv_2d(X, 5, "1")
    # X = tf.layers.dropout(
    #     inputs=X, rate=params.dropout, training=mode == tf.estimator.ModeKeys.TRAIN)
    X = utils.conv_2d(X, 10, "2")

    X = utils.conv_2d(X, 20, "3")

    X = tf.reshape(X, [-1, 7680])

    with tf.variable_scope("fc_1"):
        X = tf.layers.dense(inputs = X, units = 1024, activation = tf.nn.relu)
        X = tf.layers.dropout(inputs = X,
                              rate = 0.5,
                              training = mode == tf.estimator.ModeKeys.TRAIN,
                              )

    with tf.variable_scope("fc_2"):
        logits = tf.layers.dense(inputs = X,
                                 units = 5,
                                 activation = tf.nn.relu,
                                 )

    return logits
