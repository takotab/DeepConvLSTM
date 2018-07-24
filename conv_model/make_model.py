import tensorflow as tf

from . import utils  # pylint : ignore


def make_model(deep_t, params, mode):
    """ Makes the custom model.

    Args:
        deep_t:
        params: Not used yet
        mode: tf.estimator.ModeKeys

    Returns:

    """

    X = utils.batch_norm(deep_t, mode = mode)

    X = utils.conv_2d(X, 5, "1")
    # X = tf.layers.dropout(
    #     inputs=X, rate=params.dropout, training=mode == tf.estimator.ModeKeys.TRAIN)
    X = utils.conv_2d(X, 10, "2")

    X = utils.conv_2d(X, 20, "3")  # shape: [-1, 1, 3, 20]

    X = tf.reshape(X, [-1, 60])

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
