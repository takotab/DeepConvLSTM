import tensorflow as tf

from .resnetblock import Resnet_layer


def model(deep_t, is_training, num_labels = 5):
    """ This is the actual resnet/conv model

    Args:
        deep_t:
        is_training:

    Returns:

    """
    with tf.variable_scope("Resnet_layer_1"):
        x = Resnet_layer(deep_t, (3, 1), [3, 5, 5], is_training)

    x = tf.tile(x, (1, 1, 1, 2))
    x = tf.layers.max_pooling2d(x,
                                2,
                                2,
                                )

    with tf.variable_scope("Resnet_layer_2"):
        x = Resnet_layer(x, (5, 1), [3, 5, 10], is_training)

    x = tf.layers.max_pooling2d(x,
                                2,
                                2,
                                )

    x = tf.layers.flatten(x)

    x = tf.layers.dense(x, 100)
    logits = tf.layers.dense(x, num_labels)
    return logits
