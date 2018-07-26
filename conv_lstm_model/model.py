import tensorflow as tf

from conv_model.resnetblock import Resnet_layer


def model(deep_t, is_training, num_labels = 5):
    with tf.variable_scope("Resnet_layer_1"):
        x = Resnet_layer(deep_t, 3, [1, 3, 5], is_training)

    x = tf.tile(x, (1, 1, 1, 2))

    with tf.variable_scope("Resnet_layer_2"):
        x = Resnet_layer(x, 5, [1, 5, 10], is_training)

    x = tf.reshape(x, [-1, deep_t.shape[1], 10 * deep_t.shape[2]])
    # TODO: insert dropoutlayer
    # input must be a 3D tensor with shape (batch_size, timesteps, input_dim)
    x = tf.keras.layers.LSTM(128,
                             return_sequences = True)(x)
    logits = tf.keras.layers.LSTM(num_labels,
                                  return_sequences = False)(x)

    return logits
