import tensorflow as tf
import tensorflow_hub as hub

import config


def make_model():
    """
    https://www.tensorflow.org/hub/creating
    https://github.com/tensorflow/hub/blob/r0.1/examples/text_embeddings/export.py

    Returns:
        A module spec object used for constructing a TF-Hub module.
    """

    def model_fn():
        inputs = tf.placeholder(dtype = tf.float32, shape = [None, 1, config.SLIDING_WINDOW_LENGTH,
                                                             config.NB_SENSOR_CHANNELS])
        concat = tf.reshape(inputs,
                            (-1, config.SLIDING_WINDOW_LENGTH * config.NB_SENSOR_CHANNELS))
        layer1 = tf.layers.fully_connected(concat, 200)
        layer2 = tf.layers.fully_connected(layer1, 100)

        logits = tf.layers.fully_connected(layer2, 18)
        y = tf.nn.softmax(
                logits,
                )

        outputs = dict(default = y, hidden_activations = layer2)
        # Add default signature.
        hub.add_signature(inputs = inputs, outputs = outputs)

    return hub.create_module_spec(model_fn)
