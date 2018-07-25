import tensorflow as tf


def Resnet_layer(input_tensor, kernel_size, filters, is_training = False):
    """ A resnet_layer which has the advantage to easily learn the identity function

    Make sure that input_tensor.shape[-1] == filter_3 or input_tensor.shape[-1] == 1

    Args:
        input_tensor: a tensor of 4 dimensions
        kernel_size: the kernel size for layer 2
        filters: a list of length 3 for the number of filters to use in the 3 layers
        is_training: boolean

    Returns:
        a tensor with the same shape as input_tensor

    """

    filter_1, filter_2, filter_3 = filters

    assert input_tensor.shape[-1] == filter_3 or input_tensor.shape[-1] == 1

    x = tf.keras.layers.Conv2D(filter_1, (1, 1))(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x, training = is_training)
    x = tf.keras.layers.Conv2D(filter_2, kernel_size, padding = 'same')(x)
    x = tf.keras.layers.BatchNormalization()(x, training = is_training)
    x = tf.keras.layers.Conv2D(filter_3, (1, 1))(x)
    x = tf.keras.layers.BatchNormalization()(x, training = is_training)

    x = x + input_tensor

    return tf.nn.relu(x)
