import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import metric
from .model import model


def model_fn(features, labels, mode, params):
    """Builds the graph, the train operation, and the metric operations

    Args:
      features: A dict of feature_column w/:
            * "windows": (batch_size, 24* 113), np.float32
      labels: (batch_size, 1), np.int32
      mode: tf.estimator.ModeKeys.[TRAIN, EVAL, PREDICT]
      params: a Dictionary-like of configuration parameters

    Returns:
      tf.estimator.EstimatorSpec
    """

    deep_t = tf.feature_column.input_layer(
            features,
            tf.feature_column.numeric_column('windows'))
    deep_t = tf.reshape(
            deep_t, shape = (-1, 24, 113, 1))

    is_training = tf.estimator.ModeKeys.TRAIN == mode
    logits = model(deep_t, is_training)

    # outputs = dict(default = y, hidden_activations = layer2)
    # # Add default signature.
    # hub.add_signature(inputs = inputs, outputs = outputs)
    #
    # return hub.create_module_spec(model_fn)

    predicted_classes = tf.argmax(logits, -1)
    predictions = {
        'class_ids'    : predicted_classes[:, tf.newaxis],
        'probabilities': tf.nn.softmax(logits),
        'logits'       : logits,
        }
    if mode == tf.estimator.ModeKeys.PREDICT:

        return tf.estimator.EstimatorSpec(mode, predictions = predictions)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels = labels, logits = logits)

    # Compute evaluation metrics.

    metric_ops = metric.extra_metrics(labels, predicted_classes)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
                mode, loss = loss, eval_metric_ops = metric_ops, predictions = predictions)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    train_op = tf.train.AdamOptimizer(learning_rate = params.learning_rate).minimize(
            loss, global_step = tf.train.get_global_step())

    print("Trainable variables", np.sum(
            [np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

    return tf.estimator.EstimatorSpec(mode, loss = loss, train_op = train_op)


def _train(spec):
    with tf.Graph().as_default():
        m = hub.Module(spec)

        for name in m.variable_map:
            print(name)

            # p_embeddings = tf.placeholder(tf.float32)

        with tf.Session() as sess:
            sess.run([load_embeddings], feed_dict = {p_embeddings: embeddings})
        m.export(export_path, sess)
        spec = model.make_model()
        _train(spec)
# TODO: use estimator https://www.tensorflow.org/hub/api_docs/python/hub/LatestModuleExporter
