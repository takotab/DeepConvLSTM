import pytest
import tensorflow as tf

from .model import make_keras_model
from .resnetblock import ResnetIdentityBlock

tfe = tf.contrib.eager

tf.enable_eager_execution()


def test_restnetblock():
    block = ResnetIdentityBlock(1, [1, 2, 3])
    print(block(tf.zeros([1, 2, 3, 3])))
    print([x.name for x in block.variables])


@pytest.mark.parametrize("num_labels", [
    2,
    6,
    10,
    ])
def test_make_keras_model(num_labels):
    num_labels = 5
    Model = make_keras_model(num_labels, (None, 1, 10, 10), resnet_kernel_sizes = [3, 1, 5])
    y = Model(tf.random_normal((10, 1, 10, 10)))
    assert y.shape == num_labels
