import pytest
import tensorflow as tf

from .make_model import make_model
from .resnetblock import Resnet_layer

tfe = tf.contrib.eager

tf.enable_eager_execution()
_sizes = [(2, 10, 10, 1),
          (2, 10, 10, 5),
          (32, 10, 10, 2),
          (32, 25, 25, 8),
          ]


@pytest.mark.parametrize("sizes", _sizes)
@pytest.mark.parametrize("filter_3", [10,
                                      2,
                                      3,
                                      ])
def test_Resnet_layer(sizes, filter_3):
    x = tf.random_normal(sizes)
    if x.shape[-1] == filter_3 or x.shape[-1] == 1:
        x = Resnet_layer(x, 5, [3, 1, filter_3], False)
        assert len(x.shape) == 4
    else:
        with pytest.raises(AssertionError):
            x = Resnet_layer(x, 5, [3, 1, filter_3], False)


@pytest.mark.parametrize("sizes", _sizes)
@pytest.mark.parametrize("num_labels", [
    2,
    6,
    10,
    ])
def test_make_model(num_labels, sizes):
    x = tf.random_normal(sizes)
    if sizes[-1] is not 1 and sizes[-1] is not 5:
        with pytest.raises(AssertionError):
            logits = make_model(x,
                                False,
                                num_labels = num_labels,
                                )
    else:
        logits = make_model(x,
                            False,
                            num_labels = num_labels,
                            )

        assert logits.numpy().shape[-1] == num_labels
