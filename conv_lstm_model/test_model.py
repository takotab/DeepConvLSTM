import pytest
import tensorflow as tf

from .model import model

tfe = tf.contrib.eager

tf.enable_eager_execution()
_sizes = [(2, 10, 10, 1),
          (2, 10, 10, 5),
          (32, 10, 10, 2),
          (32, 25, 25, 8),
          ]


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
            logits = model(x,
                           False,
                           num_labels = num_labels,
                           )
    else:
        logits = model(x,
                       False,
                       num_labels = num_labels,
                       )

        assert logits.numpy().shape[-1] == num_labels
