import tensorflow as tf

from conv_model.make_model import make_model
from conv_model.resnetblock import Resnet_layer

tf.enable_eager_execution()

if __name__ == "__main__":
    x = Resnet_layer(tf.random_normal((32, 25, 25, 2)),
                     3,
                     [3, 3, 2],
                     False)
    a = x.numpy()
    logits = make_model(tf.random_normal((2, 10, 10, 2)),
                        False,
                        )
    print(logits
          )
    assert y.shape == num_labels
