import tensorflow as tf

from conv_model.make_model import make_model

tf.enable_eager_execution()

if __name__ == "__main__":
    logits = make_model(tf.random_normal((2, 10, 10, 1)),
                        False,
                        )
    print(logits
          )
    assert y.shape == num_labels
