import tensorflow as tf

from conv_lstm_model.model import model

tf.enable_eager_execution()

if __name__ == "__main__":
    logits = model(tf.random_normal((2, 50, 113, 1)),
                   False,
                   )
    print(logits
          )
    assert y.shape == num_labels
