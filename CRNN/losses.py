from tensorflow.keras.backend import ctc_batch_cost
import tensorflow as tf

def custom_ctc():

    def loss(y_true, y_pred):
        batch_labels = y_true[:, :, 0]
        label_length = y_true[:, 0, 1]
        input_length = y_true[:, 0, 2]

        label_length = tf.expand_dims(label_length, -1)
        input_length = tf.expand_dims(input_length, -1)


        return ctc_batch_cost(batch_labels, y_pred, input_length, label_length)
    return loss