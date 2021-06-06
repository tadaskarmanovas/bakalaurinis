import tensorflow as tf
import numpy as np
from IPython.display import display
from PIL import Image
import os

def decode_ctc(args):
    y_pred, input_length = args

    ctc_decoded = tf.keras.backend.ctc_decode(
        y_pred, input_length, greedy=True)

    return ctc_decoded


class PredVisualize(tf.keras.callbacks.Callback):

    def __init__(self, model, val_datagen, lbl_to_char_dict, printing=False):
        self.model = model
        self.datagen = iter(val_datagen)
        self.printing = printing
        self.lbl_to_char_dict = lbl_to_char_dict

    def on_epoch_end(self, batch, logs=None):
        batch_imgs, batch_labels, input_length, label_lens = next(self.datagen)

        y_preds = self.model.predict(batch_imgs)

        pred_tensor, _ = decode_ctc([y_preds, np.squeeze(input_length)])
        pred_labels = tf.keras.backend.get_value(pred_tensor[0])

        predictions = ["".join([self.lbl_to_char_dict[i] for i in word if i!=-1]) for word in pred_labels.tolist()]
        truths = ["".join([self.lbl_to_char_dict[i] for i in word]) for word in batch_labels.tolist()]

        if self.printing:
            imgs_list_arr_T = [img.transpose((1,0,2)) for img in batch_imgs]
            imgs_comb = np.hstack(imgs_list_arr_T) * 255
            imgs_comb = Image.fromarray(imgs_comb.astype(np.uint8),'RGB')
            display(imgs_comb)

        print('predictions {}'.format(predictions))

def make_save_model_cb(folder = "saved_models"):
    filename = "weights.h5"
    filepath = os.path.join(os.getcwd(), folder, filename)

    callback = tf.keras.callbacks.ModelCheckpoint(filepath,
                    save_weights_only=True, verbose=0, save_best_only=False)

    return callback

