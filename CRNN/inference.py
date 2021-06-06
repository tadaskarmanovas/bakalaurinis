import ocr_source.models as models
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


def load_images(folder="inference_test_imgs"):
    inference_dir = os.path.join(os.getcwd(),folder)

    pil_images = []
    for filename in os.listdir(inference_dir):

        filepath = os.path.join(inference_dir,filename)
        img = Image.open(filepath)
        pil_images.append(img)

    return pil_images

def load_trained_model(img_h, num_chars, folder="saved_models"):
    model = models.make_standard_CRNN(img_h, num_chars)

    filename = "weights.h5"
    filepath = os.path.join(os.getcwd(), folder, filename)
    model.load_weights(filepath)

    print("model loaded")
    return model

def preprocess_batch_imgs(pil_images, img_h):

        imgs_list_arr = []

        for ind, img in enumerate(pil_images):

            w, h = img.size
            resize_scale = h / img_h
            new_w = w / resize_scale

            pil_images[ind] = img.resize((int(new_w), int(img_h)), Image.ANTIALIAS)

        max_width = max([img.size[0] for img in pil_images])

        if max_width % 4 == 0:
            img_w = max_width
        else:
            img_w = max_width + 4 - (max_width % 4)

        for batch_ind, pil_img in enumerate(pil_images):

            width, height = pil_img.size

            new_img = Image.new(pil_img.mode, (img_w, img_h), (255,255,255))
            new_img.paste(pil_img, ((img_w - width) // 2, 0))

            img_arr = np.array(new_img).transpose((1,0,2)) / 255
    
            imgs_list_arr.append(img_arr)

        t_dist_len = int(img_w/4)
        input_length = np.full((len(pil_images)), t_dist_len, dtype=int)

        return np.array(imgs_list_arr), input_length

def inference_from_folder(printing=True):
    img_h = 32
    lbl_to_char_dict = {0: '0',
                        1: '1',
                        2: '2',
                        3: '3',
                        4: '4',
                        5: '5',
                        6: '6',
                        7: '7',
                        8: '8',
                        9: '9'}
            
    num_chars = len(lbl_to_char_dict)

    model = load_trained_model(img_h, num_chars)
    pil_images = load_images()
    imgs_array, input_length = preprocess_batch_imgs(pil_images, img_h)

    y_preds = model.predict(imgs_array)

    pred_tensor, _ = decode_ctc([y_preds, input_length])
    pred_labels = tf.keras.backend.get_value(pred_tensor[0])

    predictions = ["".join([lbl_to_char_dict[i] for i in word if i!=-1]) for word in pred_labels.tolist()]

    if printing:
        imgs_list_arr = [np.asarray(img) for img in pil_images]
        imgs_comb = np.hstack(imgs_list_arr)
        imgs_comb = Image.fromarray(imgs_comb)
        display(imgs_comb)

        print('predictions {}'.format(predictions))

if __name__ == '__main__':
    inference_from_folder()