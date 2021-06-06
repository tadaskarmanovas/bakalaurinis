import numpy as np
from PIL import Image
import itertools
from tensorflow.keras.utils import Sequence
import numpy as np

class OCR_generator(Sequence):

    def __init__(self, base_generator, batch_size, char_to_lbl_dict,
                img_h , keras_augmentor, epoch_size=500, validation=False):

        self.base_generator = base_generator
        self.batch_size = batch_size
        self.char_to_lbl_dict = char_to_lbl_dict
        self.img_h = img_h
        self.epoch_size = epoch_size
        self.validation = validation
        self.keras_augmentor = keras_augmentor

        self.num_chars = len(char_to_lbl_dict)

    def __len__(self):
        return self.epoch_size


    def __getitem__(self, index):
        label_lens = np.zeros((self.batch_size),dtype=np.float32)

        generated_content = list(list(tup) for tup in itertools.islice(self.base_generator,self.batch_size))

        generated_content, img_w, max_word_len_batch = \
                self.preprocess_batch_imgs(generated_content)

        batch_labels = np.zeros((self.batch_size, max_word_len_batch),dtype=np.float32)
        batch_imgs = np.zeros((self.batch_size, img_w, self.img_h, 3),dtype=np.float32)

        t_dist_dim = int(img_w / 4)

        input_length = np.full((self.batch_size),t_dist_dim,dtype=np.float32)

        for batch_ind in range(self.batch_size):

            img_arr, word = generated_content[batch_ind]
            batch_imgs[batch_ind,:,:] = img_arr

            labels_arr = np.array([self.char_to_lbl_dict[char] for char in word])
            batch_labels[batch_ind,0:len(labels_arr)] = labels_arr
            label_lens[batch_ind] = len(word)

            y_true = np.zeros((self.batch_size, t_dist_dim, self.num_chars),dtype=np.float32)

            y_true[:, 0:max_word_len_batch, 0] = batch_labels
            y_true[:, 0, 1] = label_lens
            y_true[:, 0, 2] = input_length


        if self.validation:

            return batch_imgs, batch_labels, input_length, label_lens

        else: #return x, y for the model
            return batch_imgs, y_true

    def preprocess_batch_imgs(self,generated_content):

        pil_images = [img for img, word in generated_content]
        max_width = max([img.size[0] for img in pil_images])
        max_word_len_batch = max([len(word) for img, word in generated_content])

        if max_width % 4 == 0:
            img_w = max_width
        else:
            img_w = max_width + 4 - (max_width % 4)

        for batch_ind in range(self.batch_size):

            pil_img = pil_images[batch_ind]
            width, height = pil_img.size

            new_img = Image.new(pil_img.mode, (img_w, self.img_h), (255,255,255))
            new_img.paste(pil_img, ((img_w - width) // 2, 0))

            img_arr = np.array(new_img)
            
            img_arr = self.keras_augmentor.random_transform(img_arr)

            generated_content[batch_ind][0] = img_arr.transpose((1,0,2)) / 255

        return generated_content, img_w, max_word_len_batch