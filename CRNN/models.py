from tensorflow.keras.layers import Dense, Conv2D, TimeDistributed, Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import MaxPooling2D, Flatten
from tensorflow.keras.layers import LSTM, Bidirectional


def downsample_block(input_img):
    x = Conv2D(8, 3, activation="relu", padding="same")(input_img)
    x = Conv2D(16, 3, activation="relu", padding="same")(x)
    x = MaxPooling2D()(x)

    x = Conv2D(32, 3, activation="relu", padding="same")(x)
    x = Conv2D(64, 3, activation="relu", padding="same")(x)
    x = MaxPooling2D()(x)

    tdist = TimeDistributed(Flatten(), name='timedistrib')(x)

    rnn_in = Dense(128, activation="relu", name='dense_in')(tdist)

    return rnn_in

def LSTM_encoding_block(rnn_in):
    x = Bidirectional(LSTM(64, return_sequences=True))(rnn_in)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)

    return x


def make_standard_CRNN(img_h, num_chars, use_lstm=False):
    input_img = Input(name='the_input', shape=(None, img_h, 3))

    rnn_in = downsample_block(input_img)

    if use_lstm:
        encoded = LSTM_encoding_block(rnn_in)

    else:
         encoded = rnn_in

    y_pred = Dense(num_chars+1, name="predictions", activation='softmax')(encoded)

    model = Model(inputs=input_img, outputs=y_pred)

    return model