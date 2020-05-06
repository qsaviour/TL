from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, \
    BatchNormalization, Activation, Add, Softmax, MaxPooling2D
from keras.regularizers import l2

regularizer_value = 1e-3


def block1(layer, filters1, filter2, filters3):
    conv1 = Conv2D(filters1, (1, 1), use_bias=False, kernel_regularizer=l2(regularizer_value))(layer)
    batch_normalization = BatchNormalization(axis=3)(conv1)
    activation = Activation('relu')(batch_normalization)
    # zero_padding = ZeroPadding2D(((1, 1), (1, 1)))(activation)
    conv2 = Conv2D(filter2, (3, 3), use_bias=False, padding='same', kernel_regularizer=l2(regularizer_value))(
        activation)
    batch_normalization = BatchNormalization(axis=3)(conv2)
    activation = Activation('relu')(batch_normalization)
    conv3 = Conv2D(filters3, (1, 1), kernel_regularizer=l2(regularizer_value))(activation)
    return conv3


def block2(layer, conv2_filters, conv2_stride, conv3_filters):
    batch_normalization = BatchNormalization(axis=3)(layer)
    activation = Activation('relu')(batch_normalization)
    conv1 = Conv2D(64, (1, 1), use_bias=False, kernel_regularizer=l2(regularizer_value))(activation)
    batch_normalization = BatchNormalization(axis=3)(conv1)
    activation = Activation('relu')(batch_normalization)
    # zero_padding = ZeroPadding2D(((1, 1), (1, 1)))(activation)
    conv2 = Conv2D(conv2_filters, (3, 3), strides=(1, 1), padding='same', use_bias=False,
                   kernel_regularizer=l2(regularizer_value))(activation)
    if conv2_stride == (2, 2):
        conv2 = MaxPooling2D((2, 2), (2, 2), padding='same')(conv2)
    batch_normalization = BatchNormalization(axis=3)(conv2)
    activation = Activation('relu')(batch_normalization)
    conv3 = Conv2D(conv3_filters, (1, 1), kernel_regularizer=l2(regularizer_value))(activation)
    return conv3


# input_shape = (128, 128, 3)
# if __name__ == '__main__':
def build_model(input_shape=(256, 256, 3)):
    inputs = Input(input_shape)
    # zero_padding = ZeroPadding2D(((3, 3), (3, 3)))(input)
    # conv = Conv2D(64, (7, 7), strides=(2, 2), padding='same', kernel_regularizer=l2(regularizer_value))(inputs)
    conv = Conv2D(64, (7, 7), strides=(1, 1), padding='same', kernel_regularizer=l2(regularizer_value))(inputs)
    # zero_padding = ZeroPadding2D(((1, 1), (1, 1)))(conv)
    max_pooling = MaxPooling2D((3, 3), (2, 2), padding='same')(conv)
    batch_normalization = BatchNormalization(axis=3)(max_pooling)
    activation = Activation('relu')(batch_normalization)

    block1_r = block1(activation, 64, 64, 256)
    block1_l = Conv2D(256, (1, 1))(activation)
    add1 = Add()([block1_l, block1_r])

    block2_r = block2(add1, 64, (1, 1), 256)
    add2 = Add()([block2_r, add1])

    block3_r = block2(add2, 64, (2, 2), 256)
    # block3_r = block2(add2, 64, (1, 1), 256)
    block3_l = MaxPooling2D((2, 2), (2, 2), padding='same')(add2)
    add3 = Add()([block3_l, block3_r])

    batch_normalization = BatchNormalization(axis=3)(add3)

    block4_r = block1(batch_normalization, 128, 128, 512)
    block4_l = Conv2D(512, (1, 1))(batch_normalization)
    add4 = Add()([block4_l, block4_r])

    block5_r = block2(add4, 128, (1, 1), 512)
    add5 = Add()([add4, block5_r])

    block6_r = block2(add5, 128, (1, 1), 512)
    add6 = Add()([add5, block6_r])

    batch_normalization = BatchNormalization(axis=3)(add6)
    activation = Activation('relu')(batch_normalization)
    global_average_polling = GlobalAveragePooling2D()(activation)
    dense = Dense(2)(global_average_polling)
    softmax = Softmax()(dense)

    return Model(inputs, softmax)


model = build_model()
model.save('test.h5')
