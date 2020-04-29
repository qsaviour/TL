import keras


def build_model():
    inputs = keras.layers.Input((128, 128))
    l1 = keras.layers.Conv2D(32, (3, 3))

    model = keras.Model(inputs, l1)
    return model
