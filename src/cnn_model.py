from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D


def buildCNNModel(
    input_shape,
    num_classes,
    filters,
    kernel_size=[(3, 3)],
    pool_size=(2, 2),
    stride=1,
    dropout=0.05,
):
    model = Sequential()
    model.add(
        Conv2D(
            filters[0],
            kernel_size[0],
            strides=stride,
            padding="valid",
            input_shape=input_shape,
        )
    )
    conv_out1 = Activation("relu")
    model.add(conv_out1)

    model.add(Conv2D(filters[1], kernel_size[1]))
    conv_out2 = Activation("relu")
    model.add(conv_out2)

    model.add(Conv2D(filters[2], kernel_size[2]))
    conv_out3 = Activation("relu")
    model.add(conv_out3)

    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Dropout(dropout))

    model.add(Flatten())

    model.add(Dense(num_classes))

    model.add(Activation("softmax"))

    return model
