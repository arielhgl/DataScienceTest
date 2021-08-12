from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.python.keras import activations


def create_mlp(dim, regress=False):
    # create MLP architecture
    model = Sequential()
    model.add(Dense(32, input_dim=dim, activation='relu'))
    #model.add(Dropout(0.2))
    # model.add(Dense(64,activation = 'relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(32,activation = 'relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(16,activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(8,activation = 'relu'))
    # model.add(Dropout(0.2))
    model.add(Dense(4, activation='relu'))
    # check to see if the regression node should be added
    if regress:
        model.add(Dense(1, activation='linear'))
    # return our model
    return model


# def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
#     # initialize the input shape and channel dimension, assuming
#     # TensorFlow/channels-last ordering
#     inputShape = (height, width, depth)
#     chanDim = -1
#     # define the model input
#     inputs = Input(shape=inputShape)
#     # loop over the number of filters
#     for (i, f) in enumerate(filters):
#         # if this is the first CONV layer then set the input
#         # appropriately
#         if i == 0:
#             x = inputs
#         # CONV => RELU => BN => POOL
#         x = Conv2D(f, (3, 3), padding="same")(x)
#         x = Activation("relu")(x)
#         x = BatchNormalization(axis=chanDim)(x)
#         x = MaxPooling2D(pool_size=(2, 2))(x)
#     # flatten the volume, then FC => RELU => BN => DROPOUT
#     x = Flatten()(x)
#     x = Dense(16)(x)
#     x = Activation("relu")(x)
#     x = BatchNormalization(axis=chanDim)(x)
#     x = Dropout(0.5)(x)
#     # apply another FC layer, this one to match the number of nodes
#     # coming out of the MLP
#     x = Dense(4)(x)
#     x = Activation("relu")(x)
#     # check to see if the regression node should be added
#     if regress:
#         x = Dense(1, activation="linear")(x)
#     # construct the CNN
#     model = Model(inputs, x)
#     # return the CNN
#     return model


def create_cnn(width, height, depth):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    inputShape = (height, width, depth)
    # define the model input
    inputs = Input(shape=inputShape)

    # CONV => RELU => BN => POOL
    conv1  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(inputs)
    conv2  = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(conv1)
    pool1  = MaxPooling2D((2, 2))(conv2)

    conv3  = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(pool1)
    conv4  = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(conv3)
    pool2  = MaxPooling2D((2, 2))(conv4)

    conv5  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(pool2)
    conv6  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv5)
    conv7  = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(conv6)
    pool3  = MaxPooling2D((2, 2))(conv7)

    conv8  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool3)
    conv9  = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv8)
    conv10 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv9)
    pool4  = MaxPooling2D((2, 2))(conv10)

    conv11 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(pool4)
    conv12 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv11)
    conv13 = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(conv12)
    pool5  = MaxPooling2D((2, 2))(conv13)

    flat   = Flatten()(pool5)
    dense1 = Dense(4096, activation="relu")(flat)
    dense2 = Dense(4096, activation="relu")(dense1)
    output = Dense(1000, activation="softmax")(dense2)

    vgg16_model  = Model(inputs=inputs, outputs=output)
    return vgg16_model
