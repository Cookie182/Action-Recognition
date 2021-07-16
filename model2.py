import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # nopep8
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = '1'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from trainvaltest import trainvaltest

tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

MODEL_NAME = "the_second_war"
MODEL_PATH = os.path.join("D:\Python\Action Recognition", MODEL_NAME)
BATCH_SIZE = 8
(LABELS, INPUT_SHAPE), (train, val, test) = trainvaltest(BATCH_SIZE=BATCH_SIZE, show_image=False)
EPOCHS = 10
PATIENCE = 3
VERBOSE = 1


class CNNBlock(layers.Layer):
    def __init__(self, filters, quad=False, conv_kernel_size=(3, 3), conv_strides=(1, 1), pool_size=(2, 2), pool_strides=(2, 2), padding='same'):
        """block of either double (or triple) conv layers

        Args:
            filters (int): numbers of filters for the conv layers within this block
            quad (bool, optional): whether this conv block contains double (2) or quad (4) conv layers. Defaults to False.
            conv_strides (tuple, optional): tuple to set strides value for conv layers. Defaults to (1, 1).
            conv_kernel_size (tuple, optional): kernel size for the conv layers in this block. Defaults to (3, 3).
            pool_size (tuple, optional): pool size for pooling layer for this block. Defaults to (2, 2).
            pool_strides (tuple, optional): strides value for pooling for this block. Defaults to (2, 2).
            padding (str, optional): padding value of conv layers. Defaults to 'same'.
        """
        super(CNNBlock, self).__init__()
        self.quad = quad
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.filters = filters
        self.conv_kernel_size = conv_kernel_size
        self.conv_strides = conv_strides
        self.padding = padding
        self.regularizer = keras.regularizers.L2(l2=1e-6)

        # self.conv = get_conv(filters=self.filters, conv_kernel_size=self.conv_kernel_size, conv_strides=self.conv_strides, padding=self.padding)
        # self.conv2 = get_conv(filters=self.filters, conv_kernel_size=self.conv_kernel_size, conv_strides=self.conv_strides, padding=self.padding)
        self.conv1 = self.__conv_layer__()
        self.conv2 = self.__conv_layer__()
        if self.quad == True:
            self.conv3 = self.__conv_layer__()
            self.conv4 = self.__conv_layer__()
        self.bn = layers.BatchNormalization()
        self.avgpooling = layers.MaxPooling2D(pool_size=self.pool_size, strides=self.pool_strides)

    def __conv_layer__(self):
        """return a conv layer for the block.

        Returns:
            keras layer: 1 conv layer for a CNN block.
        """
        layer = layers.Conv2D(filters=self.filters, kernel_size=self.conv_kernel_size, strides=self.conv_strides,
                              padding=self.padding, kernel_regularizer=self.regularizer)
        return layer

    def call(self, input_tensor, training=False):
        """forward propagation

        Args:
            input_tensor (input_tensor): input tensor for this data point
            training (bool): whether to set batch normalization to training or not

        Returns:
            tensor: output of the current CNN block
        """
        x = self.conv1(input_tensor)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = tf.nn.relu(x)
        if self.quad == True:
            x = self.conv3(x)
            x = tf.nn.relu(x)
            x = self.conv4(x)
            x = tf.nn.relu(x)
        x = self.bn(x, training=training)
        x = self.avgpooling(x)
        return x


class Model(keras.Model):
    def __init__(self, n_labels):
        """model build via subclassing

        Args:
            n_labels (int): amount of labels for the model to predict
        """
        super(Model, self).__init__()
        self.n_labels = n_labels

        self.cnnblock1 = CNNBlock(filters=64)
        self.cnnblock2 = CNNBlock(filters=64)
        self.cnnblock3 = CNNBlock(filters=128)
        self.cnnblock4 = CNNBlock(filters=256, quad=True)
        self.cnnblock5 = CNNBlock(filters=512, quad=True)
        self.cnnblock6 = CNNBlock(filters=512, quad=True)
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(4096, activation='relu', name='fc1')
        self.fc2 = layers.Dense(4096, activation='relu', name='fc2')
        self.outputs = layers.Dense(self.n_labels)

    def call(self, input_tensor):
        """forward propagation for the entire model between each layer

        Args:
            input_tensor (tensor): output of the previous layer

        Returns:
            tensor: output of the previous tensor
        """
        x = self.cnnblock1(input_tensor)
        x = self.cnnblock2(x)
        x = self.cnnblock3(x)
        x = self.cnnblock4(x)
        x = self.cnnblock5(x)
        x = self.cnnblock6(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.outputs(x)
        return x


def create_model(inp_shape, n_labels, model_name, layer_names):
    """creates model (input and output), name layers and compile

    Args:
        inp_shape (tuple(int)): tuple of ints, input shape 
        n_labels (int): number of labels for last layer
        model_name (str): name of model
        layer_names (list(str)): list of names for each layer in model

    Returns:
        model: named and configured model with input/output and named layers
    """

    model = Model(n_labels=n_labels)
    model.build(input_shape=(None, *inp_shape))
    model._name = model_name

    for i, layer in enumerate(model.layers):
        layer._name = layer_names[i]

    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=keras.optimizers.Adam(learning_rate=1e-4, amsgrad=True),
                  metrics=[keras.metrics.SparseCategoricalAccuracy(name='acc')])

    model.summary()

    return model


if __name__ == '__main__':
    layer_names = tuple([
        "CNNBlock64_Double_1",
        "CNNBlock64_Double_2",
        "CNNBlock128_Double",
        "CNNBlock256_Quad",
        "CNNBlock512_Quad_1",
        "CNNBlock512_Quad_2",
        "Flatten",
        "FC1",
        "FC2",
        "Outputs"
    ])

    model = create_model(inp_shape=INPUT_SHAPE, n_labels=LABELS, model_name=MODEL_NAME, layer_names=layer_names)

    reducelronplateau = keras.callbacks.ReduceLROnPlateau(min_lr=1e-5, cooldown=1, patience=PATIENCE, verbose=VERBOSE, factor=0.1, monitor='val_loss')
    earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=VERBOSE)

    model.fit(train,
              epochs=EPOCHS,
              verbose=1,
              steps_per_epoch=len(train) // BATCH_SIZE,
              callbacks=[reducelronplateau, earlystopping],
              validation_data=val,
              validation_steps=len(val) // BATCH_SIZE,
              use_multiprocessing=True,
              workers=-1)

    model.evaluate(test, steps=len(test) // BATCH_SIZE, workers=-1, use_multiprocessing=True)
    model.save(MODEL_PATH)
