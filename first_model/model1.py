import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # nopep8
os.environ["TF_ENABLE_AUTO_MIXED_PRECISION"] = '1'

FILE_PATH = os.path.realpath(__file__)
import sys
sys.path.append('\\'.join(FILE_PATH.split('\\')[:3]))
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys
sys.path.append("../")
from trainvaltest import trainvaltest
import preprocess
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

MODEL_NAME = "the_first_war"
MODEL_PATH = os.path.join("\\".join(FILE_PATH.split("\\")[:-1]), f"{MODEL_NAME}.h5")
BATCH_SIZE = 16
LABELS, INPUT_SHAPE, Train_Data, Val_Data, Test_Data = trainvaltest(BATCH_SIZE=BATCH_SIZE)
EPOCHS = 20
VERBOSE = 1


def conv_layer(filters, kernel_size, padding, strides):
    """Creates a convolution layer for CNNBlock

    Args:
        filters (int): number of filters in conv layer
        kernel_size (int): kernel size for conv layer
        padding (str): either 'same' padding or 'valid' padding
        strides (int): strides for conv layer

    Returns:
        conv layer: returns configured conv layer for CNNBlock
    """
    layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, padding=padding, strides=strides,
                          activation=layers.ReLU(), use_bias=False, kernel_initializer='he_normal')
    return layer


class CNNBlock(layers.Layer):
    def __init__(self, filters, triple=False, conv_kernel_size=(3, 3), conv_strides=(1, 1), pool_size=(2, 2), pool_strides=(2, 2), padding='same'):
        """block of either double (or triple) conv layers

        Args:
            filters (int): numbers of filters for the conv layers within this block
            triple (bool, optional): whether this conv block contains double (2) or triple (3) conv layers. Defaults to False.
            conv_strides (tuple, optional): tuple to set strides value for conv layers. Defaults to (1, 1).
            conv_kernel_size (tuple, optional): kernel size for the conv layers in this block. Defaults to (3, 3).
            pool_size (tuple, optional): pool size for pooling layer for this block. Defaults to (2, 2).
            pool_strides (tuple, optional): strides value for pooling for this block. Defaults to (2, 2).
            padding (str, optional): padding value of conv layers. Defaults to 'same'.
        """
        super(CNNBlock, self).__init__()
        self.triple = triple
        self.pool_size = pool_size
        self.pool_strides = pool_strides
        self.filters = filters
        self.conv_kernel_size = conv_kernel_size
        self.conv_strides = conv_strides
        self.padding = padding

        self.conv1 = conv_layer(filters=self.filters, kernel_size=self.conv_kernel_size, padding=self.padding, strides=self.conv_strides)
        self.conv2 = conv_layer(filters=self.filters, kernel_size=self.conv_kernel_size, padding=self.padding, strides=self.conv_strides)
        if self.triple == True:
            self.conv3 = conv_layer(filters=self.filters, kernel_size=self.conv_kernel_size, padding=self.padding, strides=self.conv_strides)
        self.bn = layers.BatchNormalization()
        self.avgpooling = layers.AveragePooling2D(pool_size=self.pool_size, strides=self.pool_strides)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'triple': self.triple,
            'pool_size': self.pool_size,
            'pool_strides': self.pool_strides,
            'filters': self.filters,
            'conv_kernel_size': self.conv_kernel_size,
            'conv_strides': self.conv_strides,
            'padding': self.padding
        })
        return config

    def __call__(self, input_tensor, training=False):
        """forward propagation

        Args:
            input_tensor (input_tensor): input tensor for this data point
            training (bool): whether to set batch normalization to training or not

        Returns:
            tensor: output of the current CNN block
        """
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        if self.triple == True:
            x = self.conv3(x)
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
        self.preprocess = preprocess.PreprocessingLayers()
        self.cnnblock1 = CNNBlock(filters=64)
        self.cnnblock2 = CNNBlock(filters=128)
        self.cnnblock3 = CNNBlock(filters=256, triple=True)
        self.cnnblock4 = CNNBlock(filters=512, triple=True)
        self.cnnblock5 = CNNBlock(filters=512, triple=True)
        self.globalmaxpooling = layers.GlobalMaxPooling2D()
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(2048, activation=layers.ReLU())
        self.dropout = layers.Dropout(0.1)
        self.fc2 = layers.Dense(1024, activation=layers.ReLU())
        self.dropout2 = layers.Dropout(0.1)
        self.outputs = layers.Dense(self.n_labels)

    @tf.function
    def call(self, input_tensor):
        """forward propagation for the entire model between each layer

        Args:
            input_tensor (tensor): output of the previous layer

        Returns:
            tensor: output of the previous tensor
        """
        x = self.preprocess(input_tensor)
        x = self.cnnblock1(x)
        x = self.cnnblock2(x)
        x = self.cnnblock3(x)
        x = self.cnnblock4(x)
        x = self.cnnblock5(x)
        x = self.globalmaxpooling(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = self.outputs(x)
        return x


def create_model(inp_shape, n_labels, model_name):
    """creates model (input and output), name layers and compile

    Args:
        inp_shape (tuple(int)): tuple of ints, input shape
        n_labels (int): number of labels for last layer
        model_name (str): name of model
        layer_names (list(str)): list of names for each layer in model

    Returns:
        model: named and configured model with input/output and named layers
    """
    model = keras.Sequential(Model(n_labels=n_labels).layers)
    model._name = model_name
    model.build(input_shape=(None, *inp_shape))

    model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                  metrics=[keras.metrics.SparseCategoricalAccuracy(name='acc')])

    model.summary()
    return model


if __name__ == '__main__':
    model = create_model(inp_shape=INPUT_SHAPE, n_labels=LABELS, model_name=MODEL_NAME)

    model_png_path = os.path.join(*FILE_PATH.split("\\")[2:-1], f"{MODEL_NAME}.png")
    keras.utils.plot_model(model, to_file=model_png_path, show_shapes=True)
    
    earlystopping = keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=VERBOSE)
    callbacks = [earlystopping]

    parser = argparse.ArgumentParser()
    parser.add_argument('--save', help='save the model', action='store_true')
    args = parser.parse_args()
    if args.save:
        model_png_path = os.join(*MODEL_PATH.split("\\")[:-1], f"{MODEL_PATH}".h5)
        best_checkpoint = keras.callbacks.ModelCheckpoint(filepath=model_png_path,
                                                          monitor='val_acc',
                                                          save_best_only=True,
                                                          save_freq='epoch',
                                                          verbose=VERBOSE)
        callbacks.append(best_checkpoint)
        print("\nModel IS being saved after every epoch!\n")
    else:
        print("\nModel is NOT being saved!\n")

    keras.utils.plot_model(model, to_file=f"{MODEL_NAME}.png", show_shapes=True)

    train = model.fit(Train_Data,
                      epochs=EPOCHS,
                      verbose=VERBOSE,
                      steps_per_epoch=len(Train_Data) // BATCH_SIZE,
                      callbacks=callbacks,
                      validation_data=Val_Data,
                      validation_steps=len(Val_Data) // BATCH_SIZE,
                      use_multiprocessing=True,
                      workers=-1)

    test = model.evaluate(Test_Data, steps=len(Test_Data) // BATCH_SIZE, workers=-1, use_multiprocessing=True, verbose=VERBOSE)
