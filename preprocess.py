import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # nopep8
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing


class PreprocessingLayers(layers.Layer):
    def __init__(self, factor=0.15, scale=1.0 / 255.0, flipmode='horizontal', seed=182):
        """Image preprocessing layer Block

        Args:
            factor (float, optional): factor to set for rotation, brightness, zoom and image shifting. Defaults to 0.2.
            scale (float, optional): float to multiple all features by and normalize tensorflow. Defaults to 1.0/255.0.
            seed (int, optional): set the seed value for all preprocessing layers. Defaults to 182
        """
        super(PreprocessingLayers, self).__init__()
        self.factor = factor
        self.scale = scale
        self.flipmode = flipmode
        self.seed = seed

        self.rescale = preprocessing.Rescaling(scale=self.scale)
        self.randomrotate = preprocessing.RandomRotation(factor=self.factor, seed=self.seed)
        self.randomzoom = preprocessing.RandomZoom(height_factor=self.factor, width_factor=self.factor, seed=self.seed)
        self.translation = preprocessing.RandomTranslation(height_factor=self.factor, width_factor=self.factor, seed=self.seed)
        self.contrast = preprocessing.RandomContrast(factor=self.factor, seed=self.seed)
        self.flip = preprocessing.RandomFlip(mode=self.flipmode, seed=self.seed)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'factor': self.factor,
            'scale': self.scale,
            'seed': self.seed,
            'flipmode': self.flipmode
        })
        return config

    @tf.function
    def call(self, image):
        """apply all preprocessing steps in

        Args:
            image (tensor): numerical data of image

        Returns:
            tensor: preprocessed data
        """
        image = self.rescale(image)
        image = self.randomrotate(image)
        image = self.randomzoom(image)
        image = self.translation(image)
        image = self.contrast(image)
        image = self.flip(image)
        return image
