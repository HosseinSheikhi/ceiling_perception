import tensorflow as tf
from tensorflow.keras import layers


class Layer(tf.keras.layers.Layer):
    def __init__(self, growth_rate: int, mode: str, name: str) -> None:
        """
        The "layer" building block as is defined in the paper.
        "layer"s will be used for building dense blocks

        Arguments:

        @param growth_rate: feature map's size
        @param mode: Train or Test. Will be used for BN layers
        @param name: name
        """
        super(Layer, self).__init__()
        self.growth_rate = growth_rate
        self.training = True if mode == 'Train' else False
        self.layer_name = name
        self.BN_layer = None
        self.conv_layer = None

    def build(self, input_shape):
        self.BN_layer = layers.BatchNormalization(trainable=self.training, epsilon=1.001e-5,
                                                  name=self.layer_name + '_bn_1_')
        self.conv_layer = layers.Conv2D(self.growth_rate, 3, padding='same', use_bias=False,
                                        name=self.layer_name + '_conv_1_')

    def call(self, inputs, **kwargs):
        x = self.BN_layer(inputs)
        x = layers.Activation('relu')(x)
        x = self.conv_layer(x)
        if self.training:
            x = layers.Dropout(rate=0.2)(x)
        x = layers.concatenate([x, inputs])  # section 3.1 DenseNet concatenation ([x1,x0])
        return x
