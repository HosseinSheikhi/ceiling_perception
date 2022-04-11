from tensorflow.keras import layers


class TDBlk(layers.Layer):
    def __init__(self, mode: str, name: str) -> None:
        """
        Transition Down Building Block as is defined in the paper

        @param mode: Train or Test. Will be used for BN layers
        @param name: name
        """
        super(TDBlk, self).__init__()

        self.training = True if mode == 'Train' else False
        self.layer_name = name
        self.BN_layer = None
        self.conv_layer = None

    def build(self, input_shape):
        self.BN_layer = layers.BatchNormalization(trainable=self.training, epsilon=1.001e-5, name=self.name + "_BN_1_")
        self.conv_layer = layers.Conv2D(input_shape[-1], 1, use_bias=False, name=self.name + '_conv_1_')

    def call(self, inputs, **kwargs):
        x = self.BN_layer(inputs)
        x = layers.Activation('relu')(x)
        x = self.conv_layer(x)
        if self.training:
            x = layers.Dropout(0.2)(x)
        x = layers.MaxPooling2D((2, 2), strides=2)(x)
        return x
