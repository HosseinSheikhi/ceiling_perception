import tensorflow as tf
from tensorflow.keras import layers
from DenseNet.models.encoder import Encoder
from DenseNet.models.decoder import Decoder
from DenseNet.models.dense_block import DenseBlk

FCDenseNetVariants = {
    56: {'layers_in_dense_blocks': [4, 4, 4, 4, 4], 'layers_in_bottleneck': 4, 'layers_in_TU_blocks': [4, 4, 4, 4, 4],
         'growth_rate': 12, 'name': 'FCDenseNet56'},
    67: {'layers_in_dense_blocks': [5, 5, 5, 5, 5], 'layers_in_bottleneck': 5, 'layers_in_TU_blocks': [5, 5, 5, 5, 5],
         'growth_rate': 16, 'name': 'FCDenseNet67'},
    103: {'layers_in_dense_blocks': [4, 5, 7, 10, 12], 'layers_in_bottleneck': 15,
          'layers_in_TU_blocks': [15, 12, 10, 7, 5], 'growth_rate': 16, 'name': 'FCDenseNet103'}}


class FCDenseNet(tf.keras.Model):
    def __init__(self,
                 class_num: int,
                 variant: int,
                 mode: str) -> None:
        """
        Whole UNET of different variants of FCDenseNet (including 3x3 conv - encoder - bottleneck - decoder - 1x1 conv)
        @param class_num: corresponding to number of categories to be segmented
        @param variant: corresponding to variant of FCDenseNet i.e. 56, 67, 103
        @param mode: either 'Train' or 'Test'
        """
        super(FCDenseNet, self).__init__()
        self.class_num = class_num
        self.mode = mode

        if variant not in [56, 67, 103]:
            print("FCDenseNet variant must be one of the following list [56,67,103]")
        self.layers_in_dense_blocks = FCDenseNetVariants[variant]['layers_in_dense_blocks']
        self.layers_in_bottleneck = FCDenseNetVariants[variant]['layers_in_bottleneck']
        self.layers_in_TU_blocks = FCDenseNetVariants[variant]['layers_in_TU_blocks']
        self.growth_rate = FCDenseNetVariants[variant]['growth_rate']
        self.layer_name = FCDenseNetVariants[variant]['name']

        self.first_conv = None
        self.encoder = None
        self.decoder = None
        self.bottleneck = None
        self.last_conv = None

    def build(self, input_shape):
        self.first_conv = layers.Conv2D(48, 3, strides=1, padding='same', activation='relu', use_bias=False,
                                        name='First_Conv')
        self.encoder = Encoder(self.layers_in_dense_blocks, self.growth_rate, self.mode, self.name)
        self.decoder = Decoder(self.layers_in_dense_blocks[::-1], self.layers_in_TU_blocks, self.growth_rate, self.mode,
                               self.name)
        self.bottleneck = DenseBlk(self.layers_in_bottleneck, self.growth_rate, self.mode, self.name)
        self.last_conv = layers.Conv2D(self.class_num, 1, activation='softmax', use_bias=False, name="Last_Conv")

    def call(self, inputs, training=None, mask=None):
        x = self.first_conv(inputs)
        encoder_out, skip_connections = self.encoder(x)
        bottleneck_out = self.bottleneck(encoder_out)
        decoder_out = self.decoder(bottleneck_out, **skip_connections)
        output = self.last_conv(decoder_out)

        return output
