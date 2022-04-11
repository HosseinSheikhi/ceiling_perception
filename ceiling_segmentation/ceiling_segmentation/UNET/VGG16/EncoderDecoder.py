import tensorflow as tf
from UNET.VGG16.Encoder import Encoder
from UNET.VGG16.Decoder import Decoder


class EncoderDecoder(tf.keras.Model):
    def __init__(self, num_classes, batch_norm=False):
        """

        :param num_classes: num of classed we would like to segment (e.g. 2 for free vs occluded)
        :param batch_norm: run with or without batch normalization
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(batch_norm)  # create an instance of Encoder
        self.decoder = Decoder(batch_norm)  # create an instance of Decoder
        self.middle_conv = tf.keras.layers.Conv2D(1024, 3, strides=1, padding="same", activation="relu")  # parameters are based on VGG16 architecture
        self.last_conv = tf.keras.layers.Conv2D(num_classes, 1, strides=1, padding="same", activation="softmax")

    def call(self, inputs, training=False):
        # pass the input image to the encoder and get the output of each vgg blk
        blk_1_out, blk_2_out, blk_3_out, blk_4_out, blk_5_out, x = self.encoder(inputs, training)
        x = self.middle_conv(x)
        x = self.decoder(x, blk_1_out, blk_2_out, blk_3_out, blk_4_out, blk_5_out, training)

        output = self.last_conv(x)
        return output
