from tensorflow.keras import layers
from DenseNet.models.dense_block import DenseBlk
from DenseNet.models.TU_block import TUBlk


class Decoder(layers.Layer):
    def __init__(self, layers_in_dense_blocks: list,
                 tu_feature_map_size: list,
                 growth_rate: list,
                 mode: str,
                 name: str) -> object:
        """
        Decoder Block

        @param layers_in_dense_blocks: list by size = 5 (cause all the variants of FC-DenseNet* have 5 dense blocks in decoder ),
                                        containing number of layers in individual dense_blocks
        @param tu_feature_map_size: number of filters in TU blocks which will be multiplied by growth_rate
        @param growth_rate: corresponds to the feature map's size, all the layers in all dense blocks in
                                     the architecture have same growth_rate
        @param mode: Train or Test
        @param name: Name of the FCDenseNet
        """
        super(Decoder, self).__init__()
        self.layers_in_dense_blocks = layers_in_dense_blocks
        self.tu_feature_map_size = tu_feature_map_size
        self.growth_rate = growth_rate
        self.mode = mode
        self.layer_name = name + "_Decoder_"
        self.dense_blocks = None
        self.TU_blocks = None

    def build(self, input_shape):
        self.dense_blocks = [
            DenseBlk(self.layers_in_dense_blocks[i], self.growth_rate, self.mode,
                     self.layer_name + "_DB_" + str(i) + "_")
            for i in range(5)]
        self.TU_blocks = [TUBlk(self.tu_feature_map_size[i]*self.growth_rate, self.name + '_TU_' + str(i) + "_") for i in range(5)]

    def call(self, inputs, **kwargs):
        x0 = inputs

        for i in range(5):
            x_temp = self.TU_blocks[i](x0)
            x = layers.concatenate([kwargs["skip_connection_" + str(5 - i) + "_"], x_temp])
            x0 = self.dense_blocks[i](x)

        return x0
