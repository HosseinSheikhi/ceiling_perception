from tensorflow.keras import layers
from DenseNet.models.dense_block import DenseBlk
from DenseNet.models.TD_block import TDBlk


class Encoder(layers.Layer):
    def __init__(self, layers_in_dense_blocks: list,
                 growth_rate: int,
                 mode: str,
                 name: str) -> None:
        """
        Encoder Block
        @param layers_in_dense_blocks: list by size = 5 (cause all the variants of FC-DenseNet* have 5 dense block in encoder ),
                                        containing number of layers in individual dense_blocks
        @param growth_rate: corresponds to the feature map's size, all the layers in all dense blocks in
                            the architecture have same growth_rate
        @param mode: Train or Test
        @param name: Name of the FCDenseNet
        """
        super(Encoder, self).__init__()
        self.layers_in_dense_blocks = layers_in_dense_blocks
        self.growth_rate = growth_rate
        self.mode = mode
        self.layer_name = name + "_Encoder_"

        self.dense_blocks = None
        self.TD_blocks = None
        self.skip_connections = {}

    def build(self, input_shape):
        self.dense_blocks = [
            DenseBlk(self.layers_in_dense_blocks[i], self.growth_rate, self.mode,
                     self.layer_name + "_DB_" + str(i) + "_")
            for i in range(5)]
        self.TD_blocks = [TDBlk(self.mode, self.layer_name + "_TD_" + str(i) + "_") for i in
                          range(5)]

    def call(self, inputs, **kwargs):
        x0 = inputs

        for i in range(5):
            x_tmp = self.dense_blocks[i](x0)  # Dense blk
            self.skip_connections["skip_connection_" + str(i + 1) + "_"] = x_tmp
            x0 = self.TD_blocks[i](x_tmp)  # TD blk

        return x0, self.skip_connections
