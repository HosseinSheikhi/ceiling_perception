from tensorflow.keras import layers
from DenseNet.models.layer import Layer


class DenseBlk(layers.Layer):
    def __init__(self, layers_num: int,
                 growth_rate: int,
                 mode: str,
                 name: str) -> None:
        """
        Dense block as is defined in the paper

        Arguments:
        @param layers_num: corresponds to the number of Layers in dense block
        @param growth_rate: corresponds to the feature map's size, all the layers in a all dense blocks in
                            the architecture have same growth_rate
        @param mode: Train or Test
        @param name: something like blah_DB_1_
        """
        super(DenseBlk, self).__init__()
        self.layers_num = layers_num
        self.growth_rate = growth_rate
        self.mode = mode
        self.layer_name = name
        self.conv_layers = None

    def build(self, input_shape):
        self.conv_layers = [Layer(self.growth_rate, self.mode, self.layer_name + '_Layer_' + str(i) + '_') for i in
                            range(self.layers_num)]

    def call(self, inputs, **kwargs):
        x = inputs
        for i in range(self.layers_num):
            x = self.conv_layers[i](x)

        return x
