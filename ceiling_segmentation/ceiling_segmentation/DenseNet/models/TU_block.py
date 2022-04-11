from tensorflow.keras import layers


class TUBlk(layers.Layer):
    def __init__(self, growth_rate: int, name: str) -> None:
        """
         Transition Up Building Block as is defined in the paper

         @param growth_rate: corresponds to the number of feature maps
         @param name: name
         """
        super(TUBlk, self).__init__()
        self.growth_rate = growth_rate
        self.layer_name = name + "_Transposed_"
        self.transposed_conv = None

    def build(self, input_shape):
        self.transposed_conv = layers.Conv2DTranspose(self.growth_rate, 3, strides=2, padding='same', activation='relu',
                                                      use_bias=False, name=self.layer_name)

    def call(self, inputs, **kwargs):
        x = self.transposed_conv(inputs)
        return x
