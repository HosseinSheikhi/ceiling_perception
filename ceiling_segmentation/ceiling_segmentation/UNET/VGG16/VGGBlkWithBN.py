import tensorflow as tf
from tensorflow.keras import layers


class VggBlockWithBN(layers.Layer):
    def __init__(self, layers_num, filters, kernel_size, name, stride=1):
        """
        Defines custom layer attributes, and creates layer state variables that
        do not depend on input shapes, using add_weight()
        :param layers_num: number of convolution layers in block
        :param filters: filters for conv layer
        :param kernel_size: kernel_size for conv layer
        :param name: name of the VGG block
        :param stride: stride for conv layers
        """
        super(VggBlockWithBN, self).__init__()
        self.kernel_size = kernel_size
        self.filters = filters
        self.stride = stride
        self.layers = layers_num
        self.layer_name = name
        self.conv_layers = None
        self.bn_layers = None

    def build(self, input_shape):
        """
        This method can be used to create weights that depend on the shape(s) of the input(s), using add_weight().
        __call__() will automatically build the layer (if it has not been built yet) by calling build()
        :param input_shape: is fed automatically
        :return:
        """
        self.conv_layers = [layers.Conv2D(self.filters, self.kernel_size, strides=self.stride, padding="SAME",
                                          kernel_initializer='he_normal', name=self.layer_name + "_" + str(i))
                            for i in range(self.layers)]
        self.bn_layers = [layers.BatchNormalization(name=self.layer_name + "_" + str(i)) for i in
                          range(self.layers)]

    def call(self, inputs, training):
        """
        performs the logic of applying the layer to the input tensors (which should be passed in as argument).
        passes the input tensors to the conv layers
        :param inputs: are fed in either encoder or decoder
        :param training: if its in training mode true otherwise false
        :return: output of vgg layer
        """
        x = inputs
        for i in range(self.layers):
            x = self.conv_layers[i](x)
            x = self.bn_layers[i](x, training=training)
            x = tf.nn.relu(x)
        return x
