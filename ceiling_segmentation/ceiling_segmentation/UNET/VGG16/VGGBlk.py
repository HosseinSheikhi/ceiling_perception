"""
Effectively, the Layer class corresponds to what we refer to in the literature as a "layer" (as in "convolution layer"
or "recurrent layer") or as a "block" (as in "ResNet block" or "Inception block").
Meanwhile, the Model class corresponds to what is referred to in the literature as a "UNET"
(as in "deep learning UNET") or as a "network" (as in "deep neural network").

So if you're wondering, "should I use the Layer class or the Model class?", ask yourself: will I need to call fit() on it?
Will I need to call save() on it? If so, go with Model. If not (either because your class is just a block in a bigger
system, or because you are writing training & saving code yourself), use Layer.

For more info for subclass layer: https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer

This Layer subclass is used to create basic VGG blocks as is defined in readme file

"""

import tensorflow as tf
from tensorflow.keras import layers

#TODO we do not need to build function cauase weights are not based on input size
class VggBlock(layers.Layer):
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
        super(VggBlock, self).__init__()
        self.layers = layers_num
        self.filters = filters
        self.kernel_size = kernel_size
        self.layer_name = name
        self.stride = stride
        self.conv_layers = None

    def build(self, input_shape):
        """
        This method can be used to create weights that depend on the shape(s) of the input(s), using add_weight().
        __call__() will automatically build the layer (if it has not been built yet) by calling build()
        :param input_shape: is fed automatically
        :return: None
        """
        self.conv_layers = [
            layers.Conv2D(self.filters, self.kernel_size, strides=self.stride, padding="same", activation='relu',
                          kernel_initializer='he_normal', name=self.layer_name + "_" + str(i))
            for i in range(self.layers)]

    def call(self, inputs, training=None):
        """
        performs the logic of applying the layer to the input tensors (which should be passed in as argument).
        passes the input tensors to the conv layers
        :param inputs: are fed in either encoder or decoder
        :param training: if its in training mode true otherwise false
        :return: output of vgg layer
        """
        x = inputs
        for conv in self.conv_layers:
            x = conv(x)

        return x
