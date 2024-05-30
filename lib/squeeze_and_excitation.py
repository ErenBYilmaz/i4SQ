"""
Source (modified): https://github.com/titu1994/keras-squeeze-excite-network/blob/master/keras_squeeze_excite_network/se.py
References
-   [Squeeze and Excitation Networks](https://arxiv.org/abs/1709.01507)
-   [Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks](https://arxiv.org/abs/1803.02579)
"""
from typing import Optional

import tensorflow.keras.backend
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Permute, Conv2D, add, multiply
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Layer
from tensorflow.python.keras.layers import Conv3D, GlobalAveragePooling3D

from lib.util import EBC


def _tensor_shape(tensor):
    return getattr(tensor, '_shape_val')


class SqueezeExciteBlock(Layer):
    def __init__(self, internal_num_filters: Optional[int] = None, **kwargs):
        super(SqueezeExciteBlock, self).__init__(**kwargs)
        self.internal_num_filters = internal_num_filters

    def get_config(self):
        config = {attr: getattr(self, attr) for attr in ['internal_num_filters']}
        base_config = super(SqueezeExciteBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        ndim = len(input_shape) - 2
        channel_axis = 1 if tensorflow.keras.backend.image_data_format() == "channels_first" else -1
        filters = input_shape[channel_axis]
        se_shape = tuple(1 for _ in range(ndim)) + (filters,)
        internal_num_filters = self.internal_num_filters
        if internal_num_filters is None:
            internal_num_filters = filters
        self.dense_output = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)
        self.dense_internal = Dense(internal_num_filters, activation='relu', kernel_initializer='he_normal', use_bias=False)
        self.reshape = Reshape(se_shape)
        self.squeeze_layer = self.gap_cls(ndim)()

    def call(self, inputs, *args, **kwargs):
        x = self.squeeze_layer(inputs)
        x = self.reshape(x)
        x = self.dense_internal(x)
        x = self.dense_output(x)
        x = multiply([inputs, x])
        return x

    def gap_cls(self, ndim):
        if ndim == 2:
            conv_cls = GlobalAveragePooling2D
        elif ndim == 3:
            conv_cls = GlobalAveragePooling3D
        else:
            raise NotImplementedError('TODO')
        return conv_cls


class SpatialSqueezeExciteBlock(Layer):
    def __init__(self, **kwargs):
        super(SpatialSqueezeExciteBlock, self).__init__(**kwargs)

    def call(self, inputs, *args, **kwargs):
        se = self.conv_layer(inputs)

        x = multiply([inputs, se])
        return x

    def build(self, input_shape):
        ndim = len(input_shape) - 2
        self.conv_layer = self.conv_cls(ndim)(filters=1, kernel_size=1, activation='sigmoid', use_bias=False, kernel_initializer='he_normal')

    def conv_cls(self, ndim):
        if ndim == 2:
            conv_cls = Conv2D
        elif ndim == 3:
            conv_cls = Conv3D
        else:
            raise NotImplementedError('TODO')
        return conv_cls


class ChannelSpatialSqueezeExciteBlock(Layer):
    def __init__(self, internal_num_filters: Optional[int] = None, **kwargs):
        super(ChannelSpatialSqueezeExciteBlock, self).__init__(**kwargs)
        self.internal_num_filters = internal_num_filters
        self.se_block = SqueezeExciteBlock(self.internal_num_filters)
        self.spatial_block = SpatialSqueezeExciteBlock()

    def call(self, inputs, *args, **kwargs):
        cse = self.se_block(inputs)
        sse = self.spatial_block(inputs)
        x = add([cse, sse])
        return x

    def get_config(self):
        config = {attr: getattr(self, attr) for attr in ['internal_num_filters']}
        base_config = super(ChannelSpatialSqueezeExciteBlock, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


from tensorflow.python.keras.utils.generic_utils import _GLOBAL_CUSTOM_OBJECTS

_GLOBAL_CUSTOM_OBJECTS.update({
    obj.__name__: obj
    for obj in [
        SqueezeExciteBlock,
        SpatialSqueezeExciteBlock,
        ChannelSpatialSqueezeExciteBlock,
    ]
})
