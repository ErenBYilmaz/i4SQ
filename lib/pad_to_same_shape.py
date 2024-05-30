# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import Layer

from lib import memory_control
from tensorflow.keras.initializers import Constant

from tensorflow.keras.layers import Conv3D, Concatenate, UpSampling3D

from typing import Tuple
import tensorflow
import tensorflow.keras
import numpy
from tensorflow.keras import Input, Model


class PadToSameSize(Layer):
    """
    Pad the first layer to the same size as the second layer
    """

    def __init__(self, axes: Tuple[int, ...], mode="CONSTANT", constant_values=0, **kwargs):
        super(PadToSameSize, self).__init__(**kwargs)
        if len(set(axes)) != len(axes):
            raise ValueError
        self.constant_values = constant_values
        self.mode = mode
        self.supports_masking = True
        self.axes = axes

    def call(self, inputs, **kwargs):
        input_to_pad = inputs[0]
        pad_to_size = _shape(inputs[1])

        # pad with the difference in size, but only the specified axes
        size_difference = pad_to_size - _shape(input_to_pad)
        axes = tensorflow.keras.backend.constant([[ax] for ax in self.axes], dtype='int32')
        paddings = tensorflow.scatter_nd(indices=tensorflow.keras.backend.expand_dims(axes, -1),
                                         updates=tensorflow.keras.backend.gather(size_difference, axes),
                                         shape=_shape(_shape(inputs[1])))

        # one value for the left and one value for the right side
        paddings_per_side = tensorflow.keras.backend.stack([_ceil(paddings / 2),
                                                 _floor(paddings / 2)], axis=-1)

        return tensorflow.pad(input_to_pad,
                              paddings=tensorflow.keras.backend.cast(paddings_per_side, 'int32'),
                              mode=self.mode,
                              constant_values=self.constant_values)

    def get_config(self):
        config = {attr: getattr(self, attr) for attr in ['axes', 'mode', 'constant_values']}
        base_config = super(PadToSameSize, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return tuple(is2 if idx in self.axes else is1
                     for idx, (is1, is2) in enumerate(zip(*input_shape)))


def _shape(tensor):
    return tensorflow.keras.backend.shape(tensor)


def _floor(tensor):
    return tensorflow.keras.backend.round(tensor - 0.5 + tensorflow.keras.backend.epsilon())


def _ceil(tensor):
    return tensorflow.keras.backend.round(tensor + 0.5 - tensorflow.keras.backend.epsilon())

