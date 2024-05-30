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
from tensorflow.python.keras.utils.generic_utils import _GLOBAL_CUSTOM_OBJECTS


class CropToSameSize(Layer):
    """
    Pad the first layer to the same size as the second layer
    """

    def __init__(self, axes: Tuple[int, ...], **kwargs):
        super(CropToSameSize, self).__init__(**kwargs)
        if len(set(axes)) != len(axes):
            raise ValueError
        self.supports_masking = False
        self.axes = axes

    def call(self, inputs, **kwargs):
        input_to_crop = inputs[0]
        current_size = _shape(inputs[0])
        crop_to_size = _shape(inputs[1])
        size_difference = current_size - crop_to_size

        axes = tensorflow.keras.backend.constant([[ax] for ax in self.axes], dtype='int32')
        croppings = tensorflow.scatter_nd(indices=tensorflow.keras.backend.expand_dims(axes, -1),
                                          updates=tensorflow.keras.backend.gather(size_difference, axes),
                                          shape=_shape(_shape(inputs[1])))
        croppings_per_side = tensorflow.keras.backend.stack([_ceil(croppings / 2),
                                                  _floor(croppings / 2)], axis=-1)
        croppings_per_side = tensorflow.keras.backend.cast(croppings_per_side, 'int32')

        idx = tuple(slice(croppings_per_side[axis, 0],
                          current_size[axis] - croppings_per_side[axis, 1]) for axis in range(len(inputs[0].shape.as_list())))
        return input_to_crop[idx]

    def get_config(self):
        config = {attr: getattr(self, attr) for attr in ['axes', ]}
        base_config = super(CropToSameSize, self).get_config()
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


_GLOBAL_CUSTOM_OBJECTS.update({
    obj.__name__: obj
    # If your model uses any custom layers, add them here
    for obj in [CropToSameSize]
})
