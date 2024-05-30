from math import radians
from typing import Optional

import tensorflow
import tensorflow.keras
from tensorflow import Tensor
from tensorflow.keras.layers import Layer
from tensorflow_addons.image import rotate

from lib.profiling_tools import profile_wall_time_instead_if_profiling

AVAILABLE_AXES = [0, 1, 2]

profile_wall_time_instead_if_profiling()


class CropLayer(Layer):
    def __init__(self, to_size, allow_smaller_outputs=False, **kwargs):
        super(CropLayer, self).__init__(**kwargs)
        if allow_smaller_outputs:
            raise NotImplementedError
        self.allow_smaller_outputs = allow_smaller_outputs
        self.to_size = to_size

    def call(self, inputs, *args, **kwargs):
        half_crop = tuple(s // 2 for s in self.to_size)
        center = tensorflow.keras.backend.shape(inputs) // 2
        bbox_begin = tensorflow.keras.backend.concatenate([[0], (center[1:-1] - half_crop), [0]], axis=0)
        bbox_size = [inputs.shape.as_list()[0] if inputs.shape.as_list()[0] is not None else tensorflow.keras.backend.shape(inputs)[0],
                     *self.to_size,
                     inputs.shape.as_list()[-1] if inputs.shape.as_list()[-1] is not None else tensorflow.keras.backend.shape(inputs)[-1]]

        return tensorflow.slice(inputs, bbox_begin, bbox_size)

    def compute_output_shape(self, input_shape):
        if not self.allow_smaller_outputs:
            for s1, s2 in zip(input_shape, self.to_size):
                if s1 is not None and s2 is not None and s2 > s1:
                    raise ValueError(f'Shape {input_shape} is too small to be cropped to {self.to_size}')
        bbox_size = [input_shape[0] if input_shape[0] is not None else input_shape[0],
                     *self.to_size,
                     input_shape[-1] if input_shape[-1] is not None else input_shape[-1]]
        return bbox_size

    def get_config(self):
        config = {attr: getattr(self, attr) for attr in ['to_size', 'allow_smaller_outputs']}
        base_config = super(CropLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RandomRotation3D(Layer):
    def __init__(self, axis, max_abs_angle_deg, interpolation="NEAREST", crop_to_size=None, parallel_rotations=None, **kwargs):
        """
        supported interpolations: "NEAREST", "BILINEAR"
        axis must be either 0, 1 or 2
        """
        super(RandomRotation3D, self).__init__(**kwargs)
        self.parallel_rotations = parallel_rotations
        assert crop_to_size is None or len(crop_to_size) == 3
        self.crop_to_size = crop_to_size
        self.interpolation = interpolation
        self.max_abs_angle_deg = max_abs_angle_deg
        self.axis = axis
        if axis not in AVAILABLE_AXES:
            raise ValueError
        self.supports_masking = True

    def call(self, inputs, *args, **kwargs):
        if self.crop_to_size:
            half_crop = tuple(round(s // 2) for s in self.crop_to_size)

        def noised():
            if self.max_abs_angle_deg == 0:
                return inputs
            input_shape: Tensor = tensorflow.keras.backend.shape(inputs)
            assert input_shape.shape.as_list()[0] == 5, inputs.shape.as_list()

            non_rotation_axes = sorted((self.axis + offset) % 3 + 1 for offset in [1, 2])
            assert all(a - 1 in AVAILABLE_AXES for a in non_rotation_axes + [self.axis + 1])
            assert len(set(non_rotation_axes + [self.axis + 1])) == 3

            # rotate allows only one batch axis so we need the map_fn for the axis that is rotated on
            # rotate expects NWHC but input is NDWHC
            # so we put the rotation axis to the beginning and map over it
            transpose_permutation = [self.axis + 1, 0, *non_rotation_axes, 4]
            revert_transpose_permutation = [transpose_permutation.index(i) for i in range(5)]
            transposed_images = tensorflow.transpose(inputs, transpose_permutation)
            # batch index is now at second position

            # also we need a separate angle for each image
            if force_rad is not None:
                angle_per_image = -force_rad
            else:
                angle_per_image = tensorflow.keras.backend.random_uniform(shape=input_shape[0:1],
                                                                          minval=-radians(self.max_abs_angle_deg),
                                                                          maxval=radians(self.max_abs_angle_deg))

            monkey_patch_tensor_to_array()

            rotated_transposed_images = tensorflow.nest.map_structure(
                tensorflow.stop_gradient,
                tensorflow.map_fn(
                    lambda img: rotate(images=img,
                                       angles=angle_per_image,
                                       interpolation=self.interpolation),
                    elems=transposed_images,
                    parallel_iterations=self.parallel_rotations, ))

            # crop if necessary
            if self.crop_to_size:
                center = tensorflow.keras.backend.shape(inputs) // 2
                bbox_begin = tensorflow.keras.backend.concatenate([[0], (center[1:-1] - half_crop), [0]], axis=0)
                bbox_begin_transposed = [bbox_begin[transpose_permutation[i]] for i in range(5)]
                bbox_size = [inputs.shape.as_list()[0] if inputs.shape.as_list()[0] is not None else tensorflow.keras.backend.shape(inputs)[0],
                             *self.crop_to_size,
                             inputs.shape.as_list()[-1] if inputs.shape.as_list()[-1] is not None else tensorflow.keras.backend.shape(inputs)[-1]]
                bbox_size_transposed = [bbox_size[transpose_permutation[i]] for i in range(5)]

                cropped_transposed_images = tensorflow.slice(rotated_transposed_images, bbox_begin_transposed, bbox_size_transposed)
            else:
                cropped_transposed_images = rotated_transposed_images
            # now revert the transposing of images
            rotated_images = tensorflow.transpose(cropped_transposed_images, revert_transpose_permutation)

            return rotated_images

        def not_noised():
            if self.crop_to_size:
                center = tensorflow.keras.backend.shape(inputs) // 2
                bbox_begin = tensorflow.keras.backend.concatenate([[0], (center[1:-1] - half_crop), [0]], axis=0)
                bbox_size = [inputs.shape.as_list()[0] if inputs.shape.as_list()[0] is not None else tensorflow.keras.backend.shape(inputs)[0],
                             *self.crop_to_size,
                             inputs.shape.as_list()[-1] if inputs.shape.as_list()[-1] is not None else tensorflow.keras.backend.shape(inputs)[-1]]

                cropped_images = tensorflow.slice(inputs, bbox_begin, bbox_size)
                return cropped_images
            else:
                return inputs

        # print('WARNING: Applying random rotations also in test phase'); return noised()
        return tensorflow.keras.backend.in_train_phase(noised, not_noised)

    def get_config(self):
        config = {attr: getattr(self, attr) for attr in ['axis', 'max_abs_angle_deg', 'interpolation', 'crop_to_size']}
        base_config = super(RandomRotation3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        if self.crop_to_size is not None:
            return (input_shape[0], *self.crop_to_size, input_shape[-1])
        else:
            return input_shape


def monkey_patch_tensor_to_array():
    def __array__(self):
        raise TypeError(
            "Cannot convert a symbolic Tensor ({}) to a numpy array."
            " This error may indicate that you're trying to pass a Tensor to"
            " a NumPy call, which is not supported".format(self.name))

    Tensor.__array__ = __array__


force_rad: Optional[int] = None

from tensorflow.python.keras.utils.generic_utils import _GLOBAL_CUSTOM_OBJECTS

_GLOBAL_CUSTOM_OBJECTS.update({
    obj.__name__: obj
    # If your model uses any custom layers, add them here
    for obj in [RandomRotation3D, CropLayer]
})
