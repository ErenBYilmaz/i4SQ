# noinspection PyUnresolvedReferences
import string
from abc import abstractmethod
from typing import Sized, Union, List, Any

import numpy
import scipy.stats
import tensorflow
import tensorflow.keras
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.layers import Layer, Dropout
from tensorflow.python.keras.layers.convolutional import Conv
from tensorflow_addons.layers import InstanceNormalization


def _broadcast1d(s1: Union[Sized, Any], s2: Union[Sized, Any]):
    """
    If input is a string or any non-sized object, put it in a single item list
    then broadcast as if the arguments where 1d numpy arrays
    see also: https://docs.scipy.org/doc/numpy/user/theory.broadcasting.html#array-broadcasting-in-numpy
    """
    if not isinstance(s1, Sized) or isinstance(s1, str):
        s1 = [s1]
    if not isinstance(s2, Sized) or isinstance(s2, str):
        s2 = [s2]
    if len(s1) == len(s2):
        return s1, s2
    elif len(s1) != 1 and len(s2) != 1:
        raise ValueError('Could not broadcast.')
    elif len(s1) == 1:
        return s1 * len(s2), s2
    else:  # len(s2) == 1
        return s2 * len(s1), s1


def _broadcast_left_1d(s1: Union[Sized, Any], s2: Sized) -> Union[Sized, List[Any]]:
    """
    If input is a string or any non-sized object, put it in a single item list
    if it is a single item list afterwards make it the same length as the second list
    """
    if not isinstance(s1, Sized) or isinstance(s1, str):
        s1 = [s1]
    if len(s1) == len(s2):
        return s1
    elif len(s1) == 1:
        return s1 * len(s2)
    else:
        raise ValueError('Could not broadcast.')


class ShapedGaussianNoise(Layer):
    def __init__(self, mean=0., std=1., size=None, **kwargs):
        super(ShapedGaussianNoise, self).__init__(**kwargs)
        self.supports_masking = True
        self.std = std
        self.mean = mean
        self.size = size

    def call(self, inputs, *args, **kwargs):
        def noised():
            if self.mean == 0 and self.std == 0:
                return inputs
            input_shape = tensorflow.keras.backend.shape(inputs)
            if self.size is None:
                shape = input_shape
            else:
                size = numpy.array([0] + [
                    sz if sz else 0
                    for sz in self.size
                ])
                size_multiplier = numpy.array([
                    1 if sz else 0
                    for sz in size
                ])
                input_shape_multiplier = 1 - size_multiplier
                shape = (tensorflow.keras.backend.constant(size * size_multiplier, dtype='int32')
                         + (input_shape * tensorflow.keras.backend.constant(input_shape_multiplier, dtype='int32')))
            noise = tensorflow.keras.backend.random_normal(shape=shape, mean=self.mean, stddev=self.std)
            return tensorflow.ensure_shape(inputs + noise, shape=inputs.shape)

        return tensorflow.keras.backend.in_train_phase(noised, inputs)

    def get_config(self):
        config = {attr: getattr(self, attr) for attr in ['mean', 'std', 'size']}
        base_config = super(ShapedGaussianNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class GaussianKernel(Initializer):
    def __init__(self, sigma, mean=0., ):
        self.mean = mean
        self.sigma = sigma

    def __call__(self, shape, dtype=None, **kwargs):
        mean = self.mean
        if isinstance(mean, int):
            # last two dimensions are probably number of input and output channels
            mean = [mean] * (len(shape) - 2)
        if len(mean) == len(shape) - 2:
            mean = mean + [0, 0]
        if len(mean) != len(shape):
            raise ValueError

        sigma = self.sigma
        if isinstance(sigma, int):
            # last two dimensions are probably number of input and output channels
            sigma = [sigma] * (len(shape) - 2)
        if len(sigma) == len(shape) - 2:
            sigma = sigma + [1, 1]
        if len(sigma) != len(shape):
            raise ValueError
        valss = [
            scipy.stats.norm(loc=m, scale=sig).pdf(numpy.arange(start=-(s // 2), stop=(s + 1) // 2, dtype='float32'))
            for s, m, sig in zip(shape, mean, sigma)
            # for s in [float(s)]
            for m in [float(m)]
            for sig in [float(sig)]
        ]
        assert all(len(vals.shape) == 1 for vals in valss)
        assert all(vals.shape[0] == s for s, vals in zip(shape, valss))

        gauss_kernel = numpy.einsum(','.join(string.ascii_lowercase[:len(shape)]) + '->' + string.ascii_lowercase[:len(shape)],
                                    *valss)
        gauss_kernel = gauss_kernel / gauss_kernel.sum()

        return gauss_kernel

    def get_config(self):
        return {
            'mean': self.mean,
            'sigma': self.sigma,
        }


class GaussianBlurND(Conv):
    """
    deterministically blurs the image using a gaussian filter
    like https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
    """

    def __init__(self,
                 rank,
                 sigma,
                 truncate=4.0,
                 min_filter_size=0,
                 suppress_filter_warning=False,
                 filters=1,
                 strides=1,
                 dilation_rate=1,
                 padding='same',
                 activation=None,
                 use_bias=False,
                 **kwargs):
        self.suppress_filter_warning = suppress_filter_warning
        self.sigma = _broadcast_left_1d(sigma, range(rank))
        self.truncate = _broadcast_left_1d(truncate, range(rank))
        self.min_filter_size = min_filter_size
        _broadcast_left_1d(self.min_filter_size, range(rank))
        kernel_size = tuple(max(0 if t * s == 0 else round((t * s + 1) // 2) + 1, min_filter_size)
                            for t, s in zip(self.truncate, self.sigma))
        super(GaussianBlurND, self).__init__(rank=rank,
                                             filters=filters,
                                             kernel_size=kernel_size,
                                             strides=strides,
                                             padding=padding,
                                             dilation_rate=dilation_rate,
                                             activation=activation,
                                             use_bias=use_bias,
                                             kernel_initializer=GaussianKernel(mean=0, sigma=self.sigma),
                                             **kwargs)
        assert self.rank == rank
        self.trainable = False
        self.supports_masking = True

    def call(self, inputs, training=None):
        if not self.suppress_filter_warning:
            if tensorflow.keras.backend.image_data_format() == 'channels_first':
                channel_axis = 1
            elif tensorflow.keras.backend.image_data_format() == 'channels_last':
                channel_axis = -1
            else:
                raise NotImplementedError
            if inputs.shape.as_list()[channel_axis] != 1:
                print(
                    'WARNING: Using Gaussian blur with more than 1 channel will apply the filter to each channel then average the results to a single channel. '
                    'If this is intended, you can suppress this warning with `suppress_filter_warning=True`.')

        return super(GaussianBlurND, self).call(inputs)

    def get_config(self):
        config = {attr: getattr(self, attr) for attr in ['suppress_filter_warning', 'sigma', 'truncate', 'min_filter_size']}
        base_config = super(GaussianBlurND, self).get_config()
        base_config.pop('kernel_size')
        base_config.pop('kernel_initializer')
        return dict(list(base_config.items()) + list(config.items()))


class RandomBlurND(GaussianBlurND):
    def __init__(self,
                 sigma,
                 blur_probability: float,
                 truncate=4.0,
                 min_filter_size=0,
                 suppress_filter_warning=False,
                 filters=1,
                 strides=1,
                 dilation_rate=1,
                 padding='same',
                 activation=None,
                 use_bias=False,
                 **kwargs):
        super().__init__(sigma=sigma,
                         rank=self.conv_rank(),
                         truncate=truncate,
                         min_filter_size=min_filter_size,
                         suppress_filter_warning=suppress_filter_warning,
                         filters=filters,
                         strides=strides,
                         dilation_rate=dilation_rate,
                         padding=padding,
                         activation=activation,
                         use_bias=use_bias,
                         **kwargs)
        self.blur_probability = blur_probability

    @abstractmethod
    def conv_rank(self):
        raise NotImplementedError('Abstract method')

    def call(self, inputs, training=None):
        def noised():
            blurred = super(GaussianBlurND, self).call(inputs)
            if self.blur_probability == 1:
                return blurred
            else:
                return tensorflow.cond(tensorflow.keras.backend.random_uniform(()) < self.blur_probability,
                                       lambda: blurred,
                                       lambda: inputs)

        return tensorflow.keras.backend.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = {attr: getattr(self, attr) for attr in ['blur_probability']}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RandomBlur3D(RandomBlurND):
    def conv_rank(self):
        return 3


class RandomBlur2D(RandomBlurND):
    def conv_rank(self):
        return 2


class RandomChoice(Layer):
    def __init__(self, p_cum=None, **kwargs):
        super().__init__(**kwargs)
        if p_cum is not None:
            if any(x > y for x, y in zip(p_cum, p_cum[1:])):
                raise ValueError('p_cum is not ordered')
            if p_cum[-1] != 1:
                raise ValueError
        self.p_cum = p_cum

    def call(self, inputs, training=None):
        if self.p_cum is None:
            p_cum = [i / len(inputs) for i in range(1, len(inputs))] + [1]
            assert not any(x > y for x, y in zip(p_cum, p_cum[1:]))
        else:
            p_cum = self.p_cum
        assert len(inputs) == len(p_cum), (len(inputs), len(p_cum))

        def noised():
            r = tensorflow.keras.backend.random_uniform(())
            return tensorflow.case([(r < p, lambda i=i: i) for p, i in zip(p_cum, inputs)], exclusive=False)

        return tensorflow.keras.backend.in_train_phase(noised, inputs[0], training=training)

    def get_config(self):
        config = {attr: getattr(self, attr) for attr in ['p_cum']}
        base_config = super(RandomChoice, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        result = list(None for _ in input_shape[0])
        for axis in range(len(result)):
            for i in input_shape:
                if i[axis] is not None:
                    if result[axis] is None:
                        result[axis] = i[axis]
                    if result[axis] != i[axis]:
                        raise ValueError('Incompatible shapes', input_shape)
        return tuple(result)


class MCDropout(Dropout):
    def call(self, inputs, *args, **kwargs):
        return super().call(inputs, training=True)


class ShapedMultiplicativeGaussianNoise(Layer):
    def __init__(self, mean=1., std=1., size=None, **kwargs):
        super(ShapedMultiplicativeGaussianNoise, self).__init__(**kwargs)
        self.supports_masking = True
        self.std = std
        self.mean = mean
        self.size = size

    def call(self, inputs, *args, **kwargs):
        def noised():
            if self.mean == 1 and self.std == 0:
                return inputs
            input_shape = tensorflow.keras.backend.shape(inputs)
            if self.size is None:
                shape = input_shape
            else:
                size = numpy.array([0] + [
                    sz if sz else 0
                    for sz in self.size
                ])
                size_multiplier = numpy.array([
                    1 if sz else 0
                    for sz in size
                ])
                input_shape_multiplier = 1 - size_multiplier
                shape = (tensorflow.keras.backend.constant(size * size_multiplier, dtype='int32')
                         + (input_shape * tensorflow.keras.backend.constant(input_shape_multiplier, dtype='int32')))
            noise = tensorflow.keras.backend.random_normal(shape=shape, mean=self.mean, stddev=self.std)
            return tensorflow.ensure_shape(inputs * noise, shape=inputs.shape)

        return tensorflow.keras.backend.in_train_phase(noised, inputs)

    def get_config(self):
        config = {attr: getattr(self, attr) for attr in ['mean', 'std', 'size']}
        base_config = super(ShapedMultiplicativeGaussianNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class PointWiseMultiplication(Layer):
    def __init__(self, factor: float, **kwargs):
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, inputs, *args, **kwargs):
        if self.factor != 1:
            return inputs * self.factor
        else:
            return inputs

    def get_config(self):
        config = {attr: getattr(self, attr) for attr in ['factor']}
        base_config = super(PointWiseMultiplication, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Stack(Layer):
    def __init__(self, axis: int, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs, *args, **kwargs):
        return tensorflow.keras.backend.stack(inputs, axis=self.axis)

    def get_config(self):
        config = {attr: getattr(self, attr) for attr in ['axis']}
        base_config = super(Stack, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SpecificChannel(Layer):
    def __init__(self, channel_idx: int, **kwargs):
        super().__init__(**kwargs)
        self.channel_idx = channel_idx

    def call(self, inputs, *args, **kwargs):
        if tensorflow.keras.backend.image_data_format() != 'channels_last':
            raise NotImplementedError
        return inputs[..., self.channel_idx:self.channel_idx + 1]

    def get_config(self):
        config = {attr: getattr(self, attr) for attr in ['channel_idx']}
        base_config = super(SpecificChannel, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MyInstanceNormalization(InstanceNormalization):
    def __init__(self, **kwargs):
        """This suppresses a useless warning in the super classes constructor"""
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(**kwargs)
        assert self.groups == -1


from tensorflow.python.keras.utils.generic_utils import _GLOBAL_CUSTOM_OBJECTS

_GLOBAL_CUSTOM_OBJECTS.update({
    obj.__name__: obj
    # If your model uses any custom layers, add them here
    for obj in [
        ShapedGaussianNoise,
        ShapedMultiplicativeGaussianNoise,
        RandomChoice,
        RandomBlur3D,
        GaussianBlurND,
        PointWiseMultiplication,
        SpecificChannel,
        Stack,
        MyInstanceNormalization,
    ]
})
