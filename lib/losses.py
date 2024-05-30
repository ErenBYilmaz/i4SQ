from typing import Union

import tensorflow.keras
import numpy


def weighted_pixelwise_crossentropy(class_weights, name=None, only_specific_class: Union[int, slice] = slice(None)):
    class_weights = numpy.asarray(class_weights)
    assert len(class_weights.shape) == 1

    if isinstance(only_specific_class, int):
        only_specific_class = slice(only_specific_class, only_specific_class + 1)

    def select_class(x):
        assert isinstance(only_specific_class, slice)
        sliced = x[..., only_specific_class]  # only the class(es) that was specified by the argument
        assert len(sliced.shape) == len(x.shape)
        return sliced

    def loss(y_true, y_pred):
        epsilon = tensorflow.keras.backend.epsilon()
        y_pred = tensorflow.keras.backend.clip(y_pred, epsilon, 1. - epsilon)  # clip values near 1 and 0 (makes gradient 0 there)
        return - tensorflow.keras.backend.mean(
            tensorflow.keras.backend.sum(select_class(y_true) * tensorflow.keras.backend.log(select_class(y_pred)) * select_class(class_weights),  # weighted categorical crossentropy
                              axis=-1),  # sum over classes
            axis=None)  # mean over voxels and batch

    if name is None:
        if isinstance(only_specific_class, int):
            loss.__name__ = f'weighted_pixelwise_crossentropy_class{only_specific_class}'
        else:
            loss.__name__ = f'weighted_pixelwise_crossentropy'
    else:
        loss.__name__ = name

    return loss


def per_class_dice(class_idx, smooth=0.001, axis=(1, 2, 3)):
    import tensorflow.keras.backend as K

    def dice(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon(), )
        individual_scores = (2. * (K.sum(y_true * y_pred, axis=axis) + smooth / 2) /
                             (K.sum(y_true, axis=axis) + K.sum(y_pred, axis=axis) + smooth))
        return K.mean(individual_scores[:, class_idx])

    dice.__name__ = f'per_class_dice_{class_idx}'

    return dice
