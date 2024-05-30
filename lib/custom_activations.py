# noinspection PyUnresolvedReferences
import string

import tensorflow.keras.backend
import tensorflow.keras.layers
import tensorflow.keras.utils
from tensorflow.keras.layers import ELU


def negate(x):
    """Linear activation that negates input.
    """
    return -x


def exp_of_1000th(x):
    return tensorflow.keras.activations.exponential(x / 1000)


def exp_of_100th(x):
    return tensorflow.keras.activations.exponential(x / 100)


def exp_of_10th(x):
    return tensorflow.keras.activations.exponential(x / 10)


class ELUPlusOne(ELU):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__name__ = 'ELUPlusOne'

    def call(self, inputs):
        return super().call(inputs) + 1


tensorflow.keras.utils.get_custom_objects().update({
    obj.__name__: obj
    # If your model uses any custom layers, add them here
    for obj in [exp_of_10th, exp_of_100th, exp_of_1000th, ELUPlusOne()]
})
