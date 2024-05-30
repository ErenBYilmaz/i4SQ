import functools
from typing import Tuple

import tensorflow
import tensorflow.keras
import tensorflow.keras.backend
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras.metrics import MeanMetricWrapper, Recall
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.ops import math_ops
from tensorflow_addons.metrics import F1Score


def rmse(y_true, y_pred):
    return tensorflow.keras.backend.sqrt(tensorflow.keras.backend.mean(tensorflow.keras.backend.square(y_pred - y_true)))


def drmse(y_true, y_pred):
    return 2. * tensorflow.keras.backend.sqrt(tensorflow.keras.backend.mean(tensorflow.keras.backend.square(y_pred - y_true)))


def genant_score_from_deformities(x):
    return tensorflow.keras.backend.max(tensorflow.keras.backend.cast(x >= 0.2, 'float32') +
                                        tensorflow.keras.backend.cast(x >= 0.25, 'float32') +
                                        tensorflow.keras.backend.cast(x >= 0.4, 'float32'))


def genantscore_custom_loss(y_true, y_pred):
    cce_loss = tensorflow.keras.losses.categorical_crossentropy(y_true, y_pred)

    true_genantscore = tensorflow.math.top_k(y_true, 1).indices  # top_k returns the largest values inside y true

    gentant_scores = tensorflow.constant(([0.0, 1.0, 2.0, 3.0]))
    regression_output = tensorflow.math.multiply(gentant_scores, y_pred)
    regression_output = tensorflow.reduce_sum(regression_output, 1, keepdims=True)

    mse_loss = tensorflow.keras.losses.mean_squared_error(true_genantscore, regression_output)

    total_loss = tensorflow.math.add(mse_loss, cce_loss)
    return total_loss


def fracture_class_from_deformities(x):
    return tensorflow.keras.backend.max(tensorflow.keras.backend.cast(x >= 0.2, 'float32'), axis=-1)


def deformity_fracture_accuracy(y_true, y_pred):
    return tensorflow.keras.metrics.binary_accuracy(fracture_class_from_deformities(y_true),
                                                    fracture_class_from_deformities(y_pred))


def genantscore_regression_accuracy(y_true, y_pred, sample_weight=None, mean=True):
    rounded_y = tensorflow.round(y_pred)  # round all predictions of current batch to the nearest whole number

    zero = tensorflow.zeros_like(rounded_y)  # create tensor full of zeros and threes with the same shape as the predictions
    three = tensorflow.math.scalar_mul(3, tensorflow.ones_like(rounded_y))

    rounded_y = tensorflow.maximum(zero, rounded_y)  # if one prediction is less than zero, set it to zero
    rounded_y = tensorflow.minimum(three, rounded_y)  # if one prediction is more than three, set it to three

    equals = tensorflow.math.equal(y_true,
                                   rounded_y)  # tensor that contains 1 for each sample if rounded prediction and ground truth were the same, 0 otherwise
    equals = tensorflow.cast(equals, dtype='float32')  # both need to have the same dtype
    if sample_weight is not None:
        equals *= sample_weight  # sample_weights for counting correct values
    if mean:
        result = tensorflow.reduce_mean(equals)  # accuracy is the average of all comparisons (True => sample_weights, False => 0)
    else:
        result = equals
    return result


class GenantScoreRegressionAccuracy(MeanMetricWrapper):
    def __init__(self, name='genant_score_regression_accuracy', dtype=None):
        fn = functools.partial(genantscore_regression_accuracy, mean=False)
        super(GenantScoreRegressionAccuracy, self).__init__(fn, name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        super(GenantScoreRegressionAccuracy, self).update_state(y_true, y_pred, sample_weight)


class RestrictedCategoricalAccuracy(MeanMetricWrapper):

    def __init__(self, remove_classes: Tuple[int, ...],
                 keep_classes: Tuple[int, ...],
                 name='restricted_categorical_accuracy',
                 dtype=None):
        super(RestrictedCategoricalAccuracy, self).__init__(
            _restricted_categorical_accuracy(remove_classes=remove_classes, keep_classes=keep_classes),
            name, dtype=dtype)
        self.fixed_name = self.name
        self.remove_classes = remove_classes
        self.keep_classes = keep_classes

    def get_config(self):
        return {
            **super(RestrictedCategoricalAccuracy, self).get_config(),
            'remove_classes': self.remove_classes,
            'keep_classes': self.keep_classes,
            'name': self.fixed_name,
        }


def _restricted_categorical_accuracy(remove_classes: Tuple[int, ...], keep_classes: Tuple[int, ...]):
    if len(set(remove_classes).intersection(keep_classes)) > 0:
        raise ValueError

    keep_samples = _decision_count(remove_classes=remove_classes, keep_classes=keep_classes)

    def rca(y_true, y_pred):
        # assert y_pred.shape.as_list()[-1] == y_true.shape.as_list()[-1] or y_pred.shape.as_list()[-1] is None or y_true.shape.as_list()[-1] is None
        # assert len(y_pred.shape.as_list()) == len(y_true.shape.as_list()) == 2
        # if len(remove_classes) > 1:
        #     raise NotImplementedError
        # remove_class = remove_classes[0]
        # keep = tensorflow.math.logical_and(keras.backend.not_equal(keras.backend.argmax(y_pred, axis=-1), remove_class),
        #                                    keras.backend.not_equal(keras.backend.argmax(y_true, axis=-1), remove_class))
        keep = keep_samples(y_true, y_pred)
        yt2 = tensorflow.keras.backend.stack([y_true[keep][..., c] for c in keep_classes], axis=1)
        yp2 = tensorflow.keras.backend.stack([y_pred[keep][..., c] for c in keep_classes], axis=1)
        assert yt2.shape.as_list()[-1] == yp2.shape.as_list()[-1] == len(keep_classes)
        assert len(y_pred.shape.as_list()) == len(y_true.shape.as_list()) == 2
        return categorical_accuracy(yt2, yp2)

    rca.__name__ = 'restricted_categorical_accuracy'
    return rca


class DecisionCounter(MeanMetricWrapper):
    def __init__(self, remove_classes: Tuple[int, ...],
                 keep_classes: Tuple[int, ...],
                 name='decision_counter',
                 dtype=None):
        super(DecisionCounter, self).__init__(
            _decision_count(remove_classes=remove_classes, keep_classes=keep_classes),
            name, dtype=dtype)
        self.fixed_name = self.name
        self.remove_classes = remove_classes
        self.keep_classes = keep_classes

    def get_config(self):
        return {
            **super(DecisionCounter, self).get_config(),
            'remove_classes': self.remove_classes,
            'keep_classes': self.keep_classes,
            'name': self.fixed_name,
        }


def _decision_count(remove_classes: Tuple[int, ...], keep_classes: Tuple[int, ...]):
    if len(set(remove_classes).intersection(keep_classes)) > 0:
        raise ValueError

    def dc(y_true, y_pred):
        assert y_pred.shape.as_list()[-1] == y_true.shape.as_list()[-1] or y_pred.shape.as_list()[-1] is None or y_true.shape.as_list()[-1] is None
        assert len(y_pred.shape.as_list()) == len(y_true.shape.as_list()) == 2
        if len(remove_classes) > 1:
            raise NotImplementedError
        remove_class = remove_classes[0]
        keep = tensorflow.math.logical_and(tensorflow.keras.backend.not_equal(tensorflow.keras.backend.argmax(y_pred, axis=-1), remove_class),
                                           tensorflow.keras.backend.not_equal(tensorflow.keras.backend.argmax(y_true, axis=-1), remove_class))
        return keep

    dc.__name__ = 'decision_count'
    return dc


class Specificity(Recall):
    def correct(self):
        return self.true_positives  # the variable was named after the super class Recall, these are not actually true positives counted here

    def incorrect(self):
        return self.false_negatives  # the variable was named after the super class Recall, these are not actually false negatives counted here

    def incorrect_key(self):
        return metrics_utils.ConfusionMatrix.FALSE_POSITIVES

    def correct_key(self):
        return metrics_utils.ConfusionMatrix.TRUE_NEGATIVES

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Copied from Recall class, but using different keys for the confusion matrix variables"""
        return metrics_utils.update_confusion_matrix_variables(
            {
                self.correct_key(): self.correct(),
                self.incorrect_key(): self.incorrect(),
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight)


class PPV(Specificity):
    def incorrect_key(self):
        return metrics_utils.ConfusionMatrix.FALSE_POSITIVES

    def correct_key(self):
        return metrics_utils.ConfusionMatrix.TRUE_POSITIVES


class NPV(Specificity):
    def incorrect_key(self):
        return metrics_utils.ConfusionMatrix.FALSE_NEGATIVES

    def correct_key(self):
        return metrics_utils.ConfusionMatrix.TRUE_NEGATIVES


def deformity_specificity(y_true, y_pred):
    raise NotImplementedError
    # return specificity(fracture_class_from_deformities(y_true), fracture_class_from_deformities(y_pred))


def deformity_sensitivity(y_true, y_pred):
    raise NotImplementedError
    # return sensitivity(fracture_class_from_deformities(y_true), fracture_class_from_deformities(y_pred))


def zero_function(_y_true, _y_pred):
    return tensorflow.keras.backend.constant(0)


def mean_squared_error_by_variance(y_true, y_pred):
    """
    Computes the mean squared error between labels and predictions and then divides it by the variance of y_pred.
    Adapted from keras.losses.mean_squared_error
    """
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    return backend.mean(
        math_ops.squared_difference(y_pred, y_true) / math_ops.reduce_variance(y_true, axis=0),
        axis=-1)


class MeanSquaredErrorByVariance(MeanMetricWrapper):
    """
    Computes the mean squared error between `y_true` and `y_pred` and then divides it by the variance of y_pred.
    Adapted from keras.metrics.MeanSquaredError
    """

    def __init__(self, name='mse_by_var', dtype=None):
        super(MeanSquaredErrorByVariance, self).__init__(mean_squared_error_by_variance, name, dtype=dtype)


class F1ScoreWithoutResetStatesWarning(F1Score):
    def reset_state(self):
        return self.reset_states()


from tensorflow.python.keras.utils.generic_utils import _GLOBAL_CUSTOM_OBJECTS

sens = sensitivity = Recall
spec = specificity = Specificity

_GLOBAL_CUSTOM_OBJECTS.update({
    obj.__name__: obj
    # If your model uses any custom layers, add them here
    for obj in [zero_function, RestrictedCategoricalAccuracy, DecisionCounter, sensitivity, specificity, spec, sens,
                PPV, NPV, F1ScoreWithoutResetStatesWarning]
})
