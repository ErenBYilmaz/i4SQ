import typing
from typing import Iterable, Dict, Union

if typing.TYPE_CHECKING:
    from tensorflow.keras import Model

from lib.image_processing_tool import TrainableImageProcessingTool


def try_clone_keras_object(obj, custom_objects=None):
    from tensorflow.python.keras.utils.generic_utils import has_arg, CustomObjectScope, _GLOBAL_CUSTOM_OBJECTS
    cls = obj.__class__
    if hasattr(cls, 'from_config') and hasattr(obj, 'get_config'):
        config = obj.get_config()
        custom_objects = custom_objects or {}
        if has_arg(cls.from_config, 'custom_objects'):
            return cls.from_config(
                config,
                custom_objects=dict(list(_GLOBAL_CUSTOM_OBJECTS.items()) +
                                    list(custom_objects.items())))
        with CustomObjectScope(custom_objects):
            return cls.from_config(config)
    else:
        return obj


def clone_model_or_tool(model: Union['Model', TrainableImageProcessingTool], custom_objects=None) -> Union[TrainableImageProcessingTool, 'Model']:
    if isinstance(model, TrainableImageProcessingTool):
        return model.clone()
    else:
        return clone_compiled_model(model, custom_objects=custom_objects)


def clone_compiled_model(model: 'Model', custom_objects=None):
    import tensorflow
    m2 = tensorflow.keras.models.clone_model(model)

    # try to clone metrics
    metrics = get_metrics_from_model(model)
    if isinstance(metrics, Dict):
        metrics = {
            output_name: [try_clone_keras_object(metric, custom_objects=custom_objects) for metric in metrics]
            for output_name, metrics in metrics.items()
        }
    elif isinstance(metrics, Iterable):
        metrics = [
            try_clone_keras_object(metric, custom_objects=custom_objects)
            for metric in metrics
        ]

    weighted_metrics = get_weighted_metrics_from_model(model)
    if isinstance(weighted_metrics, Dict):
        weighted_metrics = {
            output_name: [try_clone_keras_object(metric, custom_objects=custom_objects) for metric in metrics]
            for output_name, metrics in weighted_metrics.items()
        }
    elif isinstance(weighted_metrics, Iterable):
        weighted_metrics = [
            try_clone_keras_object(metric, custom_objects=custom_objects)
            for metric in weighted_metrics
        ]

    loss_weights = get_loss_from_model(model)
    m2.compile(optimizer=try_clone_keras_object(model.optimizer, custom_objects=custom_objects),
               loss=try_clone_keras_object(model.loss, custom_objects=custom_objects),
               metrics=metrics,
               loss_weights=loss_weights,
               weighted_metrics=weighted_metrics)
    assert model.output_names == m2.output_names
    return m2


def get_loss_from_model(model):
    if model.compiled_loss is None:
        loss_weights = model.loss_weights
    else:
        loss_weights = model.compiled_loss._user_loss_weights
    return loss_weights


def get_weighted_metrics_from_model(model):
    if model.compiled_metrics is None:
        weighted_metrics = model._compile_weighted_metrics
    else:
        weighted_metrics = model.compiled_metrics._user_weighted_metrics
    return weighted_metrics


def get_metrics_from_model(model):
    if model.compiled_metrics is None:
        metrics = model._compile_metrics
    else:
        metrics = model.compiled_metrics._user_metrics
    return metrics
