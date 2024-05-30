from copy import deepcopy
from typing import Union, List

import numpy
import tensorflow.keras.backend
from tensorflow.keras import Model
from tensorflow.keras.layers import serialize
# noinspection PyUnresolvedReferences
from tensorflow.python.keras.engine.input_layer import InputLayer
from tensorflow.python.keras.models import model_from_config
from tensorflow.keras.layers import Layer


def flatten_model(model: Model, custom_objects=None) -> Model:
    # custom_objects = custom_objects or {}
    # custom_objects = {**_GLOBAL_CUSTOM_OBJECTS, **custom_objects}
    config = serialize(model)
    flattened_config = flattened_model_config(config)
    if flattened_config == config:
        return model  # nothing to do

    # build model graph
    flat_model: Model = model_from_config(flattened_config, custom_objects=custom_objects)
    # load weights
    assert all(w1.shape == w2.shape for w1, w2 in zip(flattened_weights(model), flat_model.get_weights()))
    flat_model.set_weights(flattened_weights(model))
    return flat_model


def flattened_weights(model: Model) -> List[numpy.ndarray]:
    """Similar to model.get_weights() but returns the weights in a different order.
    More precisely, the weights of sub-models are ordered by the order of sub-layers.
    # Returns
        A flat list of Numpy arrays.
    """
    return tensorflow.keras.backend.batch_get_value(flattened_weight_tensors(model))


def flattened_weight_tensors(model: Model) -> List[numpy.ndarray]:
    """Same as flattened_weights, but does not return the actual weight values but the tensors holding the data.
    # Returns
        A flat list of Tensor objects.
    """
    weights = model._trainable_weights + model._non_trainable_weights
    for layer in model.layers:
        if isinstance(layer, Model):
            weights += flattened_weight_tensors(layer)
        else:
            weights += layer.weights
    return weights


def flattened_model_config(model_or_config_dict: Union[Model, dict]):
    if hasattr(model_or_config_dict, 'get_config'):
        config: dict = serialize(model_or_config_dict)
    else:
        config: dict = model_or_config_dict
    config = deepcopy(config)
    idx = 0
    output_node_map = {}
    # iterate the model's layers' configs
    layers = config['config']['layers']
    while idx < len(layers):
        layer = layers[idx]
        layer_config = layer['config']
        # connect output nodes of prev. sub-models to inputs of this layer
        for inbound_nodes in layer['inbound_nodes']:
            for inbound_node in inbound_nodes:
                if inbound_node[0] in output_node_map:
                    inbound_node[0] = output_node_map[inbound_node[0]]
                    inbound_node[1] = -1  # TODO just a good guess, this may not always be correct

        if 'input_layers' in layer_config and 'output_layers' in layer_config:
            sub_config = flattened_model_config(layer)['config']
            sub_inputs = sub_config['input_layers']
            sub_layers = sub_config['layers']
            sub_outputs = sub_config['output_layers']
            # connect inputs of sub_model
            sub_input_names = [sub_input[0] for sub_input in sub_inputs]
            sub_input_configs = [
                sub_layer for sub_layer in sub_layers
                if sub_layer['name'] in sub_input_names
            ]
            assert len(sub_input_configs) > 0
            assert len(layer['inbound_nodes']) == len(sub_input_configs)
            for sub_input_config, inbound_node in zip(sub_input_configs, layer['inbound_nodes']):
                assert sub_input_config['class_name'] == InputLayer.__name__  # may not hold in general, but code may fail otherwise
                sub_input_config['inbound_nodes'].append(inbound_node)
                sub_input_config['class_name'] = Layer.__name__  # replace input layers by empty layers(was easier to implement than deleting)
                sub_input_config['config'].pop('sparse')

            # determine outputs of sub_model
            sub_output_names = [output[0] for output in sub_outputs]
            for sub_output_name in sub_output_names:
                assert layer['name'] not in output_node_map
                output_node_map[layer['name']] = sub_output_name

            # replace single "Model"-Layer in the config with multiple layers
            layers[idx:idx + 1] = sub_config['layers']
            idx += len(sub_config['layers'])
        else:
            idx += 1

    return config
