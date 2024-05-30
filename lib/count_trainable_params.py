import numpy


def count_trainable_params(model):
    import tensorflow.keras
    result = 0
    already_counted = set()
    for idx in range(len(model.trainable_weights)):
        name = model.trainable_weights[idx].name
        if name not in already_counted:
            result += tensorflow.keras.backend.count_params(model.trainable_weights[idx])
            already_counted.add(name)
    if isinstance(result, numpy.integer):
        result = result.item()
    assert isinstance(result, int)
    return result


def try_counting_params(model):
    count_trainable_params(model)
