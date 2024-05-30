from keras import Model


def split(model, start, stop, custom_objects=None):
    configs = model.get_config()
    kept_layers = set()
    for i, l in enumerate(configs['layers']):
        if i == 0:
            configs['layers'][0]['config']['batch_input_shape'] = model.layers[start].input_shape
            if i != start:
                configs['layers'][0]['config']['name'] = configs['layers'][0]['name']
        elif i < start or i >= stop:
            continue
        kept_layers.add(l['name'])
    # filter layers
    layers = [l for l in configs['layers'] if l['name'] in kept_layers]
    layers[1]['inbound_nodes'][0][0][0] = layers[0]['name']
    # set conf
    configs['layers'] = layers
    configs['input_layers'][0][0] = layers[0]['name']
    configs['output_layers'][0][0] = layers[-1]['name']
    # create new model
    new_model = Model.from_config(configs, custom_objects=custom_objects)
    for l in new_model.layers:
        orig_l = model.get_layer(l.name)
        if orig_l is not None:
            l.set_weights(orig_l.get_weights())
    return new_model
