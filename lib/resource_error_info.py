from keras import Model


def resource_error_info(e, model_or_layer):
    def locate_layer(e, model):
        results = []
        for idx, layer in enumerate(model.layers):
            if layer.name in str(e):
                if isinstance(layer, Model):
                    results += [f'{result} of layer {idx}/{len(model.layers)}'
                                for result in locate_layer(e, layer)]
                else:
                    results += [f'layer {idx}/{len(model.layers)}']
        return results

    model = model_or_layer
    print(f'The error occurred while allocating the following layers: {locate_layer(e, model)}')