def attr_dir(obj, include_methods=False, ignore=None):
    if ignore is None:
        ignore = []
    return {attr: obj.__getattr__(attr)
            for attr in dir(obj)
            if not attr.startswith('_') and (
                    include_methods or not callable(obj.__getattr__(attr))) and attr not in ignore}


def try_convert_to_number(image_attrs):
    for k in image_attrs:
        if image_attrs[k] == 'False':
            image_attrs[k] = False
            continue
        if image_attrs[k] == 'True':
            image_attrs[k] = True
            continue
        if isinstance(image_attrs[k], list):
            for idx in range(len(image_attrs[k])):
                try:
                    image_attrs[k][idx] = int(image_attrs[k][idx])
                except ValueError:
                    try:
                        image_attrs[k][idx] = float(image_attrs[k][idx])
                    except ValueError:
                        pass
        else:
            try:
                image_attrs[k] = int(image_attrs[k])
            except ValueError:
                try:
                    image_attrs[k] = float(image_attrs[k])
                except ValueError:
                    pass
