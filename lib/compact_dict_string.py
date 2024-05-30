def compact_object_string(o, max_line_length=120):
    if isinstance(o, list):
        return compact_list_string(o, max_line_length)
    elif isinstance(o, tuple):
        return compact_tuple_string(o, max_line_length)
    elif isinstance(o, dict):
        return compact_dict_string(o, max_line_length)
    else:
        return str(o)


def compact_list_string(xs: list, max_line_length=120):
    # try to fit everything in one line with the default str method
    single_line_result = str(xs)
    if len(single_line_result) <= max_line_length:
        return single_line_result
    # try to fit everything in one line
    single_item_strings = []
    for v in xs:
        single_item_strings.append(str(v))
    single_line_result = '{' + ', '.join(single_item_strings) + '}'
    if len(single_line_result) <= max_line_length:
        return single_line_result

    # put compact value string next to the key strings
    single_item_strings = []
    for v in xs:
        prefix = ''
        without_indent: str = prefix + f'{compact_object_string(v, max_line_length=max_line_length - len(prefix))}'
        single_item_strings.append(('\n' + ' ' * (len(prefix) + 1)).join([line for line in without_indent.splitlines()]))
    return '[' + ',\n '.join(single_item_strings) + ']'

def compact_tuple_string(xs: tuple, max_line_length=120):
    # try to fit everything in one line with the default str method
    single_line_result = str(xs)
    if len(single_line_result) <= max_line_length:
        return single_line_result
    # try to fit everything in one line
    single_item_strings = []
    for v in xs:
        single_item_strings.append(str(v))
    single_line_result = '{' + ', '.join(single_item_strings) + '}'
    if len(single_line_result) <= max_line_length:
        return single_line_result

    # put compact value string next to the key strings
    single_item_strings = []
    for v in xs:
        prefix = ''
        without_indent: str = prefix + f'{compact_object_string(v, max_line_length=max_line_length - len(prefix))}'
        single_item_strings.append(('\n' + ' ' * (len(prefix) + 1)).join([line for line in without_indent.splitlines()]))
    return '(' + ',\n '.join(single_item_strings) + ')'


def compact_dict_string(d: dict, max_line_length=120):
    # try to fit everything in one line with the default str method
    single_line_result = str(d)
    if len(single_line_result) <= max_line_length:
        return single_line_result

    # try to fit everything in one line
    single_item_strings = []
    for k, v in d.items():
        single_item_strings.append(f'{k}: {v}')
    single_line_result = '{' + ', '.join(single_item_strings) + '}'
    if len(single_line_result) <= max_line_length:
        return single_line_result

    # try to put compact value string next to the key strings
    single_item_strings = []
    for k, v in d.items():
        prefix = str(k) + ': '
        without_indent: str = prefix + f'{compact_object_string(v, max_line_length=max_line_length - len(prefix))}'
        single_item_strings.append(('\n' + ' ' * (len(prefix) + 1)).join([line for line in without_indent.splitlines()]))
    multi_line_result_right = '{' + ',\n '.join(single_item_strings) + '}'

    # try to put compact value string below key strings
    single_item_strings = []
    for k, v in d.items():
        prefix = '  '
        without_indent: str = prefix + f'{compact_object_string(v, max_line_length=max_line_length - len(prefix))}'
        single_item_strings.append(str(k) + ':\n' + ('\n' + ' ' * (len(prefix) + 1)).join([line for line in without_indent.splitlines()]))
    multi_line_result_below = '{' + ',\n '.join(single_item_strings) + '\n}'

    if len(multi_line_result_right.splitlines()) < len(multi_line_result_below.splitlines()):
        return multi_line_result_right
    else:
        return multi_line_result_below

