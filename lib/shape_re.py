import re
from typing import Tuple, Union, Optional

import numpy

MinElements = int
MaxElements = int
RegEx = str
ellipsis = type(Ellipsis)
ShapeReEntry = Optional[Union[int, ellipsis, any, Tuple[MinElements, MaxElements], RegEx]]


def shape_to_re(shape: Tuple[ShapeReEntry, ...]) -> str:
    def s_to_re(s: ShapeReEntry):
        any_entry = r'(?:\d+|None)'
        if s is None:
            return str(None)
        elif isinstance(s, int) or isinstance(s, numpy.integer):
            return str(s)
        elif isinstance(s, ellipsis):
            return any_entry + r'(?:, ' + any_entry + r')*'
        elif s is any:
            return any_entry
        elif s is int:
            return r'\d+'
        elif isinstance(s, tuple) and len(s) == 2 and isinstance(s[0], int) and isinstance(s[1], int):
            adj_lower_bound = str(max(s[0] - 1, 0))
            adj_upper_bound = str(max(s[1] - 1, 0))
            return any_entry + r'(?:, ' + any_entry + r'){' + adj_lower_bound + r',' + adj_upper_bound + r'}'  # tODO test and fix
        elif isinstance(s, RegEx):
            return s
        else:
            raise ValueError(s)

    def empty_allowed(s: ShapeReEntry) -> bool:
        if s is None:
            return False
        elif isinstance(s, int) or isinstance(s, numpy.integer):
            return False
        elif isinstance(s, ellipsis):
            return True
        elif s is any:
            return False
        elif s is int:
            return False
        elif isinstance(s, tuple) and len(s) == 2 and isinstance(s[0], int) and isinstance(s[1], int):
            return s[0] <= 0
        elif isinstance(s, RegEx):
            return False
        else:
            raise ValueError

    res = [s_to_re(s) for s in shape]
    for idx in range(1, len(res))[::-1]:
        if empty_allowed(shape[idx]):
            res[idx] = r'(?:, ' + res[idx] + r')?'  # add comma at the beginning and make the whole thing optional
        else:
            res[idx] = r', ' + res[idx]  # add comma at the beginning
    if len(shape) >= 2:
        if empty_allowed(shape[0]) and not empty_allowed(shape[1]):
            res[1] = res[1][2:]  # remove comma and whitespace
            res[0] = r'(?:' + res[0] + r', )?'  # add comma at the end
    if len(shape) == 1 and empty_allowed(shape[0]):
        res[0] = r'(?:' + res[0] + r')?'  # add comma at the end
    body = ''.join(res)

    return r'\(' + body + r',?\)'


def fullmatch(shape_pattern: Tuple[ShapeReEntry, ...], shape: Tuple[Optional[int], ...]):
    return re.fullmatch(shape_to_re(shape_pattern), str(tuple(shape)))


def none_to_any(shape: Tuple[Optional[int], ...]) -> Tuple[Optional[int], ...]:
    return tuple(x if x is not None else any
                 for x in shape)
