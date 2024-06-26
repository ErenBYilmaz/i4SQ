import inspect
import os
import re
import sys
import traceback
from itertools import islice
from pickle import PickleError
from typing import Sized, Dict, Tuple, Type

from types import FrameType
from unittest.mock import patch

from lib.threading_timer_decorator import exit_after

try:
    import numpy
except ImportError:
    numpy = None
try:
    import torch
except ImportError:
    torch = None

FORMATTING_OPTIONS = {
    'MAX_LINE_LENGTH': 1024,
    'SHORT_LINE_THRESHOLD': 128,
    'MAX_NEWLINES': 20,
}
ID = int


# noinspection PyPep8Naming
def name_or_str(X):
    try:
        return re.search(r"<class '?(.*?)'?>", str(X))[1]
    except TypeError:  # if not found
        return str(X)


@exit_after(2)
def _type_string_with_timeout(x):
    if numpy is not None and isinstance(x, numpy.ndarray):
        return name_or_str(type(x)) + str(x.shape) + f' {x.dtype}'
    elif torch is not None and isinstance(x, torch.Tensor):
        dtype = str(x.dtype).replace("torch.", "")
        return name_or_str(type(x)) + str(tuple(x.shape)) + f' {dtype}'
    elif isinstance(x, Sized):
        return name_or_str(type(x)) + f'({len(x)})'
    else:
        return name_or_str(type(x))


def type_string(x):
    try:
        type_as_string = _type_string_with_timeout(x)
    except KeyboardInterrupt:
        type_as_string = "<TIMEOUT WHILE PRINTING TYPE>"
    except Exception as e:
        # noinspection PyBroadException
        try:
            type_as_string = f"<{type(e).__name__} WHILE PRINTING TYPE>"
        except Exception:
            type_as_string = "<ERROR WHILE PRINTING TYPE>"
    return type_as_string


@exit_after(2)
def _to_string_with_timeout(x):
    return str(x)


def value_to_string(value):
    try:
        value_string: str = _to_string_with_timeout(value)
        value_string = value_string.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t')
    except KeyboardInterrupt:
        value_string = "<TIMEOUT WHILE PRINTING VALUE>"
    except Exception:
        value_string = "<ERROR WHILE PRINTING VALUE>"
    return value_string


def nth_index(iterable, value, n):
    matches = (idx for idx, val in enumerate(iterable) if val == value)
    return next(islice(matches, n - 1, n), None)


class DumpingException(Exception):
    pass


class SubclassNotFound(ImportError):
    pass


def subclass_by_name(name: str, base: Type):
    candidates = [t for t in base.__subclasses__() if t.__name__ == name]
    if len(candidates) != 1:
        raise SubclassNotFound()
    return candidates[0]


def dont_import():
    raise ImportError


loaded_custom_picklers = False


def fixed_dill_is_builtin_module(module):
    # copied and from dill._dill
    # the only modification is the check if __file__ is None

    if not hasattr(module, "__file__"): return True
    if module.__file__ is None: return False
    # If a module file name starts with prefix, it should be a builtin
    # module, so should always be pickled as a reference.
    names = ["base_prefix", "base_exec_prefix", "exec_prefix", "prefix", "real_prefix"]
    from dill._dill import EXTENSION_SUFFIXES
    return any(os.path.realpath(module.__file__).startswith(os.path.realpath(getattr(sys, name)))
               for name in names if hasattr(sys, name)) or \
        module.__file__.endswith(EXTENSION_SUFFIXES) or \
        'site-packages' in module.__file__


def load_custom_picklers():
    global loaded_custom_picklers
    if loaded_custom_picklers:
        return
    print('Loading custom picklers...')

    typing_types = ['Dict', 'List', 'Set', 'Tuple', 'Callable', 'Optional']
    for unpicklable_type in [
        'from zmq import Socket as unpicklable',
        'from zmq import Context as unpicklable',

        'from sqlite3 import Connection as unpicklable',
        'from sqlite3 import Cursor as unpicklable',

        'from socket import socket as unpicklable',

        'from tensorflow import Tensor as unpicklable',
        'from tensorflow.python.types.core import Tensor as unpicklable',
        # 'from tensorflow.keras import Model as unpicklable',
        'from tensorflow.python.eager.def_function import Function as unpicklable',
        'from tensorflow.python.keras.utils.object_identity import _ObjectIdentityWrapper as unpicklable',
        # Next line: pybind11_builtins.pybind11_object
        'from tensorflow.python._pywrap_tfe import TFE_MonitoringBoolGauge0;unpicklable=TFE_MonitoringBoolGauge0.__base__',

        'unpicklable = subclass_by_name(\\"PyCapsule\\", object)',
        'unpicklable = subclass_by_name(\\"PyHANDLE\\", object)',
        'unpicklable = subclass_by_name(\\"_Final\\", object)',

        'from sys import version_info;unpicklable = type(version_info)',

        'from _json import make_encoder as unpicklable',
        'from _json import make_scanner as unpicklable',

        'from h5py import HLObject as unpicklable',
        'from itk import cvar; unpicklable = type(cvar)',
        'from itk import cin; unpicklable = type(cin)',
        'from itk import cout; unpicklable = type(cout)',

        'from builtins import memoryview as unpicklable',

        'from matplotlib.backends.backend_qt5agg import FigureCanvasQT as unpicklable',

        'from numpy import flatiter as unpicklable',

        # can't pickle type annotations from typing in python <= 3.6 (Next line: typing._TypingBase)
        'import sys;dont_import() if sys.version >=\\"3.7\\" else None;from typing import Optional;unpicklable = type(Optional).__base__.__base__',
        *[f'import sys;dont_import() if sys.version >=\\"3.7\\" else None;from typing import {t};unpicklable = type({t})' for t in typing_types],
        'import sys;dont_import() if sys.version >=\\"3.7\\" else None;from typing import Dict;unpicklable = type(Dict).__base__',

        'import inspect;unpicklable = type(inspect.stack()[0].frame)',

        # can't pickle thread-local data
        'from threading import local as unpicklable',
        'from _thread import _local as unpicklable',

        # can't pickle generator objects
        'unpicklable = type(_ for _ in [])',
    ]:
        try_register_unpicklable_from_string(unpicklable_type)
    fix_itk_builtin_pickling()
    fix_dill_builtin_module()

    loaded_custom_picklers = True


def try_register_unpicklable_from_string(unpicklable_type_str):
    try:
        unpicklable = eval(f'exec("{unpicklable_type_str}") or unpicklable')
    except ImportError:
        pass
    else:
        register_unpicklable(unpicklable, also_subclasses=True)
    finally:
        unpicklable = None


def fix_dill_builtin_module():
    try:
        patch('dill._dill._is_builtin_module', fixed_dill_is_builtin_module).__enter__()
    except AttributeError:
        pass


def fix_itk_builtin_pickling():
    try:
        import itk
        for itk_submodule_name in itk.__dict__:
            itk_submodule = itk.__dict__[itk_submodule_name]
            if type(itk_submodule).__name__ != 'module':
                continue
            for k, v in itk_submodule.__dict__.items():
                if type(v).__name__ == 'builtin_function_or_method':
                    if not v.__module__.startswith('itk.'):
                        v.__module__ = 'itk.' + v.__module__
    except ImportError:
        pass


def register_unpicklable(unpicklable: Type, also_subclasses=False):
    import dill
    @dill.register(unpicklable)
    def save_unpicklable(pickler, obj):
        def recreate_unpicklable():
            return f'This was something that could not be pickled and instead was replaced with this string'

        recreate_unpicklable.__name__ = f'unpicklable_{unpicklable.__name__}'
        pickler.save_reduce(recreate_unpicklable, (), obj=obj)

    if also_subclasses:
        if not hasattr(unpicklable, '__subclasses__'):
            subclasses = []
        elif unpicklable.__subclasses__ is type.__subclasses__:
            subclasses = []
        else:
            subclasses = unpicklable.__subclasses__()
        for subclass in subclasses:
            register_unpicklable(subclass, also_subclasses=True)


def dump_stack_to_file(serialize_to, print=print, stack=None):
    if stack is None:
        stack = inspect.stack()[1:]
    try:
        import dill
    except ModuleNotFoundError:
        print('Dill not available. Not dumping stack.')
    else:
        load_custom_picklers()

        serializable_stack = []
        for frame in stack:
            if isinstance(frame, inspect.FrameInfo):
                frame = frame.frame
            serializable_stack.append({
                k: frame.__getattribute__(k)
                for k in ['f_globals', 'f_locals', 'f_code', 'f_lineno', 'f_lasti']
            })

        with open(serialize_to, 'wb') as serialize_to_file:
            try:
                dill.dump(serializable_stack, serialize_to_file)
            except (PickleError, RecursionError) as e:
                print(f'Was not able to dump the stack. Error {type(e)}: {e}')
                unpicklable_frames = []
                for frame_idx in range(len(serializable_stack)):
                    try:
                        dill.dumps(serializable_stack[frame_idx])
                    except (PickleError, RecursionError):
                        unpicklable_frames.append(frame_idx)
                print(f'Unpicklable frames (top=0, bottom={len(serializable_stack)}): {unpicklable_frames}')

                if 'typing.' in str(e):
                    print('This might be fixed by upgrading to python 3.7 or above.')
            else:
                print(f'Dumped stack. Can be loaded with:')
                print(f'with open(r"{os.path.abspath(serialize_to)}", "rb") as f: import dill; dill._dill._reverse_typemap["ClassType"] = type; stack = dill.load(f)')
        if os.path.isfile(serialize_to):
            from lib.util import backup_file
            backup_file(serialize_to)


def print_exc_plus(print=print, serialize_to=None, print_trace=True):
    """
    Print the usual traceback information, followed by a listing of all the
    local variables in each frame.
    """
    limit = FORMATTING_OPTIONS['MAX_LINE_LENGTH']
    max_newlines = FORMATTING_OPTIONS['MAX_NEWLINES']
    tb = sys.exc_info()[2]
    if numpy is not None:
        options = numpy.get_printoptions()
        numpy.set_printoptions(precision=3, edgeitems=2, floatmode='maxprec', threshold=20, linewidth=120)
    else:
        options = {}
    stack = []
    long_printed_objs: Dict[ID, Tuple[str, FrameType]] = {}

    while tb:
        stack.append(tb.tb_frame)
        tb = tb.tb_next
    if print_trace:
        for frame in stack:
            if frame is not stack[0]:
                print('-' * 40)

            print("  File \"%s\", line %s, in %s" % (frame.f_code.co_filename,
                                                     frame.f_lineno,
                                                     frame.f_code.co_name))
            for key, value in frame.f_locals.items():
                # We have to be careful not to cause a new error in our error
                # printer! Calling str() on an unknown object could cause an
                # error we don't want.

                # noinspection PyBroadException
                try:
                    key_string = _to_string_with_timeout(key)
                except KeyboardInterrupt:
                    key_string = "<TIMEOUT WHILE PRINTING KEY>"
                except Exception:
                    key_string = "<ERROR WHILE PRINTING KEY>"

                # noinspection PyBroadException
                type_as_string = type_string(value)

                if id(value) in long_printed_objs:
                    prev_key_string, prev_frame = long_printed_objs[id(value)]
                    if prev_frame is frame:
                        print("\t%s is the same as '%s'" %
                              (key_string + ' : ' + type_as_string,
                               prev_key_string))
                    else:
                        try:
                            relpath = os.path.relpath(prev_frame.f_code.co_filename)
                        except ValueError:
                            relpath = prev_frame.f_code.co_filename
                        print("\t%s is the same as '%s' in frame %s in %s at line %s." %
                              (key_string + ' : ' + type_as_string,
                               prev_key_string,
                               prev_frame.f_code.co_name,
                               relpath,
                               prev_frame.f_lineno))
                    continue

                # noinspection PyBroadException
                value_string = value_to_string(value)
                line: str = '\t' + key_string + ' : ' + type_as_string + ' = ' + value_string
                if limit is not None and len(line) > limit:
                    line = line[:limit - 1] + '...'
                if max_newlines is not None and line.count('\n') > max_newlines:
                    line = line[:nth_index(line, '\n', max_newlines)].strip() + '... (' + str(
                        line[nth_index(line, '\n', max_newlines):].count('\n')) + ' more lines)'
                if len(line) > FORMATTING_OPTIONS['SHORT_LINE_THRESHOLD']:
                    long_printed_objs[id(value)] = key_string, frame
                print(line)

        traceback.print_exc(limit=0)

        # etype, value, tb = sys.exc_info()
        # for line in traceback.TracebackException(type(value), value, tb, limit=limit).format(chain=True):
        #     print(line)

    if serialize_to is not None:
        dump_stack_to_file(stack=stack, serialize_to=serialize_to, print=print)

    if numpy is not None:
        numpy.set_printoptions(**options)
