import functools
import itertools
import json
import math
import os
import re
import sqlite3
import sys
import threading
import time
import typing
from bisect import bisect_left
import datetime
from enum import Enum
from itertools import chain, combinations

# noinspection PyUnresolvedReferences
from unittest import mock

import SimpleITK

import sklearn.svm

import cachetools

from math import log, isnan, nan, floor, log10, gcd
from numbers import Number
from shutil import copyfile
from threading import RLock
from types import FunctionType
from typing import Union, Tuple, List, Optional, Dict, Type, Any

import numpy
import numpy as np
import pandas
import scipy.stats
import scipy.optimize
import tabulate
import hanging_threads
from scipy.ndimage import zoom

import matplotlib.pyplot as plt
import matplotlib.cm
import gc

from lib.my_logger import logging

X = Y = Z = float


class KnownIssue(Exception):
    """
    This means the code is not working and should not be used but still too valuable to be deleted
    """
    pass


def num_reasonable_decimals_by_confidence(confidence_interval_size: float):
    decimals = 0
    if confidence_interval_size < 0:
        raise ValueError(confidence_interval_size)
    if confidence_interval_size == 0:
        return 8
    while confidence_interval_size < 5:
        decimals += 1
        confidence_interval_size *= 10
    return decimals


def compare_sets_summary(a, b):
    if a == b:
        return 'equal'
    if a.issubset(b):
        return 'a subset of b, only in b: ' + str(b - a)
    if b.issubset(a):
        return 'b subset of a, only in a: ' + str(a - b)
    return 'not equal, only in a: ' + str(a - b) + ', only in b: ' + str(b - a)


def deterministic_hash(obj):
    """standard python hash changes when re-starting python, this one doesn't"""
    to_be_hashed = str(obj)
    import hashlib
    return int.from_bytes(
        hashlib.md5(to_be_hashed.encode("utf-8")).digest(),
        "big"
    ) % 2 ** 64


def deterministic_hash_str(obj) -> str:
    """standard python hash changes when re-starting python, this one doesn't"""
    return hex(deterministic_hash(obj))[2:].upper()


def powerset(iterable):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def in_debug_mode():
    """https://stackoverflow.com/a/38637774"""
    gettrace = getattr(sys, 'gettrace', None)

    if gettrace is None:
        return False
    elif gettrace() is None:
        return False
    elif gettrace():
        return True
    else:
        raise RuntimeError(gettrace())


def plot_with_conf(x, y_mean, y_conf, alpha=0.5, **kwargs):
    ax = kwargs.pop('ax', plt.gca())
    base_line, = ax.plot(x, y_mean, **kwargs)
    y_mean = np.array(y_mean)
    y_conf = np.array(y_conf)
    lb = y_mean - y_conf
    ub = y_mean + y_conf

    ax.fill_between(x, lb, ub, facecolor=base_line.get_color(), alpha=alpha)


def local_timezone():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(0))).astimezone().tzinfo


def print_attributes(obj, include_methods=False, ignore=None):
    if ignore is None:
        ignore = []
    for attr in dir(obj):
        if attr in ignore:
            continue
        if attr.startswith('_'):
            continue
        if not include_methods and callable(obj.__getattr__(attr)):
            continue
        print(attr, ':', obj.__getattr__(attr).__class__.__name__, ':', obj.__getattr__(attr))


def attr_dir(obj, include_methods=False, ignore=None):
    if ignore is None:
        ignore = []
    return {attr: obj.__getattr__(attr)
            for attr in dir(obj)
            if not attr.startswith('_') and (
                    include_methods or not callable(obj.__getattr__(attr))) and attr not in ignore}


def listdir_fullpath(d):
    """
    https://stackoverflow.com/a/120948
    """
    return sorted([os.path.join(d, f) for f in os.listdir(d)])


def zoom_to_shape(a: np.ndarray, shape: Tuple, mode: str = 'smooth', verbose=1):
    from keras import backend
    a = np.array(a, dtype=backend.floatx())  # also does a copy
    shape_dim = len(a.shape)
    if len(a.shape) != len(shape):
        raise ValueError('The shapes must have the same dimension but were len({0}) = {1} (original) '
                         'and len({2}) = {3} desired.'.format(a.shape, len(a.shape), shape, len(shape)))
    if len(shape) == 0:
        return a
    zoom_factors = tuple(shape[idx] / a.shape[idx] for idx in range(shape_dim))

    def _current_index_in_old_array():
        return tuple(slice(0, length) if axis != current_axis else slice(current_pixel_index, current_pixel_index + 1)
                     for axis, length in enumerate(a.shape))

    def _current_pixel_shape():
        return tuple(length if axis != current_axis else 1
                     for axis, length in enumerate(a.shape))

    def _current_result_index():
        return tuple(
            slice(0, length) if axis != current_axis else slice(pixel_index_in_result, pixel_index_in_result + 1)
            for axis, length in enumerate(a.shape))

    def _current_result_shape():
        return tuple(orig_length if axis != current_axis else shape[axis]
                     for axis, orig_length in enumerate(a.shape))

    if mode == 'constant':
        result = zoom(a, zoom_factors)
        assert result.shape == shape
        return result
    elif mode == 'smooth':
        result = a
        for current_axis, zoom_factor in sorted(enumerate(zoom_factors), key=lambda x: x[1]):
            result = np.zeros(_current_result_shape(), dtype=backend.floatx())
            # current_length = a.shape[current_axis]
            desired_length = shape[current_axis]
            current_pixel_index = 0
            current_pixel_part = 0  # how much of the current pixel is already read
            for pixel_index_in_result in range(desired_length):
                pixels_remaining = 1 / zoom_factor
                pixel_sum = np.zeros(_current_pixel_shape())
                while pixels_remaining + current_pixel_part > 1:
                    pixel_sum += (1 - current_pixel_part) * a[_current_index_in_old_array()]
                    current_pixel_index += 1
                    pixels_remaining -= (1 - current_pixel_part)
                    current_pixel_part = 0

                # the remaining pixel_part
                try:
                    pixel_sum += pixels_remaining * a[_current_index_in_old_array()]
                except (IndexError, ValueError):
                    if verbose:
                        print('WARNING: Skipping {0} pixels because of numerical imprecision.'.format(pixels_remaining))
                else:
                    current_pixel_part += pixels_remaining

                # insert to result
                pixel_sum *= zoom_factor

                result[_current_result_index()] = pixel_sum
            a = result

        assert result.shape == shape
        return result
    else:
        return NotImplementedError('Mode not available.')


def backup_file(filename):
    copyfile(filename, backup_file_path(filename))


def backup_file_path(filename):
    return filename + time.strftime("%Y%m%d") + '.bak'


# noinspection SpellCheckingInspection
def my_tabulate(data, tablefmt='pipe', **params):
    if data == [] and 'headers' in params:
        data = [(None for _ in params['headers'])]
    tabulate.MIN_PADDING = 0
    return tabulate.tabulate(data, tablefmt=tablefmt, **params)


def ce_loss(y_true, y_predicted):
    return -(y_true * log(y_predicted) + (1 - y_true) * log(1 - y_predicted))


class DontSaveResultsError(Exception):
    pass


def multinomial(n, bins):
    if bins == 0:
        if n > 0:
            raise ValueError('Cannot distribute to 0 bins.')
        return []
    remaining = n
    results = []
    for i in range(bins - 1):
        from numpy.random.mtrand import binomial
        x = binomial(remaining, 1 / (bins - i))
        results.append(x)
        remaining -= x

    results.append(remaining)
    return results


class UnknownTypeError(Exception):
    pass


# def shape_analysis(xs):
#     composed_dtypes = [list, tuple, np.ndarray, dict, set]
#     base_dtypes = [str, int, float, type, object]  # TODO add class and superclass of xs first element
#     all_dtypes = composed_dtypes + base_dtypes
#     if isinstance(xs, np.ndarray):
#         outer_brackets = ('[', ']')
#         shape = xs.shape
#         dtype = xs.dtype
#     elif isinstance(xs, tuple):
#         outer_brackets = ('(', ')')
#         shape = len(xs)
#         dtype = [t for t in all_dtypes if all(isinstance(x, t) for x in xs)][0]
#     elif isinstance(xs, list):
#         outer_brackets = ('[', ']')
#         shape = len(xs)
#         dtype = [t for t in all_dtypes if all(isinstance(x, t) for x in xs)][0]
#     elif isinstance(xs, dict) or isinstance(xs, set):
#         outer_brackets = ('{', '}')
#         shape = len(xs)
#         dtype = [t for t in all_dtypes if all(isinstance(x, t) for x in xs)][0]
#     elif any(isinstance(xs, t) for t in base_dtypes):
#         for t in base_dtypes:
#             if isinstance(xs, t):
#                 return str(t.__name__)
#         raise AssertionError('This should be unreachable.')
#     else:
#         raise UnknownTypeError('Unknown type:' + type(xs).__name__)
#
#     if shape and shape != '?':
#         return outer_brackets[0] + str(xs.shape) + ' * ' + str(dtype) + outer_brackets[1]
#     else:
#         return outer_brackets[0] + outer_brackets[1]


def beta_conf_interval_mle(data, conf=0.95):
    if len(data) <= 1:
        return 0, 1  # overestimates the interval
    if any(d < 0 or d > 1 or isnan(d) for d in data):
        return nan, nan
    if numpy.var(data) == 0:
        return numpy.mean(data), numpy.mean(data)
    epsilon = 1e-3
    # adjusted_data = data.copy()
    # for idx in range(len(adjusted_data)):
    #     adjusted_data[idx] *= (1 - 2 * epsilon)
    #     adjusted_data[idx] += epsilon
    alpha, beta, _, _ = scipy.stats.beta.fit(data, floc=-epsilon, fscale=1 + 2 * epsilon)

    lower, upper = scipy.stats.beta.interval(alpha=conf, a=alpha, b=beta)
    if lower < 0:
        lower = 0
    if upper < 0:
        upper = 0
    if lower > 1:
        lower = 1
    if upper > 1:
        upper = 1
    return lower, upper


def gamma_conf_interval_mle(data, conf=0.95) -> Tuple[float, float]:
    if len(data) == 0:
        return nan, nan
    if len(data) == 1:
        return nan, nan
    if any(d < 0 or isnan(d) for d in data):
        return nan, nan
    if numpy.var(data) == 0:
        return numpy.mean(data).item(), 0
    alpha, _, scale = scipy.stats.gamma.fit(data, floc=0)

    lower, upper = scipy.stats.gamma.interval(alpha=conf, a=alpha, scale=scale)
    if lower < 0:
        lower = 0
    if upper < 0:
        upper = 0
    return lower, upper


beta_quantile_cache = cachetools.LRUCache(maxsize=10)


@cachetools.cached(cache=beta_quantile_cache, key=lambda x1, p1, x2, p2, guess: (x1, x2, p1, p2))
def beta_parameters_quantiles(x1, p1, x2, p2, guess=(3, 3)):
    "Find parameters for a beta random variable X; so; that; P(X > x1) = p1 and P(X > x2) = p2.; "

    def square(x):
        return x * x

    def objective(v):
        (a, b) = v
        temp = square(scipy.stats.beta.cdf(x1, a, b) - p1)
        temp += square(scipy.stats.beta.cdf(x2, a, b) - p2)
        return temp

    xopt = scipy.optimize.fmin(objective, guess, disp=False)
    return (xopt[0], xopt[1])


def beta_conf_interval_quantile(data, conf=0.95, quantiles=(0.25, 0.75)):
    if len(data) <= 1:
        return 0, 1  # overestimates the interval
    mu = numpy.mean(data)
    v = numpy.var(data)
    data = numpy.array(data)
    if v == 0:
        return mu, mu
    lower = numpy.quantile(data, quantiles[0])
    upper = numpy.quantile(data, quantiles[1])

    alpha_guess = mu ** 2 * ((1 - mu) / v - 1 / mu)
    beta_guess = alpha_guess * (1 / mu - 1)

    alpha, beta = beta_parameters_quantiles(lower, quantiles[0], upper, quantiles[1], (alpha_guess, beta_guess))
    return scipy.stats.beta.interval(alpha=conf, a=alpha, b=beta)


def beta_stats_quantile(data, quantiles=(0.25, 0.75)):
    if len(data) < 1:
        return 0, 1  # overestimates the variance
    if len(data) == 1:
        return data[0], 1  # overestimates the variance
    data = numpy.array(data)
    mu = numpy.mean(data)
    v = numpy.var(data)
    if v == 0:
        return mu, 0
    lower = numpy.quantile(data, quantiles[0])
    upper = numpy.quantile(data, quantiles[1])

    alpha_guess = mu ** 2 * ((1 - mu) / v - 1 / mu)
    beta_guess = alpha_guess * (1 / mu - 1)

    alpha, beta = beta_parameters_quantiles(lower, quantiles[0], upper, quantiles[1], (alpha_guess, beta_guess))
    mean, variance = scipy.stats.beta.stats(a=alpha, b=beta)
    return mean, variance


def beta_stats_mle(data):
    if len(data) == 0:
        return nan, nan
    if len(data) == 1:
        return nan, nan
    if any(d < 0 or d > 1 or isnan(d) for d in data):
        return nan, nan
    if numpy.var(data) == 0:
        return numpy.mean(data), 0
    epsilon = 1e-4
    # adjusted_data = data.copy()
    # for idx in range(len(adjusted_data)):
    #     adjusted_data[idx] *= (1 - 2 * epsilon)
    #     adjusted_data[idx] += epsilon
    alpha, beta, _, _ = scipy.stats.beta.fit(data, floc=-epsilon, fscale=1 + 2 * epsilon)

    return scipy.stats.beta.stats(a=alpha, b=beta)


def gamma_stats_mle(data):
    if len(data) == 0:
        return nan, nan
    if len(data) == 1:
        return nan, nan
    if any(d < 0 or isnan(d) for d in data):
        return nan, nan
    if numpy.var(data) == 0:
        return numpy.mean(data), 0
    alpha, _, scale = scipy.stats.gamma.fit(data, floc=0)

    return scipy.stats.gamma.stats(a=alpha, scale=scale)


beta_stats = beta_stats_quantile
beta_conf_interval = beta_conf_interval_quantile
gamma_stats = gamma_stats_mle
gamma_conf_interval = gamma_conf_interval_mle


def split_df_list(df, target_column):
    """
    df = data frame to split,
    target_column = the column containing the values to split
    separator = the symbol used to perform the split
    returns: a data frame with each entry for the target column separated, with each element moved into a new row.
    The values in the other columns are duplicated across the newly divided rows.

    SOURCE: https://gist.github.com/jlln/338b4b0b55bd6984f883
    """

    def split_list_to_rows(row, row_accumulator):
        split_row = json.loads(row[target_column])
        for s in split_row:
            new_row = row.to_dict()
            new_row[target_column] = s
            row_accumulator.append(new_row)

    new_rows = []
    df.apply(split_list_to_rows, axis=1, args=(new_rows,))
    new_df = pandas.DataFrame(new_rows)
    return new_df


try:
    import winsound as win_sound


    def beep(*args, **kwargs):
        win_sound.Beep(*args, **kwargs)
except ImportError:
    win_sound = None


    def beep(*_args, **_kwargs):
        pass


def round_to_digits(x, d):
    if x == 0:
        return 0
    if isnan(x):
        return nan
    try:
        return round(x, d - 1 - int(floor(log10(abs(x)))))
    except OverflowError:
        return x


def gc_if_memory_error(f, *args, **kwargs):
    try:
        return f(*args, **kwargs)
    except MemoryError:
        print('Starting garbage collector')
        gc.collect()
        return f(*args, **kwargs)


def assert_not_empty(x):
    assert len(x)
    return x


def validation_steps(validation_dataset_size, maximum_batch_size):
    batch_size = gcd(validation_dataset_size, maximum_batch_size)
    steps = validation_dataset_size // batch_size
    assert batch_size * steps == validation_dataset_size
    return batch_size, steps


def functional_dependency_trigger(connection: sqlite3.Connection,
                                  table_name: str,
                                  determining_columns: List[str],
                                  determined_columns: List[str],
                                  exist_ok: bool, ):
    cursor = connection.cursor()
    # possible_performance_improvements
    determined_columns = [c for c in determined_columns if c not in determining_columns]
    trigger_base_name = '_'.join([table_name] + determining_columns + ['determine'] + determined_columns)

    error_message = ','.join(determining_columns) + ' must uniquely identify ' + ','.join(determined_columns)

    # when inserting check if there is already an entry with these values
    cursor.execute(f'''
    CREATE TRIGGER {'IF NOT EXISTS' if exist_ok else ''} {trigger_base_name}_after_insert
    BEFORE INSERT ON {table_name}
    WHEN EXISTS(SELECT * FROM {table_name}
         WHERE ({' AND '.join(f'NEW.{c} IS NOT NULL AND {c} = NEW.{c}' for c in determining_columns)})
         AND ({' OR '.join(f'{c} != NEW.{c}' for c in determined_columns)}))
    BEGIN SELECT RAISE(ROLLBACK, '{error_message}'); END
    ''')

    # when updating check if there is already an entry with these values (only if changed)
    cursor.execute(f'''
    CREATE TRIGGER {'IF NOT EXISTS' if exist_ok else ''} {trigger_base_name}_after_update
    BEFORE UPDATE ON {table_name}
    WHEN EXISTS(SELECT * FROM {table_name}
         WHERE ({' AND '.join(f'NEW.{c} IS NOT NULL AND {c} = NEW.{c}' for c in determining_columns)})
         AND ({' OR '.join(f'{c} != NEW.{c}' for c in determined_columns)}))
    BEGIN SELECT RAISE(ROLLBACK, '{error_message}'); END
    ''')


def heatmap_from_points(x, y,
                        x_lim: Optional[Union[int, Tuple[int, int]]] = None,
                        y_lim: Optional[Union[int, Tuple[int, int]]] = None,
                        gridsize=30):
    if isinstance(x_lim, Number):
        x_lim = (x_lim, x_lim)
    if isinstance(y_lim, Number):
        y_lim = (y_lim, y_lim)

    plt.hexbin(x, y, gridsize=gridsize, cmap=matplotlib.cm.jet, bins=None)
    if x_lim is not None:
        plt.xlim(x_lim)
    if y_lim is not None:
        plt.ylim(y_lim)

    cb = plt.colorbar()
    cb.set_label('mean value')


def strptime(date_string, fmt):
    return datetime.datetime(*(time.strptime(date_string, fmt)[0:6]))


class PrintLineRLock(RLock().__class__):
    def __init__(self, *args, name='', **kwargs):
        # noinspection PyArgumentList
        super().__init__(*args, **kwargs)
        self.name = name

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        print(f'Trying to acquire Lock {self.name}')
        result = RLock.acquire(self, blocking, timeout)
        print(f'Acquired Lock {self.name}')
        return result

    def release(self) -> None:
        print(f'Trying to release Lock {self.name}')
        # noinspection PyNoneFunctionAssignment
        result = RLock.release(self)
        print(f'Released Lock {self.name}')
        return result

    def __enter__(self, *args, **kwargs):
        print('Trying to enter Lock')
        # noinspection PyArgumentList
        super().__enter__(*args, **kwargs)
        print('Entered Lock')

    def __exit__(self, *args, **kwargs):
        print('Trying to exit Lock')
        super().__exit__(*args, **kwargs)
        print('Exited Lock')


def fixed_get_current_frames():
    """Return current threads prepared for
    further processing.
    """
    threads = {thread.ident: thread for thread in threading.enumerate()}
    return {
        thread_id: {
            'frame': hanging_threads.thread2list(frame),
            'time': None,
            'id': thread_id,
            'name': threads[thread_id].name,
            'object': threads[thread_id]
        } for thread_id, frame in sys._current_frames().items()
        if thread_id in threads  # otherwise keyerrors might happen because of race conditions
    }


hanging_threads.get_current_frames = fixed_get_current_frames


class CallCounter():
    def __init__(self, f):
        self.f = f
        self.calls = 0
        self.__name__ = f.__name__

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.f(*args, **kwargs)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__class__.__name__ + repr(self.__dict__)


def test_with_timeout(timeout=2):
    def wrapper(f):
        from lib.threading_timer_decorator import exit_after
        f = exit_after(timeout)(f)

        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            try:
                print(f'Running this test with timeout: {timeout}')
                return f(*args, **kwargs)
            except KeyboardInterrupt:
                raise AssertionError(f'Test took longer than {timeout} seconds')

        return wrapped

    return wrapper


def lru_cache_by_id(maxsize):
    return cachetools.cached(cachetools.LRUCache(maxsize=maxsize), key=id)


def iff_patch(patch: mock._patch):
    def decorator(f):
        def wrapped(*args, **kwargs):
            with patch:
                f(*args, **kwargs)
            try:
                f(*args, **kwargs)
            except:
                pass
            else:
                raise AssertionError('Test did not fail without patch')

        return wrapped

    return decorator


def iff_not_patch(patch: mock._patch):
    def decorator(f):
        def wrapped(*args, **kwargs):
            f(*args, **kwargs)
            try:
                with patch:
                    f(*args, **kwargs)
            except Exception as e:
                pass
            else:
                raise AssertionError('Test did not fail with patch')

        return wrapped

    return decorator


def required_size_for_safe_rotation(input_shape_px_after_rotation: Tuple[X, Y, Z],
                                    rotate_range_deg: float,
                                    ceil=True) -> Tuple[X, Y, Z]:
    if abs(rotate_range_deg) > 45:
        raise NotImplementedError
    base = input_shape_px_after_rotation
    if abs(rotate_range_deg) > 0:
        x_length = base[2] * math.sin(rotate_range_deg / 180 * math.pi) + base[1] * math.cos(
            rotate_range_deg / 180 * math.pi)
        y_length = base[2] * math.cos(rotate_range_deg / 180 * math.pi) + base[1] * math.sin(
            rotate_range_deg / 180 * math.pi)
        result = (base[0],
                  x_length,
                  y_length,)
    else:
        result = base
    if ceil:
        result = tuple(math.ceil(x) for x in result)
    return result


def required_size_for_safe_rotation_zyx(input_shape_px_after_rotation: Tuple[Z, Y, X],
                                        rotate_range_deg: float,
                                        ceil=True) -> Tuple[Z, Y, X]:
    return required_size_for_safe_rotation(input_shape_px_after_rotation[::-1],
                                           rotate_range_deg,
                                           ceil=ceil)[::-1]


def round_to_closest_value(x, values, assume_sorted=False):
    if not assume_sorted:
        values = sorted(values)
    next_largest = bisect_left(values, x)  # binary search
    if next_largest == 0:
        return values[0]
    if next_largest == len(values):
        return values[-1]
    next_smallest = next_largest - 1
    smaller = values[next_smallest]
    larger = values[next_largest]
    if abs(smaller - x) < abs(larger - x):
        return smaller
    else:
        return larger


def binary_search(a, x, lo=0, hi=None):
    hi = hi if hi is not None else len(a)  # hi defaults to len(a)

    pos = bisect_left(a, x, lo, hi)  # find insertion position

    return pos if pos != hi and a[pos] == x else -1  # don't walk off the end


def ceil_to_closest_value(x, values):
    values = sorted(values)
    next_largest = bisect_left(values, x)  # binary search
    if next_largest < len(values):
        return values[next_largest]
    else:
        return values[-1]  # if there is no larger value use the largest one


def print_progress_bar(iteration, total, prefix='Progress:', suffix='', decimals=1, length=50, fill='█',
                       print_eta=True):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:" + str(4 + decimals) + "." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    if getattr(print_progress_bar, 'last_printed_value', None) == (prefix, bar, percent, suffix):
        return
    print_progress_bar.last_printed_value = (prefix, bar, percent, suffix)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='')
    # Print New Line on Complete
    if iteration == total:
        print()


def get_all_subclasses(klass):
    all_subclasses = []

    for subclass in klass.__subclasses__():
        all_subclasses.append(subclass)
        all_subclasses.extend(get_all_subclasses(subclass))

    return all_subclasses


def latin1_json(data):
    return json.dumps(data, ensure_ascii=False).encode('latin-1')


def l2_norm(v1, v2):
    if len(v1) != len(v2):
        raise ValueError('Both vectors must be of the same size')
    return math.sqrt(sum([(x1 - x2) * (x1 - x2) for x1, x2 in zip(v1, v2)]))


def allow_additional_unused_keyword_arguments(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import inspect
        allowed_kwargs = [param.name for param in inspect.signature(func).parameters.values()]
        allowed_kwargs = {a: kwargs[a] for a in kwargs if a in allowed_kwargs}
        return func(*args, **allowed_kwargs)

    return wrapper


def copy_and_rename_method(func, new_name):
    funcdetails = [
        func.__code__,
        func.__globals__,
        func.__name__,
        func.__defaults__,
        func.__closure__
    ]
    old_name = func.__name__
    # copy
    # new_func = dill.loads(dill.dumps(func))
    new_func = FunctionType(*funcdetails)
    assert new_func is not funcdetails
    # rename
    new_func.__name__ = new_name
    assert func.__name__ is old_name
    return new_func


def rename(new_name):
    def decorator(f):
        f.__name__ = new_name
        return f

    return decorator


class LogicError(Exception):
    pass


def round_time(dt=None, precision=60):
    """Round a datetime object to any time lapse in seconds
    dt : datetime.datetime object, default now.
    roundTo : Closest number of seconds to round to, default 1 minute.
    Author: Thierry Husson 2012 - Use it as you want but don't blame me.
    """
    if dt is None:
        dt = datetime.datetime.now()
    if isinstance(precision, datetime.timedelta):
        precision = precision.total_seconds()
    seconds = (dt.replace(tzinfo=None) - dt.min).seconds
    rounding = (seconds + precision / 2) // precision * precision
    return dt + datetime.timedelta(seconds=rounding - seconds,
                                   microseconds=dt.microsecond)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def shorten_name(name):
    name = re.sub(r'\s+', r' ', str(name))
    name = name.replace(', ', ',')
    name = name.replace(', ', ',')
    name = name.replace(' ', '_')
    return re.sub(r'([A-Za-z])[a-z]*_?', r'\1', str(name))


def array_analysis(a: numpy.ndarray):
    print(f'  Shape: {a.shape}')
    mean = a.mean()
    print(f'  Mean: {mean}')
    print(f'  Std: {a.std()}')
    print(f'  Min, Max: {a.min(), a.max()}')
    print(f'  Mean absolute: {numpy.abs(a).mean()}')
    print(f'  Mean square: {numpy.square(a).mean()}')
    print(f'  Mean absolute difference from mean: {numpy.abs(a - mean).mean()}')
    print(f'  Mean squared difference from mean: {numpy.square(a - mean).mean()}')
    nonzero = numpy.count_nonzero(a)
    print(f'  Number of non-zeros: {nonzero}')
    print(f'  Number of zeros: {numpy.prod(a.shape) - nonzero}')
    if a.shape[-1] > 1 and a.shape[-1] <= 1000:
        # last dim is probably the number of classes
        print(f'  Class counts: {numpy.count_nonzero(a, axis=tuple(range(len(a.shape) - 1)))}')


def current_year_begin():
    return datetime.datetime(datetime.datetime.today().year, 1, 1).timestamp()


def current_day_begin():
    return datetime.datetime.today().timestamp() // (3600 * 24) * (3600 * 24)


def current_second_begin():
    return floor(datetime.datetime.today().timestamp())


def running_workers(executor):
    print(next(iter(executor._threads)).__dict__)
    return sum(1 for t in executor._threads
               if t == 1)


class Bunch(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__.update(kwargs)

    def add_method(self, m):
        setattr(self, m.__name__, functools.partial(m, self))


def queued_calls(executor):
    return len(executor._work_queue.queue)


def val_fold_by_test_fold(test_fold, num_folds):
    return (test_fold + 1) % num_folds


def remove_duplicates_using_identity(xs):
    return list({id(x): x for x in xs}.values())


def choose_threshold_using_svm(y_pred, y_true, C=1, sample_weights=None):
    if sample_weights is None:
        sample_weights = compute_sample_weights(y_true)
    assert numpy.count_nonzero(sample_weights) == sample_weights.size
    svm = sklearn.svm.LinearSVC(C=C)
    svm.fit(y_pred[..., numpy.newaxis], y_true, sample_weight=sample_weights)
    t = -svm.intercept_.item() / svm.coef_.item()
    assert y_pred.min() <= t <= y_pred.max(), t
    return t


def compute_sample_weights(y_true):
    if len(y_true.shape) != 1:
        raise ValueError
    sample_weights = numpy.zeros_like(y_true, dtype='float32')
    sample_weights[y_true.astype('bool')] = 0.5 / numpy.count_nonzero(y_true)
    sample_weights[numpy.logical_not(y_true)] = 0.5 / numpy.count_nonzero(numpy.logical_not(y_true))
    sample_weights *= sample_weights.size
    assert numpy.count_nonzero(sample_weights) == sample_weights.size
    return sample_weights


class JSONConvertible:
    SUBCLASSES_BY_NAME: Dict[str, Type['JSONConvertible']] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        JSONConvertible.SUBCLASSES_BY_NAME[cls.__name__] = cls

    @classmethod
    def subclasses(cls, strict=False):
        return [t for t in cls.SUBCLASSES_BY_NAME.values() if issubclass(t, cls) and (not strict or t != cls)]

    def to_json(self) -> Dict[str, Any]:
        raise NotImplementedError('Abstract method')

    @staticmethod
    def from_json(data: Dict[str, Any]):
        cls = JSONConvertible.SUBCLASSES_BY_NAME[data['type']]
        if cls.from_json is JSONConvertible.from_json:
            raise NotImplementedError(f'Abstract method from_json of class {cls.__name__}')
        return cls.from_json(data)

    @classmethod
    def schema(cls) -> Dict[str, Any]:
        fields = cls.field_types()
        result: Dict[str, Any] = {
            "type": "object",
            "properties": {'type': {'type': 'string', 'enum': [cls.__name__]}},
            'required': ['type'],
        }
        for field, field_type in fields.items():
            result['properties'][field] = schema_by_python_type(field_type)
            result['required'].append(field)

        subclasses = cls.subclasses(strict=True)
        if len(subclasses) > 0:
            result = {'anyOf': [result] + [{'$ref': f'#/components/schemas/{s.__name__}'} for s in subclasses]}
        return result

    @classmethod
    def schema_for_referencing(cls):
        return {'$ref': f'#/components/schemas/{cls.__name__}'}

    @classmethod
    def schema_python_types(cls) -> Dict[str, Type]:
        fields = typing.get_type_hints(cls.__init__)
        result: Dict[str, Type] = {}
        for field, field_type in fields.items():
            result[field] = field_type
        return result

    @classmethod
    def field_types(cls) -> Dict[str, Type]:
        return typing.get_type_hints(cls.__init__)


class EBC(JSONConvertible):
    def __eq__(self, other):
        return type(other) == type(self) and self.__dict__ == other.__dict__

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return f'{type(self).__name__}(**' + str(self.__dict__) + ')'

    def to_json(self) -> Dict[str, Any]:
        def obj_to_json(o):
            if isinstance(o, JSONConvertible):
                return o.to_json()
            elif isinstance(o, numpy.ndarray):
                return array_to_json(o)
            elif isinstance(o, list):
                return list_to_json(o)
            elif isinstance(o, dict):
                return dict_to_json(o)
            else:
                return o  # probably a primitive type

        def dict_to_json(d: Dict[str, Any]):
            return {k: obj_to_json(v) for k, v in d.items()}

        def list_to_json(l: list):
            return [obj_to_json(v) for v in l]

        def array_to_json(o: numpy.ndarray):
            return o.tolist()

        result: Dict[str, Any] = {
            'type': type(self).__name__,
            **self.__dict__,
        }
        for k in result:
            result[k] = obj_to_json(result[k])
        return result

    @staticmethod
    def from_json(data: Dict[str, Any]):
        cls = EBC.SUBCLASSES_BY_NAME[data['type']]
        return class_from_json(cls, data)


class _Container(EBC):
    def __init__(self, item):
        self.item = item


def container_to_json(obj: Union[list, dict, numpy.ndarray]) -> Dict[str, Any]:
    # Useful when converting list or dicts of json-convertible objects to json (recursively)
    e = _Container(obj)
    return e.to_json()['item']


def schema_by_python_type(t: Union[Type, dict]):
    if t == 'TODO':
        return t
    if isinstance(t, dict):
        return {
            "type": "object",
            "properties": {n: schema_by_python_type(t2) for n, t2 in t.items()},
            'required': [n for n in t],
        }
    elif t is int:
        return {'type': 'integer'}
    elif t is Any:
        return {}
    elif t is float:
        return {'type': 'number'}
    elif t is str:
        return {'type': 'string'}
    elif t is bool:
        return {'type': 'boolean'}
    elif t is type(None) or t is None:
        return {'type': 'null'}
    elif hasattr(t, '__origin__'):  # probably a generic typing-type
        if t.__origin__ is list:
            return {'type': 'array', 'items': schema_by_python_type(t.__args__[0])}
        elif t.__origin__ is dict:
            return {'type': 'object', 'additionalProperties': schema_by_python_type(t.__args__[1])}
        elif t.__origin__ is tuple and len(set(t.__args__)) == 1:
            return {'type': 'array', 'items': schema_by_python_type(t.__args__[0]), 'minItems': len(t.__args__), 'maxItems': len(t.__args__)}
        elif t.__origin__ is Union:
            return {'anyOf': [schema_by_python_type(t2) for t2 in t.__args__]}
        else:
            raise ValueError(f'Unknown type {t}')
    elif issubclass(t, list):
        return {'type': 'array'}
    elif issubclass(t, type) and len(set(t.__args__)) == 1:
        return {'type': 'array', 'items': schema_by_python_type(t.__args__[0]), 'minItems': len(t.__args__), 'maxItems': len(t.__args__)}
    elif t is datetime.datetime:
        from lib.date_time import SerializableDateTime
        return schema_by_python_type(SerializableDateTime)
    elif hasattr(t, 'schema_for_referencing'):
        return t.schema_for_referencing()
    else:
        raise ValueError(f'Unknown type {t}')


def class_from_json(cls, data: Dict[str, Any]):
    if isinstance(data, str):
        data = json.loads(data)
    # noinspection PyArgumentList
    try:
        return cls(**data)
    except TypeError as e:
        if "__init__() got an unexpected keyword argument 'type'" in str(e):
            # probably this was from a to_json method
            if data['type'] != cls.__name__:
                t = data['type']
                logging.warning(f'Reconstructing a {cls.__name__} from a dict with type={t}')
            data = data.copy()
            del data['type']
            for k, v in data.items():
                if probably_serialized_from_json_convertible(v):
                    data[k] = JSONConvertible.SUBCLASSES_BY_NAME[v['type']].from_json(v)
                elif isinstance(v, list):
                    data[k] = [JSONConvertible.SUBCLASSES_BY_NAME[x['type']].from_json(x)
                               if probably_serialized_from_json_convertible(x)
                               else x
                               for x in v]
                elif isinstance(v, dict):
                    data[k] = {
                        k2: JSONConvertible.SUBCLASSES_BY_NAME[x['type']].from_json(x)
                        if probably_serialized_from_json_convertible(x)
                        else x
                        for k2, x in v.items()
                    }
            try:
                return cls(**data)
            except TypeError:
                return allow_additional_unused_keyword_arguments(cls)(**data)


def probably_serialized_from_json_convertible(data):
    return isinstance(data, dict) and 'type' in data and data['type'] in JSONConvertible.SUBCLASSES_BY_NAME


class EBE(Enum):
    def __int__(self):
        return self.value

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __lt__(self, other):
        return list(type(self)).index(self) < list(type(self)).index(other)

    @classmethod
    def from_name(cls, variable_name):
        return cls.__dict__[variable_name]


def underscore(name):
    """
    source: https://stackoverflow.com/a/1176023
    """
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()


def set_additional_keys(obj, json_data):
    additional_keys = {k for k in json_data if underscore(k) not in obj.__dict__}
    if len(additional_keys) > 0:
        print(f'{type(obj).__name__}: Storing additional keys:', additional_keys)
    obj.__dict__.update({underscore(k): json_data[k] for k in additional_keys})


def convert_pvalue_to_asterisks(pvalues):
    result = numpy.full_like(pvalues, ' ', dtype=object)
    result[pvalues <= 0.1] = '.'
    result[pvalues <= 0.05] = '*'
    result[pvalues <= 0.01] = '**'
    result[pvalues <= 0.001] = '***'
    result[pvalues <= 0.0001] = '****'
    return result


def all_sets_and_disjoint(xss):
    for xs in xss:
        if len(xs) != len(set(xs)):
            return False
    total = list(itertools.chain.from_iterable(xss))
    return len(total) == len(set(total))


def load_size_and_spacing(nifti_path):
    r = SimpleITK.ImageFileReader()
    r.SetFileName(nifti_path)
    r.LoadPrivateTagsOn()
    r.ReadImageInformation()
    return r.GetSize(), r.GetSpacing()


def dicom_metadata_from_image(image):
    return {
        k: str(image.GetMetaData(k)).encode('utf-8', errors='replace').decode('utf-8')
        for k in image.GetMetaDataKeys()
    }
