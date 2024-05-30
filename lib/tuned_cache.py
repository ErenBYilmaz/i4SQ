import functools
import random
import sys
from copy import deepcopy

print('INFO: Loaded tuned variant of joblib cache.')

# noinspection PyProtectedMember,PyPep8
import joblib
# noinspection PyProtectedMember,PyPep8
from joblib.func_inspect import _clean_win_chars
# noinspection PyProtectedMember,PyPep8
from joblib.memory import MemorizedFunc, _FUNCTION_HASHES, NotMemorizedFunc, Memory

_FUNC_NAMES = {}


# noinspection SpellCheckingInspection
class TunedMemory(Memory):
    def cache(self, func=None, ignore=None, verbose=None, mmap_mode=False, identifier_cache_maxsize=0):
        """ Decorates the given function func to only compute its return
            value for input arguments not cached on disk.

            Parameters
            ----------
            func: callable, optional
                The function to be decorated
            ignore: list of strings
                A list of arguments name to ignore in the hashing
            verbose: integer, optional
                The verbosity mode of the function. By default that
                of the memory object is used.
            mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
                The memmapping mode used when loading from cache
                numpy arrays. See numpy.load for the meaning of the
                arguments. By default that of the memory object is used.

            Returns
            -------
            decorated_func: MemorizedFunc object
                The returned object is a MemorizedFunc object, that is
                callable (behaves like a function), but offers extra
                methods for cache lookup and management. See the
                documentation for :class:`joblib.memory.MemorizedFunc`.
        """
        if func is None:
            # Partial application, to be able to specify extra keyword
            # arguments in decorators
            return functools.partial(self.cache, ignore=ignore,
                                     verbose=verbose, mmap_mode=mmap_mode)
        if self.store_backend is None:
            return NotMemorizedFunc(func)
        if verbose is None:
            verbose = self._verbose
        if mmap_mode is False:
            mmap_mode = self.mmap_mode
        if isinstance(func, TunedMemorizedFunc):
            func = func.func
        return TunedMemorizedFunc(func,
                                  location=self.store_backend,
                                  backend=self.backend,
                                  ignore=ignore,
                                  mmap_mode=mmap_mode,
                                  compress=self.compress,
                                  verbose=verbose,
                                  timestamp=self.timestamp)


class TunedMemorizedFunc(MemorizedFunc):
    identifier_cache = {}
    identifier_cache_requests = 0
    identifier_cache_misses = 0

    def __call__(self, *args, **kwargs):
        # Also store in the in-memory store of function hashes
        if self.func not in _FUNCTION_HASHES:
            is_named_callable = (hasattr(self.func, '__name__') and
                                 self.func.__name__ != '<lambda>')
            if is_named_callable:
                # Don't do this for lambda functions or strange callable
                # objects, as it ends up being too fragile
                func_hash = self._hash_func()
                try:
                    _FUNCTION_HASHES[self.func] = func_hash
                except TypeError:
                    # Some callable are not hashable
                    pass

        # return same result as before
        return MemorizedFunc.__call__(self, *args, **kwargs)

    def _get_output_identifiers(self, *args, **kwargs):
        TunedMemorizedFunc.identifier_cache_requests += 1
        cache = TunedMemorizedFunc.identifier_cache
        identifier_cache_key = None
        try:
            identifier_cache_key = (self.func, *args, frozenset(kwargs.items()))
            return cache[identifier_cache_key]
        except TypeError as e:
            if 'unhashable' in str(e):
                # return same result as before
                return MemorizedFunc._get_output_identifiers(self, *args, **kwargs)
            else:
                raise
        except KeyError:
            TunedMemorizedFunc.identifier_cache_misses += 1
            if len(cache) > 10000:  # keep cache small
                requests = TunedMemorizedFunc.identifier_cache_requests
                misses = TunedMemorizedFunc.identifier_cache_misses
                print(f'Clearing randomly half of the cache, {requests - misses} hits total, {misses} misses total')
                for k in random.choices(list(cache.keys()), k=len(cache) // 2):
                    if k not in cache:
                        continue
                    del cache[k]
            assert identifier_cache_key is not None
            cache[identifier_cache_key] = MemorizedFunc._get_output_identifiers(self, *args, **kwargs)
            return cache[identifier_cache_key]


old_get_func_name = joblib.func_inspect.get_func_name


def tuned_get_func_name(func, resolv_alias=True, win_characters=True):
    if (func, resolv_alias, win_characters) not in _FUNC_NAMES:
        _FUNC_NAMES[(func, resolv_alias, win_characters)] = old_get_func_name(func, resolv_alias, win_characters)

        if len(_FUNC_NAMES) > 1000:
            # keep cache small and fast
            for idx, k in enumerate(_FUNC_NAMES.keys()):
                if idx % 2:
                    del _FUNC_NAMES[k]
        # print('cache size ', len(_FUNC_NAMES))

    return deepcopy(_FUNC_NAMES[(func, resolv_alias, win_characters)])


joblib.func_inspect.get_func_name = tuned_get_func_name
joblib.memory.get_func_name = tuned_get_func_name
