import json
import numbers
import pydoc
import types

from collections import OrderedDict
from collections.abc import MutableMapping, MutableSequence
from io import IOBase
from natsort import natsorted
from pathlib import Path
from typing import Any, Callable, IO, List, Optional, Union


class DataWrapper:
    """A abstract wrapper around some form of data."""

    def __init__(self, wrapped_data=None):
        self._wrapped_data = wrapped_data

        #: Either `None` or the path where the data is located on disk, i.e.,
        #: where it recently has been :meth:`load` ed from or :meth:`dump` ed
        #: to.
        self.data_path = None

    @property
    def wrapped_data(self):
        """The original wrapped data."""
        return self._wrapped_data

    def dump(self, path_or_fp: Union[Path, str, IO], data: Any = None) -> None:
        """Dumps the currently wrapped data encoded as YAML into the given
        `path_or_fp`.

        This method updates :attr:`data_path` as a side-effect.

        :param path_or_fp: Either a path or a file-like object.
        :param data: (Optional) Use the given data for serialization rather
                     than :attr:`wrapped_data`.
        """
        if data is None:
            data = self.wrapped_data

        dump_data(data, path_or_fp)

        self.data_path = None if isinstance(path_or_fp, IOBase) else \
            Path(path_or_fp)

    @classmethod
    def load(cls, path_or_fp: Union[Path, str, IO]) -> 'DataWrapper':
        """Loads data from `path_or_fp` and creates a new wrapper instance
        around it.

        The returned instance has a properly set :attr:`data_path`.

        :param path_or_fp: Eithe a path or a file-like object.
        :return: A new instance of the class.
        """
        data = load_data(path_or_fp)

        wrapper = cls(wrapped_data=data)

        if not isinstance(path_or_fp, IOBase):
            wrapper.data_path = Path(path_or_fp).resolve()

        return wrapper


class ObjectWrapper(DataWrapper, MutableMapping):
    """A wrapper around a structured object providing convenience methods.

    The underlying raw data can be accessed using the :attr:`wrapped_data`
    property. To update this data you can either directly modify the data
    in this property or treat the :class:`ObjectWrapper` instance as a `dict`.

    >>> wrapper = ObjectWrapper({'foo': 'bar', 13: 37})
    >>> wrapper['foo'] = 'not-bar'
    >>> del wrapper[13]
    >>> assert wrapper.wrapped_data == {'foo': 'not-bar'}
    """

    def __init__(self, wrapped_data=None, **kwargs):
        if wrapped_data is None:
            wrapped_data = dict()

        super().__init__(wrapped_data)

        props = set(dir(self))

        for key, value in kwargs.items():
            if key not in props:
                raise AttributeError('unknown attribute ' + str(key))

            setattr(self, key, value)

    def get(self, key: str, transform: Optional[Callable[[Any], Any]] = None,
            default: Any = None):
        """Convenience method to get a property's value and apply additional
        transforms to it.

        :param key: The key of the property in the wrapped object.
        :param transform: (Optional) A transform applied to the original
                          property's value before returning it.
        :param default: (Optional) A default (already transformed) value to
                        return if the property is not present.
        :return: Eventually the (transformed) value of the property.
        """
        value = self.wrapped_data.get(key)

        if value is None:
            return default

        if transform is not None:
            value = transform(value)

        return value

    def set(self, key: str, value: Any, transform=None) -> None:
        """Convenience method to update a property with an optional transform.

        Makes sure that the transform is only called when `value` is not
        `None`. In case `value` is `None`, the property is removed from the
        wrapped object.
        """
        if value is None:
            self.wrapped_data.pop(key, None)
        else:
            if transform is not None:
                value = transform(value)

            self.wrapped_data[key] = value

    def __getitem__(self, key):
        return self.wrapped_data[key]

    def __setitem__(self, key, value):
        self.wrapped_data[key] = value

    def __delitem__(self, key):
        del self.wrapped_data[key]

    def __iter__(self):
        return self.wrapped_data.__iter__()

    def __len__(self):
        return len(self.wrapped_data)

    def __str__(self):
        return '{' + ', '.join(k + ': ' + str(self[k]) for k in self) + '}'


class ListWrapper(DataWrapper, MutableSequence):
    """Wraps (and behaves like) a `list` where each item should be wrapped by a
    :class:`DataWrapper` itself and ensures that both `list` s stay in sync.

    Check :func:`sequence` for a simple way to integrate it in an
    :class:`ObjectWrapper`.
    """

    def __init__(self, *args, wrapped_data: Optional[List[Any]] = None):
        if len(args) == 0:
            items = []
        elif len(args) == 1:
            if isinstance(args[0], list):
                items = args[0]
            elif isinstance(args[0], types.GeneratorType) or \
                    isinstance(args[0], ListWrapper) or \
                    isinstance(args[0], tuple):
                items = list(args[0])
            else:
                raise RuntimeError('unable to handle type {} as an '
                                   'argument'.format(type(args[0])))
        else:
            items = args

        if wrapped_data is None:
            wrapped_data = [item.wrapped_data for item in items]

        super().__init__(wrapped_data)

        #: Wrapper type instances for each item in the original list.
        self._items = items

    def __getitem__(self, key):
        items = self._items[key]
        if isinstance(key, slice):
            items = self.__class__(items)
        return items

    def __setitem__(self, key, value):
        self._items[key] = value

        if type(key) == slice:
            value = [i.wrapped_data for i in value]

        self.wrapped_data[key] = value

    def __delitem__(self, key):
        del self._items[key]
        del self.wrapped_data[key]

    def __len__(self):
        return len(self._items)

    def __eq__(self, other):
        return type(other) in (list, type(self)) and \
            len(self) == len(other) and \
            all(a == b for a, b in zip(self, other))

    def insert(self, idx, obj):
        self._items.insert(idx, obj)
        self.wrapped_data.insert(idx, obj.wrapped_data)

    def __add__(self, other):
        new_list = self.__class__()
        new_list.extend(self)
        new_list.extend(other)
        return new_list

    def __str__(self):
        return '[' + ', '.join(str(i) for i in self._items) + ']'


class DictWrapper(DataWrapper, MutableMapping):
    """Wraps (and behaves like) a `dict` where each value should be wrapped by
    a :class:`DataWrapper` itself and ensures that both `dict` s stay in sync.

    Check :func:`mapping` for a simple way to integrate it in an
    :class:`ObjectWrapper`.
    """

    def __init__(self, *args, wrapped_data=None, **kwargs):
        """Initializes a new :class:`DictWrapper` by setting all values
        supplied via `kwargs` and optionally sets `wrapped_data`.

        Note, it's your responsibility to ensure that the entities supplied
        use wrapped data that is also contained in the optionally supplied
        `wrapped_data`.
        """
        assert 0 <= len(args) <= 1
        assert kwargs or not wrapped_data

        items = {}

        if len(args) == 1:
            if isinstance(args[0], dict):
                items = args[0]
            elif isinstance(args[0], DictWrapper):
                items = dict(args[0])
            else:
                raise RuntimeError('unable to handle type {} as an '
                                   'argument'.format(type(args[0])))

        for k, v in kwargs.items():
            items[k] = v

        if wrapped_data is None:
            wrapped_data = {k: v.wrapped_data for k, v in items.items()}

        super().__init__(wrapped_data)

        self._items = items

    def __getitem__(self, key):
        return self._items[key]

    def __setitem__(self, key, value):
        self._items[key] = value
        self.wrapped_data[key] = value.wrapped_data

    def __delitem__(self, key):
        del self._items[key]
        del self.wrapped_data[key]

    def __iter__(self):
        return self._items.__iter__()

    def __len__(self):
        return len(self._items)

    def __str__(self):
        return '{' + ', '.join(k + ': ' + str(self[k]) for k in self) + '}'


def attribute(name: str, get: Optional[Callable[[Any], Any]] = None,
              set: Optional[Callable[[Any], Any]] = None) -> property:
    """Creates a `property` that proxies a simple attribute in an
    :class:`ObjectWrapper` by writing to the underlying :attr:`wrapped_data`.

    :param name: Name of the property in the wrapped object data.
    :param get: An optional conversion to apply to the retrieved
                value.
    :param set: An optional conversion to apply to the new value about to be
                written.
    :return: A new `property` that proxies the attribute `name`.
    """
    def fget(self):
        value = self.wrapped_data.get(name)

        if get is not None and value is not None:
            value = get(value)

        return value

    def fset(self, value):
        if set is not None and value is not None:
            value = set(value)

        if value is None:
            try:
                del self.wrapped_data[name]
            except KeyError:
                pass
        else:
            self.wrapped_data[name] = value

    return property(fget, fset)


def sequence(name: str, item_type) -> property:
    """Creates a property that proxies a `list` using a :class:`ListWrapper`
    with items of `item_type`.

    **Beware:** The returned property works only in classes inheriting
    from :class:`ObjectWrapper`.

    :param name: The name of the property in the wrapped data.
    :param item_type: The type of the wrapper class of the sequence's items or
                      a `str` of the full qualifying type.
    :return: A new `property`.
    """
    field = '_' + name

    def fget(self):
        if not hasattr(self, field):
            # install empty list if not existing
            if name not in self.wrapped_data:
                self.wrapped_data[name] = []

            item_cls = item_type

            if isinstance(item_cls, str):
                item_cls = pydoc.locate(item_cls)
                assert item_cls is not None

            list_inst = ListWrapper([item_cls(wrapped_data=i) for i in
                                     self[name]], wrapped_data=self[name])

            setattr(self, field, list_inst)

        return getattr(self, field)

    def fset(self, value):
        if value is None or len(value) == 0:
            try:
                del self.wrapped_data[name]
            except KeyError:
                pass

            try:
                delattr(self, field)
            except AttributeError:
                pass
        else:
            if type(value) == list:
                value = ListWrapper(value)

            self.wrapped_data[name] = value.wrapped_data
            setattr(self, field, value)

    return property(fget, fset)


def mapping(name: str, item_type) -> property:
    """Creates a property that proxies a `dict` using a :class:`DictWrapper`
    with items of `item_type`.

    **Beware:** The returned property works only in classes inheriting
    from :class:`ObjectWrapper`.

    :param name: The name of the property in the wrapped data.
    :param item_type: The type of the wrapper class of the sequence's items or
                      a `str` of the full qualifying type.
    :return: A new `property`.
    """
    field = '_' + name

    def fget(self):
        if not hasattr(self, field):
            # install empty list if not existing
            if name not in self.wrapped_data:
                self.wrapped_data[name] = {}

            item_cls = item_type

            if isinstance(item_cls, str):
                item_cls = pydoc.locate(item_cls)
                assert item_cls is not None

            dict_inst = DictWrapper(**{k: item_cls(wrapped_data=v)
                                       for k, v in self[name].items()},
                                    wrapped_data=self[name])

            setattr(self, field, dict_inst)

        return getattr(self, field)

    def fset(self, value):
        if value is None or len(value) == 0:
            try:
                del self.wrapped_data[name]
            except KeyError:
                pass

            try:
                delattr(self, field)
            except AttributeError:
                pass
        else:
            if isinstance(value, dict):
                value = DictWrapper(**value)

            self.wrapped_data[name] = value.wrapped_data
            setattr(self, field, value)

    return property(fget, fset)


def dump_data(data: Any, path_or_fp: Union[str, Path, IO]) -> None:
    """Dumps the given `data` into `path_or_fp` serialized as JSON."""
    def to_json(o, indent=0):
        ret = ''
        if isinstance(o, dict):
            indent += 2
            items = []
            for k, v in o.items():
                key = json.dumps(str(k))
                value = to_json(v, indent + len(key) + 2)

                items.append((key, value))

            items = natsorted(items, key=lambda x: x[0])

            ret += '{ '
            ret += (',\n' + ' ' * indent).join(k + ': ' + v for k, v in items)
            ret += ' }'
        elif isinstance(o, list):
            if all(isinstance(o, numbers.Real) for o in o):
                start, sep, end = '[', ', ', ']'
            else:
                start, sep, end = '[ ', '\n' + ' ' * indent + ', ', ' ]'

            ret += start
            ret += sep.join(to_json(o, indent + 2) for o in o)
            ret += end
        else:
            ret += json.dumps(o)
        return ret

    def dump(fp, obj):
        fp.write(to_json(obj))

    if isinstance(path_or_fp, IOBase):
        dump(path_or_fp, data)
    else:
        with open(str(path_or_fp), 'w') as fp:
            dump(fp, data)


def load_data(path_or_fp: Union[str, Path, IO]):
    """Loads a JSON serialized data structure from `path_or_fp`."""
    def load(fp):
        return json.load(fp, object_pairs_hook=OrderedDict)

    if isinstance(path_or_fp, IOBase):
        return load(path_or_fp)
    else:
        with open(str(path_or_fp)) as fp:
            return load(fp)
