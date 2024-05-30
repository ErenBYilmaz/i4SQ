"""Various helper functions.

Image data
----------

Just given a blob of data, i.e., a `ndarray`, you can use `guess_image_shape`
to find the number of image dimensions and potentially number of channels,
if any.

Pseudo 3D data
--------------

Some applications can be seamlessly made to work with 2D and 3D image data,
by converting the 2D data into **pseudo 3D image data** prior to processing.
This is done by introducing an artificial z-axis (0-th axis).

To convert to this **pseudo 3D data** and back you can use the following
functions:

- Image data: `as_3d_image` <-> `as_2d_image`
- Coordinates: `as_3d_point` <-> `as_2d_point`
- Shapes: `as_3d_size` <-> `as_2d_size`

Logging
-------

One should use the default Python `logging` module to create meaningful
log statements. To have a proper output for such log messages, one can
use `show_logs` and `write_logs`.

Working directory
-----------------

In order to simplify the work with optional working directories, we provide
a `WorkingDirectory` class:

```python
working_dir = WorkingDir()
assert not working_dir

with working_dir.join('foo/bar'):
    assert not working_dir

with working_dir.set('joho'):
    assert working_dir == 'joho'

assert not working_dir

@working_dir.cache('file_name')
def expensive_comp():
    return 42

value = expensive_comp()
```
"""


import coloredlogs
import functools
import gzip
import logging
import numpy as np
import os
import pickle
import re
import warnings

from pathlib import Path
from typing import Tuple, Optional, Union


log = logging.getLogger(__name__)


def show_logs(level: int = logging.INFO,
              log_to_file: Optional[Union[str, Path]] = None,
              silence_matplotlib: bool = True,
              silence_tensorflow: bool = True) -> None:
    """Show logs on the console and write them optionally to a file.

    We write _ALL_ the output to the file, ignoring the given level, which
    just applies to console output.

    Args:
        level: Minimal log level to show.
        log_to_file: If given, write full logs to this file.
        silence_matplotlib: Whether to increase matplotlib level to warning.
        silence_tensorflow: Whether to increase tensorflow level to warning.
    """

    if silence_tensorflow:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ignore warning and below
        import tensorflow as tf
        tf.logging.set_verbosity(tf.logging.ERROR)

    if silence_matplotlib:
        mpl_logger = logging.getLogger('matplotlib')
        mpl_logger.setLevel(logging.WARNING)

    coloredlogs.install(fmt='%(asctime)s.%(msecs)03d  %(levelname)-1.1s  '
                            '%(message)s   [%(name)s]',
                        datefmt='%m/%d %H:%M:%S', level=level,
                        field_styles={'asctime': {'color': 'white'},
                                      'name': {'color': 'blue'},
                                      'levelname': {'color': 'magenta'}},
                        level_styles={'debug': {'color': 'cyan'},
                                      'info': {'color': 'white'},
                                      'warning': {'color': 'yellow'},
                                      'error': {'color': 'red'},
                                      'fatal': {'background': 'red',
                                                'color': 'white',
                                                'bold': True}})

    if log_to_file:
        write_logs(log_to_file)


def write_logs(path: Union[str, Path], level: int = logging.DEBUG):
    """Adds a handler to write the logs fully to a file.

    Args:
        path: Path to the file where to write the logs to.
        level: The minimal log level.
    """
    formatter = logging.Formatter(
        fmt='%(asctime)s.%(msecs)03d  %(levelname)s  %(name)s  %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S')

    file_handler = EscapeCodeFilteringFileHandler(path, encoding='utf-8')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    logging.getLogger().addHandler(file_handler)


class EscapeCodeFilteringFileHandler(logging.FileHandler):
    """Removes ESC ANSI codes from log messages."""

    PATTERN = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')

    def format(self, record):
        message = super().format(record)
        return self.PATTERN.sub('', message)


def guess_image_shape(image: np.ndarray, max_n_channels: int = 5,
                      legit_n_channels: set = {1, 3}) \
        -> Tuple[int, Optional[int]]:
    """Given an array, this function tries to guess the dimensionality
    and number of channels (if an extra dimension is used for this) for an
    image represented by the array.

    Supports (multi channel) images in 2D and 3D. However, note that there is
    some uncertainty in estimating whether the last axis corresponds to the
    channel of the image shape.

    Args:
        image: An array representing an image.
        max_n_channels: Axis shape value that is used to determine whether an
            axis corresponds to the image shape or the image's channels.
        legit_n_channels: Do not issue a user warning for these
            well-known channel counts.

    Returns:
        (`n_dims`, `n_channels`), `n_channels` is `None` if there is no
        additional axis for the channels.
    """
    image = np.asarray(image)

    assert 2 <= image.ndim <= 4

    n_channels = image.shape[-1]

    if image.ndim == 2:
        return 2, None
    elif image.ndim == 4:
        if n_channels > max_n_channels:
            warnings.warn(f'{n_channels} channels is unlikely for a 3D image '
                          f'(array shape = {image.shape})')
        return 3, n_channels
    elif n_channels > max_n_channels:
        return 3, None
    else:
        if n_channels not in legit_n_channels:
            warnings.warn('Assuming last axis represents channels of a 2D '
                          f'image (array shape = {image.shape})')
        return 2, n_channels


def dist(pos_a: np.ndarray, pos_b: np.ndarray,
         sqrt: bool = True) -> Union[float, np.ndarray]:
    """Computes Euclidean distance for one or multiple position pairs.

    Args:
        pos_a: One or multiple positions.
        pos_b: One or multiple positions.
        sqrt: Whether to apply the square root.

    Returns: Either a single distance or multiple.
    """
    pos_a = np.asarray(pos_a)
    pos_b = np.asarray(pos_b)

    is_single = pos_a.ndim == 1 and pos_b.ndim == 1

    tmp = (pos_a - pos_b)**2
    if tmp.ndim == 1:
        tmp.shape = (1, tmp.size)

    tmp = np.sum(tmp, axis=1)

    if sqrt:
        tmp = np.sqrt(tmp)

    if is_single:
        tmp = tmp[0]

    return tmp


def as_3d_image(image):
    """Converts the given image into a 3D image, if it is not already, by
    appending a new axis with shape 1."""
    image = np.asarray(image)

    n_dims, n_channels = guess_image_shape(image)

    if n_dims == 2:
        image = image.reshape((1,) + image.shape)

    return image


def as_3d_point(point):
    """Converts the given index into a 3D point, if it is not already, by
    prepending a new axis with value 0."""
    assert 2 <= len(point) <= 3

    point = np.asarray(point)

    if point.shape[0] == 2:
        point = np.hstack((0, point))

    return point


def as_3d_size(size):
    """Converts the given size into a 3D size, if it is not already, by
    prepending a new axis with size 1."""
    assert 2 <= len(size) <= 3

    size = np.asarray(size)

    if size.shape[0] == 2:
        size = np.hstack((1, size))

    return size


def as_2d_image(image):
    """Converts a pseudo 3D image into a 2D image, if it is not already a 2D
    image, by dropping the 0-th axis."""
    image = np.asarray(image)
    assert 2 <= image.ndim <= 4

    n_dims, n_channels = guess_image_shape(image)

    if n_dims == 3:
        image = image[0]

    return image


def as_2d_point(point):
    """Converts a pseudo 3D point into a 2D point, if its not already a 2D
    point, by dropping the 0-th index."""
    assert 2 <= len(point) <= 3

    point = np.asarray(point)

    if point.shape[0] == 3:
        if point[0] != 0:
            warnings.warn('Point {} might not be a pseudo 3D point'
                          .format(point))
        point = point[1:]

    return point


def as_2d_size(size):
    """Converts a pseudo 3D size into a 2D size, if it is not already a 2D
    size, by dropping the 0-th index."""
    assert 2 <= len(size) <= 3

    size = np.asarray(size)

    if size.shape[0] == 3:
        if size[0] != 1:
            warnings.warn('Size {} might not be a pseudo 3D size'
                          .format(size))
        size = size[1:]

    return size


class WorkingDirectory(os.PathLike):
    """A state-ful working directory manager.

    An instance of this class (singleton) can be used to maintain a hierarchic
    working directory structure without the need for additional parameters.
    This eases the task of caching intermediate values to disk.

    This manager propagates unset working directories without failing for
    all operations.

    Args:
        path: The initial working directory path.
        ensure_exists: Ensures that directories are created if they do not
            exist yet.
    """
    def __init__(self, path: Optional[Union[str, Path]] = None,
                 ensure_exists: bool = True):
        self._path: Optional[Path] = None

        self.ensure_exists = ensure_exists
        """Whether to ensure the directories exist by creating them if
        necessary.
        """

        if path is not None and not isinstance(path, Path):
            path = Path(path)

        self.path = path

    @property
    def path(self) -> Optional[Path]:
        """The underlying path, or `None`."""
        return self._path

    @path.setter
    def path(self, value: Optional[Union[str, Path]]) -> None:
        if value is not None:
            if not isinstance(value, Path):
                value = Path(value)

            if self.ensure_exists:
                value.mkdir(exist_ok=True, parents=True)

        self._path = value

    def join(self, segment: Union[str, Path]) -> 'JoinContext':
        """Creates a context manager that changes the working directory by
        appending this path `segment`.
        """
        return WorkingDirectory.JoinContext(self, segment)

    def join_inplace(self, segment: Union[str, Path]) -> None:
        """Changes the working directory path in place by joining the
        given `segment`.
        """
        if self.path is not None:
            self.path = self.path / segment

    def set(self, path: Union[str, Path]) -> 'SetContext':
        """Creates a context manager that changes the working directory by
        setting it to a new `path`.
        """
        return WorkingDirectory.SetContext(self, path)

    def set_inplace(self, path: Union[str, Path]) -> None:
        """Changes the working directory path in place by setting it to the
        given new `path`.
        """
        self.path = path

    def cache(self, segment: Union[str, Path]):
        """Function decorator to optionally cache the return value.

        In case the working dir has a path set, the function return value is
        written to disk on the first access, and read from it without calling
        the actual function in successive calls.

        Args:
            segment: Relative path (filename) that points to the cache file
                w.r.t. to the current working directory path.
        """
        cache_path = self / segment

        if cache_path and self.ensure_exists:
            cache_path.parent.mkdir(exist_ok=True, parents=True)

        def cache(func):
            @functools.wraps(func)
            def wrapper():
                if cache_path and cache_path.exists():
                    log.debug('Re-using resulft of previous call to %s() '
                              'stored in %s', func.__name__, cache_path)

                    with gzip.open(cache_path, 'rb') as fp:
                        return pickle.load(fp)  # noqa

                value = func()

                if cache_path:
                    with gzip.open(cache_path, 'wb', compresslevel=5) as fp:
                        pickle.dump(value, fp, pickle.HIGHEST_PROTOCOL)  # noqa

                return value
            return wrapper
        return cache

    def __bool__(self):
        return self.path is not None

    def __eq__(self, other):
        # the isinstance check on WorkingDirectory fails for PurePath objects
        if type(other) == WorkingDirectory:
            other = other.path
        elif type(other) == str:
            other = Path(other)

        return self.path == other

    def __truediv__(self, other):
        segment = Path(other)

        if self.path:
            return self.path / segment

        return None

    def __fspath__(self):
        assert self.path is not None
        return self.path.__fspath__()

    def __str__(self):
        return str(self.path) if self.path else ''

    def __repr__(self):
        return repr(self.path) if self.path else ''

    class JoinContext:
        def __init__(self, working_dir: 'WorkingDirectory',
                     segment: Union[str, Path]):
            self.working_dir = working_dir
            self.segment = segment

        def __enter__(self):
            self.prev_path = self.working_dir.path
            self.working_dir.path = self.working_dir / self.segment

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.working_dir.path = self.prev_path

    class SetContext:
        def __init__(self, working_dir: 'WorkingDirectory',
                     path: Union[str, Path]):
            self.working_dir = working_dir
            self.path = path

        def __enter__(self):
            self.prev_path = self.working_dir.path
            self.working_dir.path = self.path

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.working_dir.path = self.prev_path


def write_pointset_file(positions: np.ndarray, path: Path) -> None:
    """Writes a PointSet file as a understood by MITK.

    Args:
        positions: Set of world coordinates to write.
        path: Where to store that file, default ending is `.mps`.
    """
    if positions.ndim == 1:
        positions = positions.reshape((1, positions.size))

    if positions.shape[1] == 2:
        positions = np.concatenate((positions, np.zeros((len(positions), 1))),
                                   axis=1)

    min_pos = positions.min(axis=0)
    max_pos = positions.max(axis=0)

    def point(i, pos):
        return f'''
        <point>
            <id>{i}</id>
            <specification>0</specification>
            <x>{pos[0]}</x>
            <y>{pos[1]}</y>
            <z>{pos[2]}</z>
        </point>
        '''

    points = '\n'.join(point(*args) for args in enumerate(positions))

    contents = f'''
    <?xml version="1.0" encoding="UTF-8" ?>
    <point_set_file>
        <file_version>0.1</file_version>
        <point_set>
            <time_series>
                <time_series_id>0</time_series_id>
                <Geometry3D ImageGeometry="false" FrameOfReferenceID="0">
                    <IndexToWorld type="Matrix3x3" m_0_0="1" m_0_1="0"
                        m_0_2="0" m_1_0="0" m_1_1="1" m_1_2="0" m_2_0="0"
                        m_2_1="0" m_2_2="1" />
                    <Offset type="Vector3D" x="0" y="0" z="0" />
                    <Bounds>
                        <Min type="Vector3D" x="{min_pos[0]}" y="{min_pos[1]}"
                            z="{min_pos[2]}" />
                        <Max type="Vector3D" x="{max_pos[0]}" y="{max_pos[1]}"
                            z="{max_pos[2]}" />
                    </Bounds>
                </Geometry3D>
                {points}
            </time_series>
        </point_set>
    </point_set_file>
    '''

    path.write_text(contents)
