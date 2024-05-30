import gzip
import itk
import json
import logging
import numbers
import numpy as np
import os
import os.path
import psutil
import SimpleITK as sitk  # noqa
import threading
import voluptuous as v
import warnings

from copy import copy, deepcopy
from humanfriendly import format_size
from io import IOBase
from pathlib import Path
from typing import Any, Dict, IO, Optional, Union
from voluptuous import Invalid, Required, Schema, ALLOW_EXTRA

from ..wrapper import ObjectWrapper, ListWrapper, attribute, sequence, mapping
from ..wrapper import load_data
from ..utils import as_3d_point, as_3d_size, as_2d_point, \
    as_2d_size, guess_image_shape


log = logging.getLogger(__name__)


def array(name: str) -> property:
    def to_native(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    return attribute(name, np.array, to_native)


class DataCache:
    """A class to re-use already loaded/cached images instead of loading
    them again.

    Can be used to cache arbitrary `load_image()`-able files in memory.
    """
    def __init__(self, max_in_memory: int) -> None:
        """
        Args:
            max_in_memory: Maximum bytes of image data to hold in memory before
                starting to release unused images.
        """
        self._max_in_memory = max_in_memory
        self._lock = threading.Lock()
        self._cached_bytes = 0
        self._used_recently = []
        self._data_cache = {}

    def get_or_load(self, path: Union[str, Path]) -> np.ndarray:
        """Gets the cached image or loads it from disk if is not cached yet.

        Might remove images from cache that have not been used recently to
        respect the `max_in_memory` bytes.

        Args:
            path: Path to the image to load from disk.

        Returns:
            An `np.ndarray` of the image.
        """
        path = str(Path(path).resolve())

        with self._lock:
            data = self._data_cache.get(path)

            if data is None:
                data = load_image(path)
                log.debug('Loaded new image %s with %s', path,
                          format_size(data.nbytes))

                # since this data might be shared by multiple threads, we want
                # to make sure that it is not modified
                data.flags.writeable = False

                if data.nbytes <= self._max_in_memory:
                    prev_bytes = self._cached_bytes
                    dropped = []

                    while self._cached_bytes + data.nbytes > \
                            self._max_in_memory:
                        drop = self._used_recently.pop()
                        self._cached_bytes -= self._data_cache.pop(drop).nbytes
                        dropped.append(drop)

                    if len(dropped) > 0:
                        log.debug('Dropped %i images releasing %s (%s)',
                                  len(dropped),
                                  format_size(prev_bytes - self._cached_bytes),
                                  ', '.join(dropped))

                    self._cached_bytes += data.nbytes
                    self._used_recently.insert(0, path)
                    self._data_cache[path] = data

                    log.debug('Cache now contains %i images with %s',
                              len(self._data_cache),
                              format_size(self._cached_bytes))
                else:
                    log.debug('New image greater than whole cache, droppping '
                              'everything')
                    self._cached_bytes = 0
                    self._used_recently = []
                    self._data_cache = {}
            else:
                self._used_recently.remove(path)
                self._used_recently.insert(0, path)

        return data

    def remove(self, path: Union[str, Path]) -> bool:
        """Removes a specific image from the cache if it exists.

        Returns:
            Whether an image was removed or not.
        """
        path = str(Path(path).resolve())

        with self._lock:
            data = self._data_cache.pop(path, None)

            if data is not None:
                self._cached_bytes -= data.nbytes
                self._used_recently.remove(path)

                return True

        return False

    def clear(self) -> None:
        """Removes all cached images."""
        with self._lock:
            self._cached_bytes = 0
            self._used_recently = []
            self._data_cache = {}


class Object(ObjectWrapper):
    """Represents an object in an image."""

    #: Reference point of the object, i.e., it's position.
    position = array('position')

    #: The outline of an object as list of points.
    outline = array('outline')

    #: Bounding box around an object.
    bounding_box = array('bounding_box')

    #: Dictionary mapping a part identifier to an :class:`Object`.
    parts = mapping('parts', 'hiwi.image.Object')


class Image(ObjectWrapper):
    DATA_CACHE = DataCache(
        max_in_memory=min(psutil.virtual_memory().total * 0.2, 1024**3 * 10))
    """A global cache object that is used to store loaded images."""

    def __init__(self, image_list_path: Optional[Path] = None, *args,
                 **kwargs) -> None:
        self._image_list_path = image_list_path
        self._data = None
        super().__init__(*args, **kwargs)

    @property
    def path(self):
        """Path to the image data, which is normalized to an absolute path on
        access.
        """
        def transform(value):
            path = Path(value)

            if not path.is_absolute():
                if self.data_path is not None:
                    new_path = self.data_path.parent / path

                    if new_path.exists():
                        return new_path

                if self._image_list_path is not None:
                    new_path = self._image_list_path.parent / path

                    if new_path.exists():
                        return new_path

            return path

        return self.get('path', transform)

    @path.setter
    def path(self, value):
        self['path'] = str(value)

    @property
    def name(self):
        """The name of the image _WITHOUT_ the extension.

        **Beware,** since the goal is to strip multi part extensions like
        ".npy.gz", it might strip to much if the name contains superfluous
        dots.
        """
        parts = self.path.name.split('.')

        if len(parts) > 2 and parts[-1] in ('gz', 'bz2', 'xz'):
            return '.'.join(parts[:-2])
        else:
            return '.'.join(parts[:-1])

    @property
    def full_name(self):
        """The name of the image _INCLUDING_ the extension."""
        return self.path.name

    #: Optional spacing of the image in mm/px.
    spacing = array('spacing')

    #: A list of :class:`Object` s.
    objects = sequence('objects', Object)

    @property
    def parts(self) -> Dict[str, Object]:
        """Returns the parts of the first object annotation.

        Throws an error if there is not exactly one object annotation.
        """
        assert len(self.objects) == 1, 'there must be exactly _ONE_ object, ' \
                                       'use objects to solve the ambiguity'
        return self.objects[0].parts

    @property
    def data(self) -> np.ndarray:
        """Reads the image data into a `numpy` array."""
        if self._data is not None:
            return self._data

        return Image.DATA_CACHE.get_or_load(self.path)

    @data.setter
    def data(self, data: np.ndarray) -> None:
        self._data = data

    def transformed(self, image_path: Union[str, Path],
                    scaling: Union[None, float, np.ndarray] = None,
                    normalization: Optional[str] = None) -> 'Image':
        """Creates a new :class:`Image` by transforming the original
        image and its associated meta data `spacing`, object `position` and
        object `outline`.

        Args:
            image_path: Where to store the transformed image.
            scaling: A scale parameter applied to the image and
                its meta data. Either for one or all dimensions,
                e.g. s | (sx, sy[, sz]). In case of None, this parameter
                is set to 1.
            normalization: An optional normalization to apply to the data.
                Must be one of 'standard-score' or 'abs-standard-score'. In
                case of None, it has no effect.

        Returns:
            A new :class:`Image` object referencing the transformed image
            located at `image_path`.
        """
        assert normalization in (None, 'standard-score', 'abs-standard-score')

        image_itk = sitk.ReadImage(str(self.path))
        image_data = sitk.GetArrayViewFromImage(image_itk)

        image = Image(wrapped_data=deepcopy(self.wrapped_data))
        image.path = image_path

        n_dims, n_channels = guess_image_shape(image_data)
        is_2d = n_dims == 2

        # we transform 2d images/coordinates into 3D space for simplicity
        scaling = np.ones(3) if scaling is None else \
            as_3d_size((np.ones(n_dims) * scaling)[::-1])

        if is_2d:
            scaling[0] = 1

        spacing = image.spacing
        if spacing is not None:
            spacing = as_3d_size(spacing[::-1])

        # resample image
        new_image_itk = resample_image(image_itk, scaling[-n_dims:])
        if not os.path.isfile(str(image_path)):
            image_data = sitk.GetArrayFromImage(new_image_itk)

        matrix = np.zeros((4, 4))
        matrix[([0, 1, 2], [0, 1, 2])] = scaling

        def transform_point(point):
            point = np.asarray(point, dtype=float)
            physical = image_itk.TransformContinuousIndexToPhysicalPoint(point)

            point = point + 0.5
            point = as_3d_point(point[::-1])
            point = (matrix @ np.hstack((point, 1)))[:-1]

            if is_2d:
                point = as_2d_point(point)

            point = (point - 0.5)[::-1]

            physical_new = new_image_itk \
                .TransformContinuousIndexToPhysicalPoint(point)

            assert (np.abs(np.array(physical) - physical_new) < 1e-5).all()

            return point

        if not os.path.isfile(str(image_path)):
            if normalization == 'standard-score':
                image_data = (image_data - image_data.mean()) / image_data.std()
                image_data = image_data.astype(np.float32)
            elif normalization == 'abs-standard-score':
                image_data = np.abs((image_data - image_data.mean()) /
                                    image_data.std())
                image_data = image_data.astype(np.float32)

            if normalization is not None:
                new_image_itk = sitk.GetImageFromArray(
                    image_data, isVector=n_channels is not None)

        if spacing is not None:
            spacing /= scaling
            if is_2d:
                spacing = as_2d_size(spacing)
            image.spacing = spacing[::-1]

        def transform_object(obj):
            if obj.position is not None:
                obj.position = transform_point(obj.position)

            if obj.outline is not None:
                obj.outline = np.array([transform_point(p) for p
                                        in obj.outline])

            if obj.bounding_box is not None:
                assert is_2d, 'for 3d not implemented yet'
                origin = obj.bounding_box[:2]
                p0 = transform_point(origin)
                p1 = transform_point(origin + [0, obj.bounding_box[3]])
                p2 = transform_point(origin + [obj.bounding_box[2], 0])
                p3 = transform_point(origin + obj.bounding_box[2:])

                maxp = np.amax([p0, p1, p2, p3], axis=0)
                minp = np.amin([p0, p1, p2, p3], axis=0)

                obj.bounding_box = np.hstack((minp, maxp - minp))

            for _, part_obj in obj.parts.items():
                transform_object(part_obj)

        for obj in image.objects:
            transform_object(obj)

        if not os.path.isfile(str(image_path)):
            sitk.WriteImage(new_image_itk, str(image_path))

        return image

    def dump(self, path_or_fp: Optional[Union[Path, str, IO]] = None,
             relative_path: bool = True) -> None:
        """Writes the image description to the optionally given file.

        If no `path_or_fp` is given, the data is dumped next to the image
        with a ".meta" suffix.

        :param path_or_fp: An optional path where to store the serialized data,
                           if `None`, it is placed next to the image with
                           suffix ".meta".
        :param relative_path: Whether to store the :attr:`path` relative to the
                              about to being serialized dump.
        """
        if path_or_fp is None:
            path_or_fp = self.path.parent / (self.name + '.meta')
        elif isinstance(path_or_fp, str):
            path_or_fp = Path(path_or_fp)

        base = path_or_fp.parent if isinstance(path_or_fp, Path) else None

        data = self.wrapped_data

        if relative_path and 'path' in data:
            if base is not None:
                data = copy(data)
                data['path'] = str(self.path.relative_to(base))
            else:
                warnings.warn('Unable to make the image path relative, '
                              'because no dump path is given')

        super().dump(path_or_fp, data=data)


class ImageList(ListWrapper):
    def dump(self, path_or_fp: Union[Path, str, IO],
             relative_paths: bool = True, use_references: bool = False,
             relative_refs: bool = True) -> None:
        """Dumps the :class:`ImageList` to the given `path_or_fp`.

        It can also use references to individual image-based meta dumps to
        make the serialized image lists less redundent. See `use_references`
        for more. **Note**, the contained :class:`Image` instances are dumped
        only if `data_path` is `None`, else the current `data_path` is re-used
        without dumping the image explicitly.

        :param path_or_fp: A path or a file-like object.
        :param relative_paths: Whether to store the image's `path` s relative
                               to the image list path.
        :param use_references: Whether to store paths to individual image meta
                               serialization or the image meta directly in the
                               serialized list.
        :param relative_refs: Whether to store the references relative
                              to the path of the image list.
        """
        directory = None

        if not isinstance(path_or_fp, IOBase):
            directory = Path(path_or_fp).parent

        data = []

        for image in self:
            if use_references:
                if image.data_path is None:
                    image.dump(relative_path=relative_paths)

                reference = image.data_path

                if relative_refs:
                    if directory is not None:
                        reference = os.path.relpath(str(reference),
                                                    str(directory))
                    else:
                        warnings.warn('Unable to make the image reference ',
                                      'path relative, because no dump path '
                                      'is given')

                reference = str(reference)
                if os.name == 'nt':
                    reference = reference.replace('\\', '/')

                data.append(reference)
            else:
                image_data = image.wrapped_data

                if relative_paths and 'path' in image_data:
                    if directory is not None:
                        image_data = copy(image_data)
                        try:
                            image_data['path'] = str(image.path
                                                     .relative_to(directory))
                        except ValueError:
                            warnings.warn('Unable to create relative path for '
                                          '{}, using original one'
                                          .format(image.path))
                    else:
                        warnings.warn('Unable to make the image path relative,'
                                      ' because no dump path is given')

                data.append(image_data)

        super().dump(path_or_fp, data)

    @classmethod
    def load(cls, path_or_fp: Union[Path, str, IO]) -> 'ImageList':
        path = None if isinstance(path_or_fp, IOBase) else Path(path_or_fp)

        try:
            return cls._load_legacy_list(path_or_fp)
        except json.JSONDecodeError:
            pass
        except Invalid:
            pass

        data = load_data(path_or_fp)

        try:
            return cls._load_reference_list(data, path)
        except Invalid:
            pass

        result = cls._load_full_list(data, path)
        result.name = str(path)
        return result

    @classmethod
    def _load_reference_list(cls, data: Any,
                             path: Optional[Path] = None) -> 'ImageList':
        schema = Schema([str])

        references = schema(data)

        images = cls()

        for reference in references:
            reference = Path(reference)

            if not reference.exists() or not reference.is_absolute():
                reference = path.parent / reference

            image = Image.load(reference)

            images.append(image)

        return images

    @classmethod
    def _load_full_list(cls, data: Any, path) -> 'ImageList':
        return cls([Image(wrapped_data=i, image_list_path=path) for i in data],
                   wrapped_data=data)

    @classmethod
    def _load_legacy_list(cls, path_or_fp) -> 'ImageList':
        path = None if isinstance(path_or_fp, IOBase) else Path(path_or_fp)

        if isinstance(path_or_fp, IOBase):
            data = json.load(path_or_fp)
        else:
            with open(path_or_fp) as fp:
                data = json.load(fp)

        point = [numbers.Real]

        def object_instance2(v):
            return object_instance(v)

        object_instance = Schema({
            'reference_point': point,
            'outline': [point],
            'bounding_box': point,
            'parts': {str: v.Any(object_instance2, point)}
        }, extra=ALLOW_EXTRA)

        schema = Schema([{
            Required('image_path'): str,
            'units': {'mm': {'spacing': [numbers.Real]}},
            'object_instances': [object_instance]
        }], extra=ALLOW_EXTRA)

        schema(data)

        images = []

        def create_object(object_data):
            obj = Object()

            if isinstance(object_data, list):
                obj.position = object_data
                return obj

            obj.position = object_data.get('reference_point')
            obj.outline = object_data.get('outline')

            for part_name, part_data in object_data.get('parts', {}).items():
                obj.parts[part_name] = create_object(part_data)

            return obj

        for image_data in data:
            image = Image(image_list_path=path)
            image.path = image_data.get('image_path')
            image.spacing = image_data.get('units', {}).get('mm', {}) \
                .get('spacing')

            for object_data in image_data['object_instances']:
                image.objects.append(create_object(object_data))

            images.append(image)

        return cls(images)


def load_image(path: Union[str, Path], spacing: bool = False,
               meta: bool = False) -> np.ndarray:
    """Convenience method to load an image/volume into a numpy array.

    Additionally to most common extensions, it also supports loading
    numpy arrays from .npy(.gz) files.

    Args:
        path: The path to load the image from.
        spacing: Whether to return the spacing, might be `None` though.
        meta: Whether to also return the full meta data dictionary.

    Returns:
        Either only the image or a tuple containing the image and the spacing,
        if `spacing` is `True`.
    """
    assert not (spacing and meta)

    path = str(path)

    image = None
    ret = False

    if path.endswith('.npy.gz'):
        with gzip.open(path) as fp:
            image = np.load(fp)
    elif path.endswith('.npy'):
        image = np.load(path)
    else:
        image_itk = sitk.ReadImage(path)
        image = sitk.GetArrayFromImage(image_itk)
        if spacing:
            ret = image_itk.GetSpacing()
        if meta:
            spacing = image_itk.GetSpacing()[::-1]
            origin = image_itk.GetOrigin()[::-1]

            n_dims = len(spacing)
            direction = image_itk.GetDirection()
            direction = np.array(direction).reshape(n_dims, n_dims)
            direction = tuple(direction[::-1, ::-1].flat)

            ret = dict(spacing=spacing, direction=direction, origin=origin)

    if ret is not False:
        return image, ret

    return image


def save_image(image: np.ndarray, path: Union[str, Path],
               spacing: Optional[np.ndarray] = None,
               meta: Optional[Dict[str, Any]] = None) -> None:
    """Convenience method to save an image/volume into a file based on the
    extension.

    Additionally to most common extensions, it also supports dumping
    numpy arrays into .npy(.gz) files.

    Args:
        image: The image to save.
        path: Where to save it to.
        spacing: An optional spacing.
        meta: A dictionary of meta values (spacing, origin, direction) to
            optionally save (this is exclusive to the spacing option).
    """
    path = str(path)

    if (path.endswith('.npy') or path.endswith('.npy.gz')) and \
            spacing is not None:
        warnings.warn('Dumping image to numpy array {} won\'t contain the '
                      'passed spacing {}'.format(path, spacing))

    if path.endswith('.npy.gz'):
        with gzip.open(path, 'wb') as fp:
            np.save(fp, image, allow_pickle=False)
    elif path.endswith('.npy'):
        np.save(path, image, allow_pickle=False)
    else:
        n_dims, n_channels = guess_image_shape(image)

        image_itk = sitk.GetImageFromArray(image,
                                           isVector=n_channels is not None)
        if spacing is not None:
            image_itk.SetSpacing(spacing[::-1])
        if meta is not None:
            image_itk.SetSpacing(meta['spacing'][::-1])
            image_itk.SetOrigin(meta['origin'][::-1])

            n_dims = len(meta['spacing'])
            direction = np.array(meta['direction']).reshape(n_dims, n_dims)
            direction = tuple(direction[::-1, ::-1].flat)

            image_itk.SetDirection(direction)
        sitk.WriteImage(image_itk, path)


def find_anatomical_orientation(image_itk: sitk.Image) -> str:
    """Extracts the anatomical orientation from direction cosines.

    This is basically a Python version of the ITK code [1] to extract
    the anatomical orientation from direction cosines.

    Returns the origin (index 0) for XYZ in anatomical terms:

    - R: right
    - L: left
    - A: anterior  (front)
    - P: posterior  (behind)
    - I: inferior  (below)
    - S: superior  (above)

    [1]: https://github.com/InsightSoftwareConsortium/ITK/blob/
    2074c2f9e5ed3f087d0e2059f8e0e8992fcad7ef/Modules/Core/Common/src/
    itkSpatialOrientationAdapter.cxx

    Args:
        image_itk: The ITK image to derive it for.

    Return: A three character string specifying the origin, e.g., `RIP`.
    """
    assert image_itk.GetDimension() == 3, 'works only with 3d images'

    def max3(x, y, z):
        thresh = 0.001
        x, y, z = abs(x), abs(y), abs(z)
        if x > thresh and x > y and x > z:
            return 0
        elif y > thresh and y > x and y > z:
            return 1
        elif z > thresh and z > x and z > y:
            return 2
        else:
            return 0

    def sign(x):
        return -1 if x < 0 else 1

    direction = np.array(image_itk.GetDirection()).reshape(3, 3)

    axes = [0] * 9
    for i in range(3):
        dominant_axis = max3(direction[0][i], direction[1][i],
                             direction[2][i])
        axes[dominant_axis + i * 3] = sign(direction[dominant_axis][i])

    terms = ['_'] * 3

    for i in range(3):
        for j, d, r in [(i * 3, 1, 'R'),
                        (i * 3, -1, 'L'),
                        (i * 3 + 1, 1, 'A'),
                        (i * 3 + 1, -1, 'P'),
                        (i * 3 + 2, 1, 'I'),
                        (i * 3 + 2, -1, 'S')]:
            if axes[j] == d:
                terms[i] = r
                break

    if any(t == '_' for t in terms):
        log.warning('Could not derive orientation from direction: %s', terms)
        return 'RIP'

    return ''.join(terms)


def change_anatomical_orientation(image_sitk: sitk.Image, orientation: str) \
        -> sitk.Image:
    """Uses ITK's OrientImageFilter to change the anatomical orientation.

    Args:
        image_sitk: A `SimpleITK.Image` instance.
        orientation: Index orientation in anatomical terms for XYZ, e.g., RIP.

    Return: A SimpleITK image with the changes orientation.
    """
    # ITK_COORDINATE_UNKNOWN = 0,
    # ITK_COORDINATE_Right = 2,
    # ITK_COORDINATE_Left = 3,
    # ITK_COORDINATE_Posterior = 4, // back
    # ITK_COORDINATE_Anterior = 5, // front
    # ITK_COORDINATE_Inferior = 8, // below
    # ITK_COORDINATE_Superior = 9 // above
    codes = {'R': 2, 'L': 3, 'P': 4, 'A': 5, 'I': 8, 'S': 9}
    assert len(orientation) == 3 and all(c in list(codes.keys())
                                         for c in orientation)

    # ITK_COORDINATE_PrimaryMinor = 0,
    # ITK_COORDINATE_SecondaryMinor = 8,
    # ITK_COORDINATE_TertiaryMinor = 16
    #
    # ITK_COORDINATE_ORIENTATION_SAR = (ITK_COORDINATE_Superior
    #                                   << ITK_COORDINATE_PrimaryMinor)
    # + (ITK_COORDINATE_Anterior << ITK_COORDINATE_SecondaryMinor)
    # + (ITK_COORDINATE_Right << ITK_COORDINATE_TertiaryMinor),
    code = (codes[orientation[0]] << 0) + \
        (codes[orientation[1]] << 8) + \
        (codes[orientation[2]] << 16)

    image_itk = sitk_image_to_itk_image(image_sitk)

    orienter = itk.OrientImageFilter.New(image_itk)
    orienter.UseImageDirectionOn()
    orienter.SetDesiredCoordinateOrientation(code)

    new_image_itk = orienter.GetOutput()

    return itk_image_to_sitk_image(new_image_itk)


def resample_image(image_itk: sitk.Image, scaling: np.ndarray,
                   is_segmentation: bool = False) -> sitk.Image:
    """Resamples the given `image_itk` using the `scaling` ([Z]YX).

    Uses linear interpolation unless `is_segmentation` is `True`, in which case
    a Gaussian label smoothing approach is used.
    """
    scaling = (np.ones(image_itk.GetDimension()) * scaling)[::-1]

    new_size = np.ceil(image_itk.GetSize() * scaling).astype(int)
    new_spacing = image_itk.GetSpacing() / scaling
    new_origin = image_itk.TransformContinuousIndexToPhysicalPoint(
        (0 + .5) / scaling - .5)

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size.tolist())
    resampler.SetInterpolator(sitk.sitkLabelGaussian if is_segmentation else
                              sitk.sitkBSpline)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(new_origin)
    resampler.SetOutputDirection(image_itk.GetDirection())

    return resampler.Execute(image_itk)


def itk_image_to_sitk_image(itk_image: itk.Image) -> sitk.Image:
    """Creates a `SimpleITK.Image` from an `itk.Image`."""
    sitk_image = sitk.GetImageFromArray(
        itk.GetArrayFromImage(itk_image),
        isVector=itk_image.GetNumberOfComponentsPerPixel() > 1)
    sitk_image.SetOrigin(tuple(itk_image.GetOrigin()))
    sitk_image.SetSpacing(tuple(itk_image.GetSpacing()))
    sitk_image.SetDirection(itk.GetArrayFromMatrix(itk_image.GetDirection())
                            .flatten())
    return sitk_image


def sitk_image_to_itk_image(sitk_image: sitk.Image) -> itk.Image:
    """Creates an `itk.Image` from a `SimpleITK.Image`."""
    itk_image = itk.GetImageFromArray(
        sitk.GetArrayFromImage(sitk_image),
        is_vector=sitk_image.GetNumberOfComponentsPerPixel() > 1)
    itk_image.SetOrigin(sitk_image.GetOrigin())
    itk_image.SetSpacing(sitk_image.GetSpacing())
    itk_image.SetDirection(
        itk.GetMatrixFromArray(np.reshape(np.array(sitk_image.GetDirection()),
                                          [sitk_image.GetDimension()] * 2)))
    return itk_image
