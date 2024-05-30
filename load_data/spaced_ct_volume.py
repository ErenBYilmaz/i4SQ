import math
from math import inf
from typing import Optional, Tuple
from warnings import warn

import cachetools
import numpy
import scipy.ndimage

from lib.tuned_cache import TunedMemory

import hiwi
from lib.util import EBC

from load_data import MADER_SPACING, X, Y, Z
from load_data.load_image import image_by_patient_id

spaced_volume_cache = TunedMemory(location='./.cache/spaced_volume_cache', verbose=1)
spaced_volume_shape_cache = TunedMemory(location='./.cache/spaced_volume_shape_cache', verbose=0)
spaced_volume_ram_cache = cachetools.LRUCache(maxsize=1)


class HUWindow(EBC):
    def __init__(self, minimum: float, maximum: float):
        self.minimum = minimum
        self.maximum = maximum

    def clip(self, x):
        return numpy.clip(x, self.minimum, self.maximum)

    @classmethod
    def full_range(cls):
        return cls(-inf, inf)


def cache_key(image_path: str,
              desired_spacing: Tuple[X, Y, Z] = MADER_SPACING,
              pixel_scaling: Optional[str] = 'divide_by_2k',
              dtype='float32',
              hu_window_min=-inf,
              hu_window_max=inf, ):
    return (image_path,
            desired_spacing,
            pixel_scaling,
            dtype,
            hu_window_min,
            hu_window_max)


@cachetools.cached(cache=spaced_volume_ram_cache)
@spaced_volume_cache.cache()
def spaced_ct_volume(image_path: str,
                     desired_spacing: Tuple[X, Y, Z] = MADER_SPACING,
                     pixel_scaling: Optional[str] = 'divide_by_2k',
                     dtype='float32',
                     hu_window_min=-inf,
                     hu_window_max=inf):
    """
    Returns a volume containing the whole CT image in spacing (3,1,1) and coordinate order Z, Y, X
    """
    # This coordinate order Z, Y, X for the image and X,Y,Z for the spacing
    # this spacing and volume is load of the client patient list
    volume, spacing = hiwi.load_image(path=image_path,
                                      spacing=True,
                                      meta=False)

    # if we want to use 2d files, we have to expand the dimension of the volume
    if len(spacing) == 2:
        volume = numpy.expand_dims(volume, axis=2)

    if len(spacing) == 2:
        pass
        # With image.spacing or image['spacing'] you get the spacing in the hiwi.ImageList, but this is the original
        # spacing of the tiff file. The desired spacing for instance is (50mm/224px,40mm/224px) and that is not the
        # spacing what is stored in the hiwi.ImageList. Hopefully i have it correctly understood.
        # assert (desired_spacing[0],) + tuple(spacing) == desired_spacing
    else:
        if numpy.greater(tuple(spacing), desired_spacing).any():
            warn('WARNING: scaling from low resolution to high resolution.')

    if numpy.not_equal(tuple(spacing), desired_spacing).any():
        scipy_dtype = 'int16' if dtype == 'float16' else dtype  # sadly scipy.ndimage.zoom does not work with float16
        volume = volume.astype(scipy_dtype, copy=False)
        spacing_ratio = numpy.divide(spacing, desired_spacing)[::-1]
        spacing_ratio = numpy.maximum(spacing_ratio,
                                      numpy.divide(1, volume.shape))  # prevent zooming out so far that the original image shrinks to less than one pixel
        volume = scipy.ndimage.zoom(volume, order=1, zoom=spacing_ratio, prefilter=False, output=scipy_dtype)
    assert volume.size > 0
    volume = volume.astype(dtype, copy=False)

    hu_window = HUWindow(hu_window_min, hu_window_max)
    volume = hu_window.clip(volume)

    volume = scale_pixels(volume, pixel_scaling)

    # Coordinate order Z, Y, X
    return volume


@spaced_volume_shape_cache.cache()
def spaced_volume_shape(image_path: str,
                        dtype='float16',
                        desired_spacing: Tuple[X, Y, Z] = MADER_SPACING,
                        hu_window_min=-inf,
                        hu_window_max=inf,
                        pixel_scaling=None):
    return spaced_ct_volume(image_path=image_path,
                            desired_spacing=desired_spacing,
                            pixel_scaling=pixel_scaling,
                            hu_window_min=hu_window_min,
                            hu_window_max=hu_window_max,
                            dtype=dtype).shape


def scale_pixels(volume: numpy.ndarray, pixel_scaling: str):
    """
    :param volume: an array of voxels, for example a CT image or a patch
    :param pixel_scaling:
        either
            "range" to scale values to the range from -1 to 1
            "range01" to scale values to the range from 0 to 1
            "divide" to divide values by 256
            "divide_by_2k" to divide values by 2048
            "torch_imagenet" to subtract 0.485 from the first color channel, then divide by 0.229
            "None" to do nothing
    :return:
        the modified array
    """
    if pixel_scaling == 'range':
        volume -= numpy.min(volume)  # minimum is 0 now
        volume /= numpy.max(volume)  # maximum is 1 now
        volume -= 0.5  # range is -0.5 to 0.5 now
        volume *= 2  # range is -1 to 1 now
        assert numpy.min(volume) == -1
        assert numpy.max(volume) == 1
    elif pixel_scaling == 'range01':
        volume = volume.astype('float16', copy=True)
        padding_idx = padding_mask(volume)
        non_padding_idx = ~padding_idx
        volume[padding_idx] = 0
        if not padding_idx.all():  # could happen if a localizer outputs a coordinate far out of bounds
            volume[non_padding_idx] -= numpy.min(volume[non_padding_idx])  # minimum is 0 now
            volume[non_padding_idx] /= numpy.max(volume[non_padding_idx])  # maximum is 1 now
            assert numpy.min(volume) == 0, numpy.min(volume)
            assert numpy.max(volume) == 1, numpy.max(volume)
    elif pixel_scaling == 'divide':
        volume /= 256
    elif pixel_scaling == 'divide_by_2k':
        volume /= 2048
    elif pixel_scaling == 'torch_imagenet':
        volume = scale_pixels(volume, 'range01')
        volume[..., 0] -= 0.485
        # volume[..., 1] -= 0.456
        # volume[..., 2] -= 0.406
        volume[..., 0] /= 0.229
        # volume[..., 1] /= 0.224
        # volume[..., 2] /= 0.225
    elif pixel_scaling is None:
        pass
    else:
        raise NotImplementedError(pixel_scaling)
    return volume


def padding_mask(volume):
    padding_indices = tuple(padding_columns(volume, axis)
                            for axis in range(len(volume.shape)))
    for indices in padding_indices:
        assert str(indices.dtype) == 'bool'
    non_padding_indices = tuple(~padding_columns(volume, axis)
                                for axis in range(len(volume.shape)))
    non_padding_idx = numpy.ix_(*non_padding_indices)
    # padding_idx = numpy.ix_(*padding_indices)
    result = numpy.ones(volume.shape, dtype=bool)
    result[non_padding_idx] = 0
    return result


def padding_columns(im, axis) -> numpy.ndarray:
    im = numpy.moveaxis(im, axis, 0)
    padding_rows = []
    for i in range(im.shape[0]):
        padding_rows.append(len(numpy.unique(im[i])) <= 2)
    return numpy.array(padding_rows, dtype=bool)


def replace_padding_in_array(im: numpy.ndarray, axes=None, replace_with=0.):
    if axes is None:
        axes = range(len(im.shape))  # all axes
    for axis in axes:
        padding_rows = padding_columns(im, axis)
        im = numpy.moveaxis(im, axis, 0)
        im[padding_rows] = replace_with
        im = numpy.moveaxis(im, 0, axis)
    return im


def remove_padding_from_array(im: numpy.ndarray, axes=None):
    if axes is None:
        axes = range(len(im.shape))  # all axes
    for axis in axes:
        padding_rows = padding_columns(im, axis)
        im = numpy.moveaxis(im, axis, 0)
        im = im[~padding_rows]
        im = numpy.moveaxis(im, 0, axis)
    return im
