import functools
import random
from math import inf

import cachetools
from cachetools import LRUCache

from lib.tuned_cache import TunedMemory
from typing import Tuple, List, Union, Optional

import hiwi
import numpy

from lib.util import EBC
from load_data.cylindrical_cropping import pad_cylindrical
from load_data.spaced_ct_volume import spaced_ct_volume, scale_pixels, spaced_volume_shape, spaced_volume_ram_cache, cache_key, HUWindow
from load_data import load_image, MADER_SPACING

X = Y = Z = float

vertebra_volume_cache = TunedMemory(location='./.cache/vertebra_volume_cache', verbose=0)

# for these scalign methods it does not make a difference if we crop first or apply the scaling first
# Because it is more efficient, the scaling for these is applied to whole images
# Actually for range01 it would make a difference, but we do want the cropping to happen on image level
ALLOW_PRESCALE = [None, 'divide_by_2k', 'divide']


def fill_with_crop(img, pos: Tuple[int, ...], crop):
    """
    Fills `crop` with values from `img` at `pos`,
    while accounting for the crop being off the edge of `img`.
    *Note:* negative values in `pos` are interpreted as-is, not as "from the end".
    """
    img_shape, pos, crop_shape = numpy.array(img.shape), numpy.array(pos), numpy.array(crop.shape),
    end = pos + crop_shape

    # Calculate crop slice positions
    crop_low = numpy.clip(0 - pos, a_min=0, a_max=crop_shape)
    crop_high = crop_shape - numpy.clip(end - img_shape, a_min=0, a_max=crop_shape)
    crop_slices = tuple(slice(low, high) for low, high in zip(crop_low, crop_high))
    pos = numpy.clip(pos, a_min=0, a_max=img_shape)
    end = numpy.clip(end, a_min=0, a_max=img_shape)
    img_slices = tuple(slice(low, high) for low, high in zip(pos, end))
    crop[crop_slices] = img[img_slices]


def crop_with_pad(x, pos: Union[Tuple[int, ...], str], shape: Tuple[int, ...], pad_value=0, dtype='float32'):
    """
    :param x: array to crop
    :param pos: position where to crop or the string "center"
    :param shape: size of the crop
    :param pad_value: padding if outside of array
    :param dtype: data type
    :return: cropped array, filled with pad_value where no voxel value available
    """
    if pos == 'center':
        pos = (numpy.array(x.shape) - shape) // 2
    result = numpy.full(shape=shape, fill_value=pad_value, dtype=dtype)
    fill_with_crop(img=x, pos=pos, crop=result)
    assert result.shape == shape
    return result


def pad_without_crop(x, shape: Tuple[int, ...], center, pad_value=0, dtype='float32'):
    if center:
        pos = (numpy.array(x.shape) - shape) // 2
    else:
        pos = numpy.zeros_like(x.shape)
    return crop_with_pad(x, pos=pos, shape=shape, pad_value=pad_value, dtype=dtype)


def crop_then_pad(x, crop_pos, crop_shape: Tuple[int, ...], pad_shape: Tuple[int, ...], pad_value=0, dtype='float32'):
    cropped = crop_with_pad(x, crop_pos, crop_shape, pad_value, dtype)
    padded = pad_without_crop(cropped, pad_shape, True, pad_value, dtype)
    return padded


@functools.lru_cache()
def image_list_by_name(image_list_name):
    return hiwi.ImageList.load(image_list_name)


class MissingVertebra(ValueError):
    pass


_warned = False

relative_vertebra_size = {
    'C1': 0.7,
    'C2': 0.7,
    'C3': 0.6,
    'C4': 0.6,
    'C5': 0.6,
    'C6': 0.6,
    'C7': 0.6,
    'T1': 0.75,
    'T2': 0.75,
    'T3': 0.8,
    'T4': 0.8,
    'T5': 0.8,
    'T6': 0.9,
    'T7': 0.9,
    'T8': 0.9,
    'T9': 0.9,
    'T10': 1,
    'T11': 1.1,
    'T12': 1.1,
    'L1': 1.1,
    'L2': 1.2,
    'L3': 1.2,
    'L4': 1.1,
    'L5': 1.2,
    'L6': 1.2,
}


@vertebra_volume_cache.cache(ignore=['image', 'hu_window'])
def vertebra_volume(image: hiwi.Image,
                    vertebra: str,
                    desired_size_mm: Tuple[X, Y, Z],
                    vertebra_size_adjustment=None,
                    pixel_scaling='divide_by_2k',
                    cylindrical_crop_probability=0,
                    pad_value=0,
                    desired_spacing: Tuple[X, Y, Z] = MADER_SPACING,
                    dtype='float16',
                    hu_window: HUWindow = HUWindow.full_range(),
                    _additional_cache_key=None):
    if vertebra_size_adjustment not in [None, 'pad', 'zoom']:
        raise ValueError(vertebra_size_adjustment)
    if vertebra_size_adjustment == 'zoom':
        desired_spacing = tuple(mm * relative_vertebra_size[vertebra] for mm in desired_spacing)
        desired_size_mm = tuple(mm * relative_vertebra_size[vertebra] for mm in desired_size_mm)

    crop_position_px, crop_size_px = vertebra_crop_box(image,
                                                       vertebra,
                                                       desired_size_mm,
                                                       pixel_scaling,
                                                       hu_window_min=hu_window.minimum,
                                                       hu_window_max=hu_window.maximum,
                                                       dtype=dtype,
                                                       desired_spacing=desired_spacing,
                                                       _additional_cache_key=_additional_cache_key, )

    if crop_position_px is None:
        # this can happen if a localizer did not find a vertebra that is actually in the image
        # this should only happen during evaluation, not training
        return numpy.full(shape=tuple(int(round(x)) for x in crop_size_px), fill_value=numpy.nan, dtype=dtype)
    visible_vertebrae = num_vertebra_in_bbox(image, crop_position_px, crop_size_px, bbox_spacing=desired_spacing)
    assert visible_vertebrae > 0

    pad_size_px = crop_size_px
    if vertebra_size_adjustment == 'pad':
        if relative_vertebra_size[vertebra] < 1:
            crop_position_px = tuple(p + s * (1 - relative_vertebra_size[vertebra]) // 2 for s, p in zip(crop_size_px, crop_position_px))
            crop_size_px = tuple(px * relative_vertebra_size[vertebra] for px in crop_size_px)

    # coordinate order Z, Y, X
    bypass_caches = vertebra_size_adjustment == 'zoom' and relative_vertebra_size[vertebra] < 1
    scv = spaced_ct_volume.func if bypass_caches else spaced_ct_volume
    pixel_scaling, prescale = split_scaling_parameters(pixel_scaling)
    volume: numpy.ndarray = scv(str(image.path),
                                desired_spacing=desired_spacing,
                                hu_window_min=hu_window.minimum,
                                hu_window_max=hu_window.maximum,
                                pixel_scaling=pixel_scaling if prescale else None,
                                dtype=dtype)  # pixel_scaling is applied later in this method

    # choose a value for padding (for rotation or cropping or both)
    cylindrical_pad_value = random.choice([0, -1000, -2000, -3000])  # in HU

    # random cylindrical cropping
    if random.random() < cylindrical_crop_probability:
        pad_cylindrical(volume, cylinder_axis=0, pad_value=cylindrical_pad_value)

    assert volume.shape == spaced_volume_shape(str(image.path),
                                               desired_spacing=desired_spacing,
                                               pixel_scaling=pixel_scaling if prescale else None,
                                               hu_window_min=hu_window.minimum,
                                               hu_window_max=hu_window.maximum,
                                               dtype=dtype)

    crop_position_px = tuple(int(round(x)) for x in crop_position_px)
    crop_size_px = tuple(int(round(x)) for x in crop_size_px)
    pad_size_px = tuple(int(round(x)) for x in pad_size_px)
    volume = crop_then_pad(volume,
                           crop_pos=crop_position_px,
                           crop_shape=crop_size_px,
                           pad_shape=pad_size_px,
                           pad_value=pad_value,
                           dtype=dtype)

    if not prescale:
        volume = scale_pixels(volume, pixel_scaling)

    assert volume.shape == pad_size_px, (volume.shape, pad_size_px)
    return volume


vertebra_crop_box_cache = LRUCache(maxsize=2000)  # typically one crop box per vertebra, no need to load from disk


def vertebra_crop_box_cache_key(image: hiwi.Image,
                                vertebra,
                                desired_size_mm: Tuple[X, Y, Z],
                                pixel_scaling=None,
                                desired_spacing=MADER_SPACING,
                                dtype='float16',
                                hu_window_min=-inf,
                                hu_window_max=inf,
                                _additional_cache_key=None):
    return (
        vertebra,
        desired_size_mm,
        pixel_scaling,
        dtype,
        desired_spacing,
        image['patient_id'],
        tuple(image.spacing),
        len(image.parts),
    )


@cachetools.cached(cache=vertebra_crop_box_cache, key=vertebra_crop_box_cache_key)
def vertebra_crop_box(image: hiwi.Image,
                      vertebra,
                      desired_size_mm: Tuple[X, Y, Z],
                      pixel_scaling=None,
                      desired_spacing=MADER_SPACING,
                      dtype='float16',
                      hu_window_min=-inf,
                      hu_window_max=inf,
                      _additional_cache_key=None) -> Tuple[Optional[Tuple[Z, Y, X]], Tuple[Z, Y, X]]:
    # original spacing of the image
    spacings = tuple(image.spacing)

    # convert everything to coordinate order Z, Y, X
    desired_size_mm: Tuple[Z, Y, X] = desired_size_mm[::-1]
    desired_spacing_zyx: Tuple[Z, Y, X] = desired_spacing[::-1]

    # compute patch size in pixels from spacing and patch size in mm
    desired_size_px: Tuple[Z, Y, X] = tuple(size / spacing
                                            for size, spacing
                                            in zip(desired_size_mm, desired_spacing_zyx))
    if image.parts[vertebra].position is None:
        return None, desired_size_px

    # expand the dimension of the center point and the spacing if it is necessary
    if len(spacings) == 2:
        center_px: Tuple[X, Y, Z] = (0, image.parts[vertebra].position[0], image.parts[vertebra].position[1])
        spacings = (3, image.spacing[0], image.spacing[1])
    else:
        center_px: Tuple[X, Y, Z] = tuple(image.parts[vertebra].position)
    center_px = numpy.multiply(center_px, spacings) / desired_spacing

    center_px: Tuple[Z, Y, X] = center_px[::-1]

    pixel_scaling, prescale = split_scaling_parameters(pixel_scaling)

    # check the shape of the volume
    volume_shape = spaced_volume_shape(str(image.path),
                                       dtype=dtype,
                                       desired_spacing=desired_spacing,
                                       hu_window_max=hu_window_max,
                                       hu_window_min=hu_window_min,
                                       pixel_scaling=pixel_scaling if prescale else None)

    # expand the dimension if it is necessary
    if len(image.parts[vertebra].position) == 2:
        center_px += (0,)
        volume_shape += (1,)

    crop_position_px: Tuple[Z, Y, X] = tuple((center - size / 2)
                                             for center, size in zip(center_px, desired_size_px))
    return crop_position_px, desired_size_px


def split_scaling_parameters(pixel_scaling):
    if pixel_scaling.endswith('_prescale'):
        pixel_scaling = pixel_scaling[:-len('_prescale')]
        prescale = True
    else:
        prescale = pixel_scaling in ALLOW_PRESCALE
    return pixel_scaling, prescale


def in_bbox(coordinate_px, bbox_position_px, bbox_size_px, bbox_spacing: Tuple[X, Y, Z], coordinate_spacing: Tuple[X, Y, Z]):
    bbox_size_px = numpy.multiply(bbox_size_px, bbox_spacing[::-1]) / coordinate_spacing[::-1]
    bbox_position_px = numpy.multiply(bbox_position_px, bbox_spacing[::-1]) / coordinate_spacing[::-1]
    for c, p, s in zip(coordinate_px, bbox_position_px, bbox_size_px):
        if not p <= c < p + s:  # in range does not work here because there might be floats involved
            # for example if position is 3 and size is 4 then the interval is [2.5, 6.5)
            return False
    return True


def num_sq_fracture_in_bbox(image: hiwi.Image,
                            bbox_position_px: Tuple[Z, Y, X],
                            bbox_size_px: Tuple[Z, Y, X],
                            bbox_spacing: Tuple[X, Y, Z], ):
    return sum(1 for vertebra in image.parts.values()
               if vertebra['Genant Score'] > 0
               and in_bbox(vertebra.position[::-1], bbox_position_px, bbox_size_px, bbox_spacing, coordinate_spacing=image.spacing))


def num_vertebra_in_bbox(image: hiwi.Image,
                         bbox_position_px: Tuple[Z, Y, X],
                         bbox_size_px: Tuple[Z, Y, X],
                         bbox_spacing: Tuple[X, Y, Z], ):
    count = sum(1 for vertebra in image.parts.values()
                if vertebra.position is not None
                if in_bbox(vertebra.position[::-1], bbox_position_px, bbox_size_px, bbox_spacing, coordinate_spacing=image.spacing))
    return count
