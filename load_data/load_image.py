import random
from math import ceil
from typing import Optional, Dict, Union

import SimpleITK
import cachetools
import numpy

import hiwi
from lib.tuned_cache import TunedMemory
from load_data.plane import Plane

cache = TunedMemory('.cache')


def image_by_patient_id(patient_id: str, source_image_list: hiwi.ImageList) -> hiwi.Image:
    cache = image_by_patient_id.cache
    try:
        return cache[id(source_image_list)][patient_id]
    except KeyError:
        image_by_patient_id.misses += 1
        image: Optional[hiwi.Image] = None
        for image in source_image_list:
            if image['patient_id'] == patient_id:
                break
        else:
            raise ImageNotInListError(patient_id)

        cache_size = sum(map(len, cache.values()))
        if cache_size > 4000:  # keep cache small
            print(f'Clearing randomly half of the image_by_patient_id cache, '
                  f'{image_by_patient_id.requests - image_by_patient_id.misses} hits total, '
                  f'{image_by_patient_id.misses} misses total')
            for k in list(cache.keys()):
                for k2 in random.choices(list(cache[k].keys()), k=ceil(len(cache[k]) / 2)):
                    if k not in cache:
                        continue
                    if k2 in cache[k]:
                        del cache[k][k2]
                    elif len(cache[k]) == 0:
                        del cache[k]
            assert sum(map(len, cache.values())) < cache_size

        cache.setdefault(id(source_image_list), {})[patient_id] = image
        return image
    finally:
        image_by_patient_id.requests += 1


# noinspection PyTypeHints
image_by_patient_id.cache: Dict[int, Dict[Union[str, tuple], hiwi.Image]] = {}
image_by_patient_id.requests = 0
image_by_patient_id.misses = 0


def vertebra_contained(patient_id: str, vertebra_name: str, source_image_list: hiwi.ImageList) -> bool:
    return vertebra_name in image_by_patient_id(patient_id=patient_id, source_image_list=source_image_list).parts


class ImageNotInListError(ValueError):
    pass


@cache.cache()
def load_image_with_spacing(image_path, desired_spacing):
    img = load_image_with_spacing_corrections(image_path)
    img = convert_image_to_spacing(img, desired_spacing)
    return img


def convert_image_to_spacing(img: SimpleITK.Image, spacing):
    if spacing is not None:
        if numpy.not_equal(tuple(img.GetSpacing()), spacing).any():
            r = SimpleITK.ResampleImageFilter()
            r.SetInterpolator(SimpleITK.sitkBSpline)
            r.SetOutputSpacing(spacing)
            r.SetOutputDirection(img.GetDirection())
            r.SetOutputOrigin(img.GetOrigin())
            scaling = numpy.array(img.GetSpacing()) / spacing
            r.SetSize(numpy.ceil(img.GetSize() * scaling).astype(int).tolist())
            r.SetOutputPixelType(img.GetPixelID())
            img = r.Execute(img)
    assert numpy.prod(img.GetSize()) != 0
    return img


def load_image_with_spacing_corrections(image_path):
    img = SimpleITK.ReadImage(image_path)
    apply_spacing_corrections(image_path, img)
    return img


def apply_spacing_corrections(str_containing_study_description, img):
    spacing = list(img.GetSpacing())
    try:
        from annotation_supplement_ey import spacing_corrections
    except ImportError:
        return
    for study, corrections in spacing_corrections.items():
        if study in str_containing_study_description:
            for axis in corrections:
                spacing[Plane.from_axis(axis).axis_xyz()] = corrections[axis]
    img.SetSpacing(spacing)


def apply_direction_corrections(str_containing_study_description, img):
    try:
        from annotation_supplement_ey import upside_down_image_and_annotations
    except ImportError:
        return img
    for study in upside_down_image_and_annotations:
        if study in str_containing_study_description:
            img = SimpleITK.Flip(img, [False, False, True])
    return img


@cachetools.cached(cache=cachetools.LRUCache(2000))
def coordinate_transformer_from_file(img_path, patient_id, spacing=None) -> SimpleITK.Image:
    r = SimpleITK.ImageFileReader()
    r.SetFileName(str(img_path))
    r.LoadPrivateTagsOn()
    r.ReadImageInformation()
    img = SimpleITK.GetImageFromArray(numpy.zeros((1, 1, 1), dtype='float32'))
    img.SetSpacing(r.GetSpacing())
    img.SetDirection(r.GetDirection())
    img.SetOrigin(r.GetOrigin())
    if spacing is not None:
        img = convert_image_to_spacing(img, spacing)
    apply_spacing_corrections(patient_id, img)
    return img
