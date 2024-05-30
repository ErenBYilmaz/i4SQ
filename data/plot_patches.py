import os
import sys
from math import inf
from typing import Optional, List, Tuple

import SimpleITK
import numpy
from matplotlib import pyplot

from lib.progress_bar import ProgressBar
from lib.util import EBE, Z, Y, X
from load_data.generate_annotated_vertebra_patches import AnnotatedPatchesGenerator
from model.fnet.const import TASKS


def filepath_for_vertebra_name(to_dir: str, name: str):
    patient_id, vertebra = eval(name)
    filepath = os.path.join(to_dir, f'{patient_id}_{vertebra}.png')
    return filepath


class DontSave(EBE):
    DONT_SAVE = 'DONT_SAVE'


def plot_patches_in_batch(x, names,
                          to_dir='img/generated/vertebra_patches',
                          only_names: Optional[List[str]] = None,
                          generate_sagittal_images=True,
                          generate_axial_images=True,
                          save_as_nifti=False,
                          spacing: Tuple[X, Y, Z] = None,
                          filepath_creator=filepath_for_vertebra_name,
                          generate_histograms=True,
                          skip_existing=True,
                          with_progress_bar=False):
    n_samples = x.shape[0]

    os.makedirs(to_dir, exist_ok=True)
    if with_progress_bar:
        samples = ProgressBar(n_samples, suffix='Plotting patches')
    else:
        samples = range(n_samples)
    for sample_idx in ProgressBar(n_samples):
        name = names[sample_idx]
        if only_names is not None and name not in only_names:
            continue
        patient_id, vertebra = eval(name)
        volume = x[sample_idx, ..., 0]
        is_2d = len(volume.shape) == 2
        if is_2d:
            volume = volume[..., numpy.newaxis]  # artificial lateral axis
        assert len(volume.shape) == 3

        volume = volume[::-1]  # flip up down
        filepath = filepath_creator(to_dir, name)
        if filepath is DontSave.DONT_SAVE:
            continue
        if not filepath.endswith('.png'):
            filepath += '.png'
        if generate_sagittal_images:
            if not skip_existing or not os.path.isfile(filepath):
                image_lateral_axis_length = volume.shape[2]
                shape_before: Tuple[Z, Y] = (volume.shape[0], volume.shape[1])
                import skimage.transform
                resized = skimage.transform.resize(volume[:, :, image_lateral_axis_length // 2], (round(shape_before[0] * spacing[2]), round(shape_before[1] * spacing[1])))
                pyplot.imsave(filepath, resized, cmap='gray')

        if generate_axial_images:
            if spacing is None:
                raise ValueError
            if not skip_existing or not os.path.isfile(filepath.replace('.png', '_axial.png')):
                image_lateral_axis_length = volume.shape[0]
                shape_before: Tuple[Y, X] = (volume.shape[1], volume.shape[2])
                import skimage.transform
                resized = skimage.transform.resize(volume[image_lateral_axis_length // 2, :, :], (shape_before[0] * spacing[1], shape_before[1] * spacing[0]))
                pyplot.imsave(filepath.replace('.png', '_axial.png'), resized, cmap='gray')

        if save_as_nifti:
            if spacing is None:
                raise ValueError
            if not skip_existing or not os.path.isfile(filepath.replace('.png', '.nii.gz')):
                img_itk: SimpleITK.Image = SimpleITK.GetImageFromArray(volume)
                img_itk.SetSpacing(spacing)
                SimpleITK.WriteImage(img_itk, filepath.replace('.png', '.nii.gz'))

        if generate_histograms:
            if not skip_existing or not os.path.isfile(filepath.replace('.png', '_hist.png')):
                flat = volume.flatten()
                weights = numpy.ones_like(flat) / flat.size
                pyplot.hist([flat], bins=50, weights=weights)
                pyplot.legend([f'HU distribution for {vertebra} of {patient_id}'])
                pyplot.xlim(-2, 2)
                pyplot.ylim(0, 0.4)
                pyplot.savefig(filepath.replace('.png', '_hist.png'))
                pyplot.clf()
