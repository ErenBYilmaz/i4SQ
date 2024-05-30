import itertools
import os
import random
from math import sqrt
from typing import List, Optional, Union

import SimpleITK
import matplotlib.image
import numpy
from matplotlib import pyplot

from lib.my_logger import logging
from lib.progress_bar import ProgressBar
from lib.util import EBC, shorten_name
from load_data.patch_request import PatchRequest
from tasks import VertebraTasks

FLOATX = 'float32'


class AnnotatedBatch(EBC):

    def __init__(self, xs: Optional[List[numpy.ndarray]],
                 y_trues: List[Union[numpy.ndarray, List[Optional[numpy.ndarray]]]],
                 sample_weights: List[numpy.ndarray],
                 names: List[str],
                 tasks: VertebraTasks,
                 patch_requests: List[PatchRequest], ):
        self.patch_requests = patch_requests
        self.xs = xs
        self.y_trues = y_trues
        self.sample_weights = sample_weights
        self.names = names
        self.tasks = tasks
        self.num_unlabeled = {task_idx: 0 for task_idx in range(len(self.tasks))}

    def fully_built(self):
        return self.xs is not None and all(isinstance(y, numpy.ndarray) for y in self.y_trues)

    def batch_size(self):
        return len(self.names)

    def check_batch_size(self, expected_size=None):
        if expected_size is None:
            expected_size = self.batch_size()
        assert len(self.names) == expected_size
        assert all(b.shape[0] == expected_size for b in self.xs + self.y_trues)

    def check_num_tasks(self, expected_size=None):
        if expected_size is None:
            expected_size = len(self.tasks)
        assert len(self.tasks) == expected_size
        assert len(self.y_trues) == expected_size
        assert len(self.num_unlabeled) == expected_size
        assert len(self.sample_weights) == expected_size

    @staticmethod
    def initialize_new_batch(batch_size, tasks: VertebraTasks, patch_requests: List[PatchRequest]) -> 'AnnotatedBatch':
        return AnnotatedBatch(xs=None,
                              y_trues=[[None for _ in range(batch_size)] for _ in tasks],
                              sample_weights=[numpy.zeros((batch_size,), dtype=FLOATX) for _ in range(len(tasks))],
                              names=['' for _ in range(batch_size)],
                              tasks=tasks,
                              patch_requests=patch_requests)

    def as_tuple_for_keras(self):
        for y_idx, y in enumerate(self.y_trues):
            if not isinstance(y, numpy.ndarray):
                raise ValueError(y)
        if self.xs is None:
            raise ValueError(self.xs)
        for a in itertools.chain(self.xs, self.y_trues, self.sample_weights):
            if not numpy.issubdtype(a.dtype, numpy.floating):
                raise RuntimeError(f'Please provide batch entries in a floating format (ideally {FLOATX}, '
                                   f'was {a.dtype})')
        return (self.xs, self.y_trues, self.sample_weights)

    def check_no_nan_in_y_trues(self):
        # check for nan values
        assert all(not numpy.isnan(numpy.sum(b)) for b in self.y_trues)

    def adjust_sample_weights_for_unlabeled_vertebrae(self):
        for task_idx in range(len(self.tasks)):
            # adjust sample weights for missing values, so that we are still in a similar range of values even if unlabeled samples are added
            num_labeled_this_batch = (self.batch_size() - self.num_unlabeled[task_idx])
            if num_labeled_this_batch == 0:
                logging.warning(f'Batch did not contain any labeled vertebrae for task {self.tasks[task_idx].output_layer_name()}')
                continue
            self.sample_weights[task_idx] *= self.batch_size() / num_labeled_this_batch

    def serialize(self, generate_images=True, save_as_nifti=True, generate_histograms=False, relative_dir_name: str = None, progress_bar=False, verbose=False):
        if relative_dir_name is None:
            relative_dir_name = str(random.randrange(200000))
        full_dirname = os.path.abspath(f'img/generated/batch_contents/{relative_dir_name}')
        dirname_link = os.path.abspath(full_dirname).replace('\\', '/')
        print(f'Writing batch of size {self.batch_size()} to file:///{dirname_link}')
        idx_patch_idx = [(input_idx, input_patch, sample_idx) for input_idx, input_patch in enumerate(self.xs) for sample_idx in range(input_patch.shape[0])]
        if progress_bar:
            idx_patch_idx = ProgressBar(idx_patch_idx)
        for input_idx, input_patch, sample_idx in idx_patch_idx:
            volume = input_patch[sample_idx, ..., 0]
            is_2d = len(volume.shape) == 2
            if is_2d:
                volume = volume[..., numpy.newaxis]  # artificial lateral axis
            assert len(volume.shape) == 3

            patient_number, vertebra = self.names[sample_idx][1:-1].replace("'", "").split(', ')

            os.makedirs(full_dirname, exist_ok=True)
            pnr = shorten_name(patient_number)
            filepath = f'{full_dirname}/{pnr}_{shorten_name(vertebra)}_{input_idx}.png'
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            if generate_images and not os.path.isfile(filepath):
                image_lateral_axis_length = volume.shape[2]
                matplotlib.image.imsave(filepath, volume[:, :, image_lateral_axis_length // 2], cmap='gray')
                if verbose:
                    print(f'Wrote {filepath}')

            if save_as_nifti:
                nii_path = filepath.replace('.png', '.nii.gz')
                if not os.path.isfile(nii_path):
                    img_itk: SimpleITK.Image = SimpleITK.GetImageFromArray(volume)
                    img_itk.SetSpacing(self.patch_requests[input_idx].spacing)
                    SimpleITK.WriteImage(img_itk, nii_path)
                    if verbose:
                        logging.info('Wrote ' + nii_path)

            if generate_histograms:
                hist_path = filepath.replace('.png', '_hist.png')
                if not os.path.isfile(hist_path):
                    bins = 100
                    pyplot.hist([volume.flatten()], bins=bins)
                    pyplot.legend(['pixel distribution for {1}∕{2} of {0}'.format(patient_number, vertebra, input_idx)])
                    patch_size_px = volume.size
                    # pyplot.xlim(-2, 2)
                    pyplot.ylim(0, patch_size_px // sqrt(bins))
                    pyplot.savefig(hist_path)
                    pyplot.clf()
                    pyplot.close()
                    if verbose:
                        print(f'Wrote {hist_path}')

    def check_patch_size(self):
        assert len(self.patch_requests) == len(self.xs)
        for patch_request, x in zip(self.patch_requests, self.xs):
            assert x.shape[1:(1 + len(patch_request.size_px()))] == patch_request.size_px()
