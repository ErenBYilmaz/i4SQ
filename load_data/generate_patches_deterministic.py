import math
import os
import sys
from threading import RLock

from ct_dirs import DERIVED_PATH
from load_data import crop
from lib.prime_math import get_prime_factors
from lib.progress_bar import ProgressBar

import hiwi
import matplotlib.image
import matplotlib.pyplot as plt
import numpy
from keras import backend

from lib.shorten_name import shorten_name
from load_data.generate_full_volumes import FullVolumesGenerator, powerset


def get_set_of_patch_indices(start, stop, step):
    return numpy.asarray(numpy.mgrid[start[0]:stop[0]:step[0], start[1]:stop[1]:step[1],
                         start[2]:stop[2]:step[2]].reshape(3, -1).T, dtype=numpy.int)


def compute_patch_indices(image_shape, patch_size, overlap, start=None):
    if isinstance(overlap, int):
        overlap = numpy.asarray([overlap] * len(image_shape))
    if start is None:
        n_patches = numpy.ceil(image_shape / (patch_size - overlap))
        overflow = (patch_size - overlap) * n_patches - image_shape + overlap
        start = -numpy.ceil(overflow / 2)
    elif isinstance(start, int):
        start = numpy.asarray([start] * len(image_shape))
    stop = image_shape + start
    step = patch_size - overlap
    return get_set_of_patch_indices(start, stop, step)


class DeterministicPatchesGenerator:
    def __init__(self,
                 image_list: hiwi.ImageList,
                 batch_size,
                 patch_size,
                 pixel_scaling='divide_by_2k',
                 cache_batches=True,
                 mixup_rate=0,
                 auto_tune_batch_size=False,
                 overlap=0,):
        """
        :param image_list: the dataset
        :param batch_size: number of CT images per batch
        :param patch_size: size of individual patches, of they extend outside of the CT image, 0 padding is used
        :param pixel_scaling: how to scale the values of the individual pixels. this is passed to the "scale_pixels" method
        :param cache_batches:
            whether or not to cache individual batches in RAM. can be helpful for validation sets that do not need data augmentation
            Also, cache size in "number of batches x batch size" will be printed once if this is True.
            Keep in mind that this will prevent random data augmentation the second time a batch containing the same vertebrae is
            requested (should not be a problem for validation and testing, if you don't use data augmentation for that)
        :param mixup_rate:
            alpha parameter for mixup. not recommended
        :param auto_tune_batch_size:
            if True, the batch size will be automatically determined so that it divides the dataset size, i.e.
            the batch size will be chosen such that it is as large as possible, but still smaller or equal to the given batch size,
            such that self.steps_per_epoch() * self.batch_size == self.num_samples()
        :param overlap:
            3 integers stating how much the patches should overlap with the adjacent patches or a single integer that is used for
            all 3 dimensions
        """
        self.overlap = overlap
        self.batch_cache = {}
        self.patch_size = patch_size
        self.backend = FullVolumesGenerator(image_list=image_list,
                                            batch_size=1,
                                            random_mode=None,
                                            weighted=False,
                                            pixel_scaling=pixel_scaling,
                                            random_flip_lr=False,
                                            cache_batches=False,
                                            mixup_rate=mixup_rate,
                                            auto_tune_batch_size=auto_tune_batch_size,
                                            rotate_range_deg=0,
                                            cylindrical_crop_probability=0,
                                            random_crop=None,
                                            task='segmentation')
        self.generator_lock = RLock()
        self.cache_batches = cache_batches
        self.step_count = 0
        self._data_order_cache = []
        for image in image_list:
            for patch in compute_patch_indices(image.data.shape,
                                               self.patch_size,
                                               self.overlap, ):
                self._data_order_cache.append((image['patient_id'], patch))
        self.batch_size: int = -1
        self.tune_batch_size(batch_size)
        self.backend_call_cache = {}

        if self.cache_batches:
            step = 1
            while step * self.batch_size % self.num_samples() != 0:
                step += 1
            print('{0}: Cache size = {1} x {2}'.format(self, step, self.batch_size))

    def tune_batch_size(self, maximum_batch_size):
        """
        the batch size will be chosen such that it is as large as possible, but still smaller or equal to the current batch size,
        such that self.steps_per_epoch() * self.batch_size == self.num_samples()
        """
        import numpy
        largest_value = 1
        all_factors = list(get_prime_factors(self.num_samples()))
        all_factors = [factor for factor, amount in all_factors for _ in range(amount)]
        for factors in powerset(all_factors):
            product = numpy.prod(factors)
            if largest_value < product <= maximum_batch_size:
                largest_value = product
        self.batch_size = largest_value
        assert self.batch_size * self.steps_per_epoch() == self.num_samples()
        # print(self.batch_size, self.num_samples(), maximum_batch_size)

    def steps_per_epoch(self):
        return math.ceil(self.num_samples() / self.batch_size)

    def __iter__(self):
        with self.generator_lock:
            self.step_count = 0
            return self

    def num_samples(self):
        return len(self._data_order_cache)

    def __next__(self):
        with self.generator_lock:
            # this calls the __getitem__ method
            result = self[self.step_count]
            self.step_count += 1
            return result

    def clear_cache(self):
        self.batch_cache = {}

    def __getitem__(self, step: int):
        with self.generator_lock:
            offset = step * self.batch_size % self.num_samples()
            self.last_offset = offset
            if self.cache_batches and offset in self.batch_cache:
                self.last_batch_names = self.batch_cache[offset][2]
                return self.batch_cache[offset][:2]

            # initialize numpy arrays and name list for the batch
            y1 = numpy.zeros((self.batch_size, 1,), dtype=backend.floatx())  # Labels for fractured or not

            names = ['' for _ in range(self.batch_size)]

            volumes = numpy.empty(shape=(self.batch_size, *self.patch_size, 1), dtype='float16')
            label_maps = numpy.empty(shape=(self.batch_size, *self.patch_size, 3), dtype='int8')
            fracture_labels = numpy.empty(shape=(self.batch_size, 1), dtype='float32')

            batch_idx = 0
            while batch_idx < self.batch_size:  # actually fill the batch
                assert not self.cache_batches or self.batch_size * self.steps_per_epoch() == self.num_samples()
                # chose a vertebra from the dataset
                patient_id, patch_position = self.choose_patch_for_batch(batch_idx, offset)
                if patient_id not in self.backend_call_cache:  # maximum size is 1 item, should be enough
                    self.backend_call_cache = {
                        patient_id: self.backend[patient_id]
                    }
                (volume,), labels, _ = self.backend_call_cache[patient_id]
                label_map = labels['sseg']
                assert volume.shape[0] == 1
                assert volume.shape[-1] == 1
                assert label_map.shape[0] == 1
                assert label_map.shape[-1] == 3

                volume = crop.crop_with_pad(volume,
                                            desired_shape=(1, *self.patch_size, 1),
                                            pad_value=volume[tuple(0 for _ in volume.shape)],
                                            pos=(0, *patch_position, 0), )
                label_map = crop.crop_with_pad(label_map,
                                               desired_shape=(1, *self.patch_size, 3),
                                               pad_value=[1, 0, 0],
                                               dtype='int8',
                                               pos=(0, *patch_position, 0))
                volumes[batch_idx, ...] = volume
                label_maps[batch_idx, ...] = label_map
                # V = 4/3 * π * a * b * c = 31415,927
                # patch positive if at least 50% of a fractured vertebra is visible
                fracture_labels[batch_idx, 0] = numpy.sum(label_maps, axis=(0, 1, 2, 3))[2] > 0.5 * 31415.927

                # the label
                names[batch_idx] = str((patient_id, patch_position))

                batch_idx += 1

            # check that the batch has the desired size
            assert volumes is not None
            assert volumes.shape[0] == self.batch_size
            assert tuple(volumes.shape[1:4]) == tuple(self.patch_size)
            assert y1.shape[0] == self.batch_size
            x = [volumes]
            y = [label_maps, fracture_labels]
            assert not numpy.isnan(numpy.sum(volumes))
            assert not numpy.isnan(numpy.sum(label_maps))
            assert not numpy.isnan(numpy.sum(fracture_labels))

            # compose the batch
            batch = [x, y]

            result = tuple(batch)
            if self.cache_batches:
                # store in cache if desired
                self.batch_cache[offset] = (*batch, names)
            self.last_batch_names = names
            sys.stdout.flush()
            return result

    def choose_patch_for_batch(self, batch_idx, offset):
        return self._data_order_cache[(offset + batch_idx) % self.num_samples()]


def main(generate_images=False, generate_histograms=False, save_as_nifti=False):
    pixel_scalings = [
        'divide_by_2k',
        # 'range01',
        # 'range',
    ]
    for dataset in [hiwi.ImageList.load(os.path.join(DERIVED_PATH, 'spacing_3_1_1', 'clean_patients.iml')), ]:
        for scaling in pixel_scalings:
            print('Now doing', scaling, len(dataset))
            generator = DeterministicPatchesGenerator(dataset,
                                                      patch_size=(128, 128, 64),
                                                      batch_size=1,
                                                      pixel_scaling=scaling,
                                                      cache_batches=False,
                                                      mixup_rate=0,
                                                      )
            # print(vertebra_dataset_size(dataset))
            for _ in ProgressBar(generator.steps_per_epoch()):
                sys.stdout.flush()
                x, y = next(generator)
                names = generator.last_batch_names
                volume = x[0][0, :, :, :, 0]
                assert ((y[0] == 0) | (y[0] == 1)).all()  # check if it contains only ones and zeros
                assert (numpy.sum(y[0], axis=-1) == 1).all()  # check if it has exactly one 1 per voxel
                label_map = y[0][0, :, :, :, :].astype('uint8')
                assert ((label_map == 0) | (label_map == 1)).all()  # check if it contains only ones and zeros
                assert (numpy.sum(label_map, axis=-1) == 1).all()  # check if it has exactly one 1 per voxel

                name = names[0]

                if generate_images:
                    full_dirname = 'img/generated/deterministic_patches/'
                    os.makedirs(full_dirname, exist_ok=True)
                    filename = '{2}/{1}_{0}'.format(shorten_name(scaling),
                                                    shorten_name(name),
                                                    full_dirname)
                    matplotlib.image.imsave(filename + '_mid_sagittal.png',
                                            volume[:, :, volume.shape[2] // 2],
                                            cmap='gray',
                                            vmin=-1,
                                            vmax=1)
                    matplotlib.image.imsave(filename + '_mid_axial.png',
                                            numpy.kron(volume[volume.shape[0] // 2, :, :], numpy.ones((1, 3), dtype=volume.dtype)),
                                            cmap='gray',
                                            vmin=-1,
                                            vmax=1)
                    matplotlib.image.imsave(filename + '_mid_coronal.png',
                                            numpy.kron(volume[:, volume.shape[1] // 2, :], numpy.ones((1, 3), dtype=volume.dtype)),
                                            cmap='gray',
                                            vmin=-1,
                                            vmax=1)
                    matplotlib.image.imsave(filename + '_mid_sagittal_label.png',
                                            label_map[:, :, label_map.shape[2] // 2, :].astype('float16'),
                                            vmin=-1,
                                            vmax=1)
                    matplotlib.image.imsave(filename + '_mid_axial_label.png',
                                            numpy.kron(label_map[label_map.shape[0] // 2, :, :, :], numpy.ones((1, 3, 1), dtype='float16')),
                                            vmin=-1,
                                            vmax=1)
                    matplotlib.image.imsave(filename + '_mid_coronal_label.png',
                                            numpy.kron(label_map[:, label_map.shape[1] // 2, :, :], numpy.ones((1, 3, 1), dtype='float16')),
                                            vmin=-1,
                                            vmax=1)

                if save_as_nifti:
                    raise NotImplementedError
                    # img_itk: SimpleITK.Image = SimpleITK.GetImageFromArray(volume)
                    # os.makedirs('nifti_tmp', exist_ok=True)
                    # SimpleITK.WriteImage(img_itk, os.path.join('nifti_tmp', f'{patient_number}'))
                    #
                if generate_histograms:
                    full_dirname = 'img/generated/deterministic_patches/histograms/'.format()
                    os.makedirs(full_dirname, exist_ok=True)
                    filename = '{2}/{1}_{0}.png'.format(shorten_name(scaling),
                                                        shorten_name(name),
                                                        full_dirname)
                    plt.hist([x[0].flatten()], bins=100)
                    plt.legend([f'pixel distribution for {name}'])
                    plt.xlim(-2, 2)
                    plt.ylim(0, 10000)
                    plt.savefig(filename)
                    plt.clf()


if __name__ == '__main__':
    main(generate_images=True,
         generate_histograms=False,
         save_as_nifti=False)
