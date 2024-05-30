import math
import os
import random
import sys
from builtins import NotImplementedError
from itertools import chain, combinations
from threading import RLock
from typing import List, Optional, Tuple, Union, Dict, Callable

import SimpleITK
import numpy
import scipy

import hiwi
from hiwi import ImageList
from lib.main_wrapper import main_wrapper
from lib.my_logger import logging
from lib.prime_math import get_prime_factors
from lib.print_exc_plus import print_exc_plus
from lib.util import EBC, required_size_for_safe_rotation
from load_data import Batch, VERTEBRAE
from load_data.annotated_batch import AnnotatedBatch
from load_data.load_image import image_by_patient_id, vertebra_contained
from load_data.patch_request import PatchRequest
from load_data.spaced_ct_volume import HUWindow
from load_data.vertebra_volume import vertebra_volume
from tasks import VertebraTasks, VertebraClassificationTask, GroupByClass, DummyTask, VertebraTask

FLOATX = 'float32'

Z = Y = X = float


def powerset(iterable):
    """powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


class LogicError(Exception):
    pass


class AnnotatedPatchesGenerator(EBC):
    class EmptyDataset(RuntimeError):
        pass

    def __init__(self,
                 image_list: hiwi.ImageList,
                 tasks: VertebraTasks,
                 vertebra_size_adjustment: Optional[str],
                 patch_requests: List[PatchRequest],
                 batch_size=1,
                 random_mode: Optional[Union[str, bool]] = None,
                 weighted=False,
                 pixel_scaling='divide_by_2k',
                 random_flip_lr=False,
                 random_flip_ud=False,
                 random_flip_fb=False,
                 random_negate_hu=False,
                 random_shift_mm: Union[int, Tuple[X, Y, Z]] = 0,
                 random_noise_sigma_a_1=0.0,
                 random_noise_sigma_a_2=0.0,
                 random_noise_sigma_m_1=0.0,
                 random_noise_sigma_m_2=0.0,
                 sigma_blur=0.0,
                 cache_batches=False,
                 cache_vertebra_volumes_in_ram=True,
                 auto_tune_batch_size=False,
                 data_amount: float = 1,
                 genders=frozenset({'m', 'w', None}),
                 additional_random_crops: float = 0.,
                 pad_value=0,
                 pad_augment_ratios: Tuple[Z, Y, X] = (0., 0., 0.),
                 pad_augment_probabilities: Tuple[Z, Y, X] = (0., 0., 0.),
                 random_project_x=0.,
                 random_project_y=0.,
                 random_project_z=0.,
                 rotate_range_deg: float = 0,
                 label_smoothing: float = 0,
                 cylindrical_crop_probability: float = 0,
                 cache_vertebra_volumes_on_disk: bool = True,
                 shuffle_labels=False,
                 ignore_border_vertebrae=False,
                 exclude_site_6=False,
                 spatial_dims: int = 0,
                 exclude_patients=None,
                 only_vertebra_levels: Optional[List[str]] = None,
                 hu_window=HUWindow.full_range(),
                 exclude_unlabeled=True,
                 verbose=True,
                 projection_methods: List[Callable[[numpy.ndarray], numpy.ndarray]] = None):
        """
        A keras data generator.
        :param image_list:
            the image list containing the dataset with one image per patient
        :param use_tiff_files: use tiff files is True or False, standard is False
        :param batch_size: number of patches per batch
        :param random_mode:
            one of "random_vertebra", "random class" or False
            "random_vertebra": for each patch that is put in a batch a vertebra is chosen randomly
            "random_class": for each patch that is put in a batch, first a class is chosen randomly (fractured or not fractured)
            False: vertebrae are put in the batch in the order that they appear in the list

            Neither "random_vertebra" nor "random_class" guarantee that after a
            certain number of batches each vertebra was fed into the model at least once.

            False guarantees that after generator.steps_per_epoch() batches each vertebra was fed into the
            model at least once.
        :param weighted: whether to apply class weights or not
        :param pixel_scaling: how to scale the values of the individual pixels. this is passed to the "scale_pixels" method
        :param random_flip_lr: flip along the x axis randomly for data augmentation (after cropping)
        :param random_flip_lr: flip along the z axis randomly for data augmentation (after cropping)
        :param random_shift_mm:
            randomly shift the crop for data augmentation along every axis for at most random_shift_mm in a random direction
            For example, a value of 2 would mean that the center of the crop is shifted by up to 2 mm in x direction,
            by up to 2 mm in y direction, and by up to 2 mm in z direction (independently)
            Can also be a tuple specifying different values for the dimensions in order X, Y, Z
        :param cache_batches:
            whether or not to cache individual batches in RAM. can be helpful for validation sets that do not need data augmentation
            Also, cache size in "number of batches x batch size" will be printed once if this is True.
            Keep in mind that this will prevent random data augmentation the second time a batch containing the same vertebrae is
            requested (should not be a problem for validation and testing, if you don't use data augmentation for that)
        :param auto_tune_batch_size:
            if True, the batch size will be automatically determined so that it divides the dataset size, i.e.
            the batch size will be chosen such that it is as large as possible, but still smaller or equal to the given batch size,
            such that self.steps_per_epoch() * self.batch_size == self.num_samples()
        :param data_amount:
            if this is less than 1, a random subset of vertebrae is dropped. This simulates training with less data and
            potentially allows to predict how much better the model would perform if more data was available.
        :param genders:
            only use the patients from the image list that are listed here
        :param additional_random_crops:
            If > 0 , some additional crops that do not contain vertebrae are mixed into the training process.
            More precisely, with probability "additional_random_crops" a the crop will be done at a random center
            and the label is set to 0.5 in that case.
            Not recommended to be used, because
            1. it slows down training (generating batches takes more time)
            2. it does not make much sense to have labels of 0.5 with binary crossentropy loss
        :param pad_value:
            if any padding is applied, the pixels will be filled with this value.
            Also if a crop is taken too close to the border of the CT image, missing values will be filled with this value
        :param pad_augment_probabilities:
            Tuple[Z, Y, X] of probabilities that determine how likely it is that some part of the image is cropped.
            For example (0.2,0.1,0) would mean that with 20% probability padding is applied along the z axis
            (the upper or the lower part, chosen randomly, of the image is filled with `pad_value`) and with 10% along the y axis.
            This simulates missing values at the border of the CT image.
        :param pad_augment_ratios:
            How large should the region be that is filled with `pad_value` in the case that `pad_augment_probabilities` is not 0.
            Tuple[Z, Y, X] with one value for each axis.
            For example a value of 0.2 in the would mean that up to 20% (chosen randomly between 0 and 20%)
            of the image are filled with `pad_value`.
        :param rotate_range_deg:
            How much to rotate the images for data augmentation on the sagittal plane (in degrees).
        :param cylindrical_crop_probability:
            with some probability the image is cropped into cylindrical shape and the missing pixels are filled with
            either 0, -1000, -2000 or -3000 HU. in that case, one of these values is chosen per sample (not per voxel)
            This is used to simulate the cylindrical padding around some CT images and can be used as data augmentation.
            It is only helpful if the padding is visible at least sometimes in the patches, for example if only
            the vertebrae in the center of the CT image are cropped, this may not have any effect.
        :param shuffle_labels
            randomly assign labels to vertebrae in the same class ratio as before. not recommended unless you know what you are doing
        """
        if spatial_dims != 0:
            raise NotImplementedError('Deprecated')
        if projection_methods is None:
            projection_methods = [numpy.mean, numpy.max]
        self.projection_methods = projection_methods
        self.label_smoothing = label_smoothing
        self.random_project_x = random_project_x
        self.random_project_y = random_project_y
        self.random_project_z = random_project_z
        self.sigma_blur = sigma_blur
        self.hu_window = hu_window
        self.patch_requests = patch_requests
        if exclude_patients is None:
            exclude_patients = []
        self.exclude_patients = exclude_patients
        if len(self.exclude_patients) > 0:
            logging.info(
                f'Excluding {len(set(img["patient_id"] for img in image_list).intersection(self.exclude_patients))} of {len(image_list)} patients from training')
        self.only_vertebra_levels = only_vertebra_levels
        self.tasks = tasks
        self.spatial_dims = spatial_dims
        self.genders = genders
        if cylindrical_crop_probability > 0 and cache_vertebra_volumes_on_disk:
            raise ValueError('You cant use the vertebra cache and at the same time apply random cylindrical cropping '
                             'to the CT images. Maybe you instead wanted to use `cache_batches`?')
        self.cache_vertebra_volumes_on_disk = cache_vertebra_volumes_on_disk
        self.cylindrical_crop_probability = cylindrical_crop_probability
        self.additional_random_crops = additional_random_crops
        self.rotate_range_deg = rotate_range_deg
        self.pad_augment_probabilities = pad_augment_probabilities
        self.pad_augment_ratios = pad_augment_ratios
        self.pad_value = pad_value
        self.image_list = image_list
        self.ignore_border_vertebrae = ignore_border_vertebrae
        self.exclude_site_6 = exclude_site_6
        self.vertebra_size_adjustment = vertebra_size_adjustment
        if self.vertebra_size_adjustment not in [None, 'zoom', 'pad']:
            raise NotImplementedError
        assert len(set(i['patient_id'] for i in self.image_list)) == len(self.image_list)

        # remove patients with wrong gender
        for patient_idx in range(len(self.image_list))[::-1]:
            if 'gender' in self.image_list[patient_idx] and self.image_list[patient_idx]['gender'] not in self.genders:
                del self.image_list[patient_idx]
        assert len(set(i['patient_id'] for i in self.image_list)) == len(self.image_list)

        # randomly drop some data if desired
        assert 0 <= data_amount <= 1
        if data_amount < 1:
            new_indices = numpy.random.choice(range(len(image_list)), size=round(len(image_list) * data_amount), replace=False)
            assert len(new_indices) == len(set(new_indices))
            self.image_list = ImageList(self.image_list[img_idx] for img_idx in sorted(new_indices))
            logging.warning(f'Randomly dropped {1 - len(self.image_list) / len(image_list):.2%} patients')
        assert len(set(i['patient_id'] for i in self.image_list)) == len(self.image_list)
        assert len(self.image_list) > 0

        if shuffle_labels:
            raise NotImplementedError

        excluded_vertebrae = [(image['patient_id'], vertebra)
                              for image in self.image_list
                              for vertebra in image.parts
                              if any(task.is_excluded(image, vertebra) for task in self.tasks)]
        self.groups_by_class_by_task: List[GroupByClass] = [task.group_by_class(self.image_list,
                                                                                exclude_vertebrae=excluded_vertebrae,
                                                                                on_invisible_vertebra='same_as_unlabeled',
                                                                                exclude_unclassified=True if exclude_unlabeled else 'dummy_label')
                                                            for task in self.tasks]
        if exclude_unlabeled:  # Recommendation: set to False when doing predictions, in case a localizer predicts a vertebra that has no ground truth
            self.all_vertebrae = list(set(vertebra_name
                                          for groupss in self.groups_by_class_by_task
                                          for group in groupss.groups()
                                          for vertebra_name in group))
        else:
            self.all_vertebrae = list(set((img['patient_id'], vertebra)
                                          for img in self.image_list
                                          for vertebra in img.parts
                                          if img.parts[vertebra].position is not None
                                          # this could happen if a localizer says that a vertebra is not in the image
                                          ))
            self.filter_vertebra_sublists()  # to ensure that no vertebrae with position None are in the sublists

        if len(self.all_vertebrae) == 0:
            raise self.EmptyDataset('Did you configure the correct tasks? Is ground truth available? Is the ImageList not empty?')

        if verbose:
            self.class_summary()
        if self.ignore_border_vertebrae:
            self.filter_border_vertebrae()
        if self.exclude_site_6:
            site = '6'
            self.filter_by_site(site)
        self.filter_vertebra_range()

        if len(self.all_vertebrae) == 0:
            raise self.EmptyDataset()

        for groups in self.groups_by_class_by_task:
            self.groups_by_class_by_task[VertebraTasks.MAIN_TASK_IDX].check_subset(self.all_vertebrae)
            # groups.check_all_classes_present()
            groups.check_all_sets_and_disjoint()
        assert self.main_task() is self.groups_by_class_by_task[VertebraTasks.MAIN_TASK_IDX].task
        assert len(self.tasks) == len(self.groups_by_class_by_task)

        if random_mode == 'random_class':
            try:
                self.groups_by_class_by_task[VertebraTasks.MAIN_TASK_IDX].check_all_vertebrae_present(self.all_vertebrae)
            except AssertionError:
                raise ValueError(f'You want to pick a random vertebra from a random class from task {self.main_task().output_layer_name()},\n'
                                 f'but this task has some (probably unlabeled) vertebrae that would never be chosen this way.')
        # compute class weights
        self.weighted = weighted
        self.class_weights: List[Dict[str, float]] = self.compute_class_weights()

        self.step_count = 0
        self.batch_size = batch_size
        if auto_tune_batch_size:
            self.tune_batch_size(self.batch_size)

        self.deformity_label_names = []
        self.score_label_names = []
        self.ddx_classes = []
        self.ddx_cat_classes = []

        self._label_cache: Dict = {}
        self.random_shift_px: Tuple[int, int, int]
        if isinstance(random_shift_mm, int):
            self.random_shift_mm = (random_shift_mm, random_shift_mm, random_shift_mm,)
        elif isinstance(random_shift_mm, tuple) and len(random_shift_mm) == 3:
            self.random_shift_mm = random_shift_mm
        else:
            raise ValueError
        self.random_flip_lr = random_flip_lr
        self.random_flip_ud = random_flip_ud
        self.random_flip_fb = random_flip_fb
        self.random_negate_hu = random_negate_hu
        self.random_noise_sigma_a_1 = random_noise_sigma_a_1
        self.random_noise_sigma_a_2 = random_noise_sigma_a_2
        self.random_noise_sigma_m_1 = random_noise_sigma_m_1
        self.random_noise_sigma_m_2 = random_noise_sigma_m_2
        self.pixel_scaling = pixel_scaling
        self.cache_batches = cache_batches
        self.cache_vertebra_volumes_in_ram = cache_vertebra_volumes_in_ram
        self.last_batch_names: Optional[List[str]] = None
        self.last_batch_vertebra_counts: Optional[List[int]] = None
        self.last_batch_sq_fracture_counts: Optional[List[int]] = None
        self.last_batch_osteoporotic_fracture_counts: Optional[List[int]] = None
        self.last_offset: Optional[int] = None

        if self.cache_batches:
            step = 1
            while step * self.batch_size % self.num_samples() != 0:
                step += 1
            print('{0}: Cache size = {1} x {2}'.format(type(self).__name__, step, self.batch_size))
        self.vertebra_volume_cache = {}
        self.batch_cache: Dict[int, AnnotatedBatch] = {}
        self.random_mode = random_mode
        self.generator_lock = RLock()
        # self.generator_lock = PrintLineRLock(name=str(self.cache_batches) + ' ' + str(len(self.ct_dirs)))
        self.tmp_image: Optional[SimpleITK.Image] = None

    def compute_class_weights(self):
        if self.weighted == 'running':
            raise DeprecationWarning('weighted="running" is deprecated')
        if self.weighted:
            assert len(self.main_task().class_names()) == self.groups_by_class_by_task[VertebraTasks.MAIN_TASK_IDX].num_classes()
            result = [groups.class_weights() for groups in self.groups_by_class_by_task]
        else:
            result = [{label_name: 1 for label_name in task.class_names()} for task in self.tasks]
        return result

    def recompute_class_weights(self):
        self.class_weights = self.compute_class_weights()

    def class_summary(self):
        summary = '{' + ', '.join(group.summary() for group in self.groups_by_class_by_task) + '}'
        logging.info(f'Class size summary (n={self.num_samples()}): {summary}')

    def patient_ids(self):
        return list(set([patient_id for patient_id, vertebra_name in self.all_vertebrae]))
        # return [i['patient_id'] for i in self.image_list]

    def classification_tasks(self):
        return [t for t in self.tasks if isinstance(t, VertebraClassificationTask)]

    def filter_by_site(self, exclude_site):
        remove_indices = []
        for v_idx, v in enumerate(self.all_vertebrae):
            # these are sorted so we can just compare the patient ids of the previous and next vertebra
            patient_id: str = v[0]
            if patient_id.startswith(str(exclude_site)):
                remove_indices.append(v_idx)
        for remove_idx in reversed(remove_indices):
            del self.all_vertebrae[remove_idx]
        self.filter_vertebra_sublists()

    def filter_vertebra_range(self):
        if self.only_vertebra_levels is None:
            return
        remove_indices = []
        for v_idx, v in enumerate(self.all_vertebrae):
            vertebra: str = v[1]
            if vertebra not in self.only_vertebra_levels:
                remove_indices.append(v_idx)
        for remove_idx in reversed(remove_indices):
            del self.all_vertebrae[remove_idx]
        self.filter_vertebra_sublists()

    def filter_border_vertebrae(self):
        remove_indices = []
        for v_idx, v in enumerate(self.all_vertebrae):
            # these are sorted so we can just compare the patient ids of the previous and next vertebra
            patient_id = v[0]
            if v_idx == 0 or \
                    v_idx == len(self.all_vertebrae) - 1 or \
                    patient_id != self.all_vertebrae[v_idx - 1][0] or \
                    patient_id != self.all_vertebrae[v_idx + 1][0]:
                remove_indices.append(v_idx)
        for remove_idx in reversed(remove_indices):
            del self.all_vertebrae[remove_idx]
        self.filter_vertebra_sublists()

    def filter_vertebra_sublists(self):
        for vertebra_list in self.vertebra_sublists():
            vertebra_list[:] = [v for v in vertebra_list if v in self.all_vertebrae]
        if len(self.all_vertebrae) == 0:
            raise self.EmptyDataset()

    def vertebra_sublists(self):
        return [vertebrae for groupss in self.groups_by_class_by_task for vertebrae in groupss.groups()]

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
        if self.batch_size != maximum_batch_size and self.batch_size != self.num_samples():
            logging.info(f'Automatically chosen batch size: {self.batch_size}/{self.num_samples()}')
        # print(self.batch_size, self.num_samples(), maximum_batch_size)

    def required_size_mm(self, input_patch: PatchRequest) -> Tuple[X, Y, Z]:
        """
        compute how large the crop must be such that cropping and rotating do produce unknown values
        :return: crop size in mm
        """
        rotate_range_deg = self.rotate_range_deg
        # need some space for random cropping
        # noinspection PyTypeChecker
        base: Tuple[X, Y, Z] = tuple([size + max_shift * 2
                                      for size, max_shift in zip(input_patch.size_mm,
                                                                 self.random_shift_mm)])

        # need some space for random rotations
        result = required_size_for_safe_rotation(base, rotate_range_deg)
        if all(max_shift == 0 for max_shift in self.random_shift_mm) and rotate_range_deg == 0:
            assert tuple(result) == tuple(input_patch.size_mm), (result, input_patch.size_mm)
        return result

    def steps_per_epoch(self):
        return math.ceil(self.num_samples() / self.batch_size)

    def __iter__(self):
        with self.generator_lock:
            self.step_count = 0
            return self

    def num_samples(self):
        return len(self.all_vertebrae)

    def __next__(self):
        with self.generator_lock:
            step_idx = self.step_count
            self.step_count += 1
        # this calls the __getitem__ method
        return self[step_idx]

    def clear_caches(self):
        self.batch_cache.clear()
        self.__dict__.clear()

    def invalidate(self):
        self.clear_caches()
        self.image_list = None
        self.all_vertebrae = None
        self.groups_by_class_by_task = None
        self.patch_requests = None
        self.tasks = None
        self.class_weights = None
        self.last_ = None

    def __getitem__(self, step: int):
        batch = self.get_batch(step)
        self.last_batch_names = batch.names
        # batch.serialize(generate_histograms=True, save_as_nifti=False)
        return batch.as_tuple_for_keras()

    def get_batch(self, step) -> AnnotatedBatch:
        offset = step * self.batch_size % self.num_samples()
        self.last_offset = offset
        if self.cache_batches and offset in self.batch_cache:
            batch = self.batch_cache[offset]
        else:
            batch = AnnotatedBatch.initialize_new_batch(self.batch_size, self.tasks, patch_requests=self.patch_requests)

            self.fill_batch_with_data(batch, offset)

            if self.weighted:
                batch.adjust_sample_weights_for_unlabeled_vertebrae()

            batch.y_trues = [numpy.array(label, dtype=FLOATX) for label in batch.y_trues]

            assert batch.fully_built()
            batch.check_batch_size(expected_size=self.batch_size)
            batch.check_num_tasks(expected_size=len(self.tasks))
            batch.check_no_nan_in_y_trues()  # x can be nan if the position is undefined, but y must be defined
            batch.check_patch_size()

            if self.cache_batches:
                # store in cache if desired
                self.batch_cache[offset] = batch
        return batch

    def fill_batch_with_data(self, batch: AnnotatedBatch, offset: int):
        batch_idx = 0
        while batch_idx < self.batch_size:  # actually fill the batch
            assert not self.cache_batches or self.batch_size * self.steps_per_epoch() == self.num_samples()
            patient_id, center_vertebra = self.choose_vertebra_for_batch(batch_idx, offset)

            if random.random() < self.additional_random_crops:
                raise NotImplementedError

            # load the label into/from a smaller cache
            self.load_image_labels_and_class_weights(patient_id, center_vertebra, batch, batch_idx)

            # load the actual patch containing a vertebra
            flip_ud = self.random_flip_ud and random.random() < 0.5
            flip_fb = self.random_flip_fb and random.random() < 0.5
            flip_lr = self.random_flip_lr and random.random() < 0.5

            # noinspection PyTypeChecker
            actual_shift: Tuple[X, Y, Z] = tuple(random.uniform(-max_shift, max_shift)
                                                 for max_shift in self.random_shift_mm)
            negate_hu = random.random() < 0.5

            for input_idx, input_patch in enumerate(self.patch_requests):
                vertebra = self.name_of_vertebra_at_input_patch(center_vertebra, input_patch, patient_id=patient_id)
                if vertebra == 'out_of_image':
                    continue
                raw_volume = self.load_vertebra_from_cache_or_method(patient_id, vertebra, input_patch)
                # coordinate order is now UD (z), FB (y), LR (x)

                # random_rotate
                if self.rotate_range_deg > 0:
                    raw_volume[:] = self.apply_rotation_augmentation(raw_volume)

                raw_volume = self.apply_noise_or_blur_augmentation_if_needed(raw_volume)

                # compute coordinates where to crop after shift and rotation
                crop_start: Tuple[Z, Y, X] = (
                    round((raw_volume.shape[0] - input_patch.size_px()[0]) / 2 + actual_shift[2] / input_patch.spacing[2]),
                    round((raw_volume.shape[1] - input_patch.size_px()[1]) / 2 + actual_shift[1] / input_patch.spacing[1]),
                    round((raw_volume.shape[2] - input_patch.size_px()[2]) / 2 + actual_shift[0] / input_patch.spacing[0]),
                )

                volume = raw_volume[
                         crop_start[0]:crop_start[0] + math.ceil(input_patch.size_px()[0]),
                         crop_start[1]:crop_start[1] + math.ceil(input_patch.size_px()[1]),
                         crop_start[2]:crop_start[2] + math.ceil(input_patch.size_px()[2]),
                         ]  # cut volume

                del raw_volume

                projection_method = random.choice(self.projection_methods)
                if random.random() < self.random_project_x:
                    volume[:] = projection_method(volume, axis=2, keepdims=True)
                if random.random() < self.random_project_y:
                    volume[:] = projection_method(volume, axis=1, keepdims=True)
                if random.random() < self.random_project_z:
                    volume[:] = projection_method(volume, axis=0, keepdims=True)

                # Random flipping (same for adjacent vertebrae)
                if flip_ud:
                    volume = numpy.flip(volume, axis=0)
                if flip_fb:
                    volume = numpy.flip(volume, axis=1)
                if flip_lr:
                    volume = numpy.flip(volume, axis=2)

                # pretend there was some padding like at the border of the CT but not rotated (data augmentation)
                augment_pad_slices = []
                for dim in range(3):
                    max_ratio = self.pad_augment_ratios[dim]
                    pad_probability = self.pad_augment_probabilities[dim]
                    if max_ratio == 0 or random.random() > pad_probability:
                        augment_pad_slices.append(slice(0))
                        continue
                    ratio = random.uniform(0, max_ratio)
                    if random.random() < 0.5:
                        augment_pad_slices.append(slice(0, math.floor(ratio * volume.shape[dim])))
                    else:
                        augment_pad_slices.append(
                            slice(volume.shape[dim] - math.floor(ratio * volume.shape[dim]), volume.shape[dim]))
                if any(s != slice(0) for s in augment_pad_slices):
                    volume = volume.copy()
                    volume[augment_pad_slices[0], :, :] = self.pad_value
                    volume[:, augment_pad_slices[1], :] = self.pad_value
                    volume[:, :, augment_pad_slices[2]] = self.pad_value

                if self.random_negate_hu and negate_hu:
                    volume *= -1

                # add the data to the batch
                if batch.xs is None:
                    batch.xs = [numpy.full(shape=(self.batch_size, *volume.shape, 1), fill_value=self.pad_value, dtype=FLOATX)
                                for _ in self.patch_requests]
                batch.xs[input_idx][batch_idx, :, :, :, 0] = volume

            if self.cache_batches:  # computationally expensive, but if we cache it anyways it is affordable
                if not all(numpy.isfinite(x).all() for x in batch.xs):
                    raise RuntimeError('Batch contains NaN on input side. Could be intended but more likely not.')

            # store the name in a list of batch_size
            batch.names[batch_idx] = str((patient_id, center_vertebra))

            sys.stdout.flush()
            batch_idx += 1

    def all_vertebra_names(self):
        return [str((patient_id, vertebra_name))
                for patient_id, vertebra_name in self.all_vertebrae]

    def load_vertebra_from_cache_or_method(self, patient_id, vertebra, input_patch: PatchRequest):
        cache_key = (patient_id, vertebra, input_patch.spacing, self.rounded_required_size_mm(input_patch))
        if self.cache_vertebra_volumes_in_ram:
            if cache_key not in self.vertebra_volume_cache:
                self.vertebra_volume_cache[cache_key] = self.load_vertebra_volume(patient_id, vertebra, input_patch)
            raw_volume = self.vertebra_volume_cache[cache_key].copy()  # the copy is needed to avoid mutating the cache
        else:
            raw_volume = self.load_vertebra_volume(patient_id, vertebra, input_patch)
        return raw_volume

    def name_of_vertebra_at_input_patch(self, center_vertebra, input_patch: PatchRequest, patient_id: str):
        if input_patch.offset == 0:
            vertebra_name = center_vertebra
        elif center_vertebra in VERTEBRAE:
            center_vertebra_idx = VERTEBRAE.index(center_vertebra)
            vertebra_name = self.vertebra_name_by_idx(patient_id, vertebra_idx=center_vertebra_idx + input_patch.offset)
        else:
            raise NotImplementedError('TODO: if this is a number use the successors and predecessors')
        return vertebra_name

    def vertebra_name_by_idx(self, patient_id, vertebra_idx):
        if 0 <= vertebra_idx < len(VERTEBRAE) and vertebra_contained(patient_id, VERTEBRAE[vertebra_idx], self.image_list):
            vertebra_name = VERTEBRAE[vertebra_idx]
        else:
            if self.ignore_border_vertebrae:
                # use another vertebra
                raise LogicError('This is a bug. The method choose_vertebra_for_batch should not have picked a border vertebra')
            else:
                # just put a black image as an input there
                vertebra_name = 'out_of_image'
        return vertebra_name

    def apply_noise_or_blur_augmentation_if_needed(self, raw_volume):
        if self.sigma_blur > 0 and any(r > 0 for r in [self.random_noise_sigma_a_1,
                                                       self.random_noise_sigma_a_2,
                                                       self.random_noise_sigma_m_1,
                                                       self.random_noise_sigma_m_2]):
            raise NotImplementedError('Having both blur and noise augmentation is not implemented yet.')
        if self.random_noise_sigma_a_1 > 0:
            raw_volume += numpy.random.normal(0, self.random_noise_sigma_a_1, size=raw_volume.shape)
        if self.random_noise_sigma_a_2 > 0:
            raw_volume += numpy.random.normal(0, self.random_noise_sigma_a_2, size=(1,))
        if self.random_noise_sigma_m_1 > 0:
            raw_volume *= numpy.random.normal(1, self.random_noise_sigma_m_1, size=raw_volume.shape)
        if self.random_noise_sigma_m_2 > 0:
            raw_volume *= numpy.random.normal(1, self.random_noise_sigma_m_2, size=(1,))
        if self.sigma_blur > 0:
            scipy.ndimage.filters.gaussian_filter(raw_volume, sigma=self.sigma_blur, output=raw_volume,
                                                  mode='constant', cval=0)  # reflect might make more sense but this is what was used in training
        return raw_volume

    def apply_rotation_augmentation(self, raw_volume):
        # use the instance variable self.tmp_image for the temporary array, so that we do not have to
        # allocate memory every time we want to rotate
        if self.tmp_image is None:
            self.tmp_image = SimpleITK.GetImageFromArray(raw_volume)
        else:
            from SimpleITK import _SimpleITK
            try:
                _SimpleITK._SetImageFromArray(raw_volume, self.tmp_image)
            except RuntimeError as e:
                if e.args[0] == 'Size mismatch of image and Buffer.':
                    # in this case we actually need to allocate new memory because the size of the crop has changed
                    self.tmp_image = SimpleITK.GetImageFromArray(raw_volume)
                else:
                    raise
        transform = SimpleITK.AffineTransform(3)
        transform.SetCenter(tuple(s // 2 for s in reversed(raw_volume.shape)))
        radians = random.uniform(-self.rotate_range_deg / 180 * math.pi,
                                 self.rotate_range_deg / 180 * math.pi)
        transform.Rotate(axis1=1, axis2=2,
                         angle=radians)  # reverse axis order, in numpy these are axes 0 and 1
        self.tmp_image = SimpleITK.Resample(self.tmp_image, transform, SimpleITK.sitkLinear, 0,
                                            SimpleITK.sitkFloat32)
        raw_volume = SimpleITK.GetArrayViewFromImage(self.tmp_image)
        return raw_volume

    def load_vertebra_volume(self, patient_id, vertebra, input_patch: PatchRequest) -> numpy.ndarray:
        img = image_by_patient_id(patient_id, self.image_list)
        iml_name = self.image_list.name if hasattr(self.image_list, 'name') else None
        assert iml_name is not None, 'missing name for image_list, please set image_list.name to a unique identifier for the image_list'
        additional_cache_key = (str(img.path), iml_name, self.hu_window.minimum, self.hu_window.maximum)
        if self.cache_vertebra_volumes_on_disk:
            # use the HDD-cache if available and write the result in cache afterwards
            vertebra_volume_function = vertebra_volume.__call__
        else:
            # bypass the HDD-cache and call the function and do not store the result in cache afterwards
            vertebra_volume_function = vertebra_volume.func
        result = vertebra_volume_function(img,
                                          vertebra,
                                          desired_spacing=input_patch.spacing,
                                          desired_size_mm=tuple(round(s) for s in self.rounded_required_size_mm(input_patch)),
                                          pixel_scaling=self.pixel_scaling,
                                          cylindrical_crop_probability=self.cylindrical_crop_probability,
                                          # dtype='float16', # save some RAM, cast to 32 bit later, should be done by default
                                          pad_value=self.pad_value,
                                          hu_window=self.hu_window,
                                          vertebra_size_adjustment=self.vertebra_size_adjustment,
                                          _additional_cache_key=additional_cache_key)
        assert str(result.dtype) == 'float16'
        return result.astype('float32')

    def rounded_required_size_mm(self, input_patch, ceiling_offset: int = 2, ceil_to: int = 10):
        adjust = lambda x: math.ceil((round(x) - ceiling_offset) / ceil_to) * ceil_to + ceiling_offset
        non_rounded = self.required_size_mm(input_patch)
        result = tuple(adjust(s) for s in non_rounded)
        for s, r in zip(non_rounded, result):
            assert s <= r, (s, r)
        return result

    def main_task(self) -> VertebraTask:
        return self.tasks.main_task()

    def load_image_labels_and_class_weights(self, patient_id, vertebra, batch: AnnotatedBatch, batch_idx: int):
        # load into cache if not available
        if (patient_id, vertebra) not in self._label_cache:
            self._load_vertebra_data_into_label_cache(patient_id, vertebra)

        image_labels, sample_weights, is_unlabeled = self._label_cache[(patient_id, vertebra)]
        assert len(image_labels) == len(self.tasks) == len(batch.sample_weights) == len(is_unlabeled)
        batch.check_num_tasks(len(self.tasks))
        for task_idx in range(len(self.tasks)):
            if is_unlabeled[task_idx]:
                batch.num_unlabeled[task_idx] += 1
            batch.y_trues[task_idx][batch_idx] = image_labels[task_idx]
            batch.sample_weights[task_idx][batch_idx] = sample_weights[task_idx]

    def _load_vertebra_data_into_label_cache(self, patient_id, vertebra):
        image_labels: List[numpy.ndarray] = []
        sample_weights: List[Optional[float]] = []
        is_unlabeled: List[Optional[float]] = []
        assert len(self.tasks) == len(self.class_weights)

        image = image_by_patient_id(patient_id, self.image_list)
        for task_idx, task in enumerate(self.tasks):
            try:
                label = task.y_true_with_additional_spatial_dimensions(image, vertebra_name=vertebra, n_spatial_dims=self.spatial_dims)
                label = task.smooth_label(label, self.label_smoothing)
                image_labels.append(label)
            except VertebraClassificationTask.UnlabeledVertebra:
                # in this case there is nothing we could use as a label
                sample_weights.append(1e-8)
                image_labels.append(task.default_label())
                is_unlabeled.append(True)
            else:
                is_unlabeled.append(False)
                try:
                    label_name = task.class_name_of_label(label)
                except VertebraClassificationTask.UnlabeledVertebra:
                    raise RuntimeError('class_idx_of_label should not throw UnlabeledVertebra.\n'
                                       'If this was because we have no ground truth, then y_true_from_hiwi_image_and_vertebra should throw it.\n'
                                       'If this was because the label does not fit any class, then LabelWithoutClass is the correct exception.')
                except VertebraClassificationTask.LabelWithoutClass:
                    # this could be a soft pseudo-label generated from a teacher model
                    sample_weights.append(1)
                else:
                    # default case, here we have a label that clearly belongs to a class
                    sample_weights.append(self.class_weights[task_idx][label_name])

        if self.random_mode == 'random_class':
            # class weights relative to oversampling, see relative_class_weights.png
            self.adjust_sample_weights_by_oversampling(sample_weights)

        assert len(self.tasks) == len(image_labels) == len(sample_weights) == len(is_unlabeled)
        self._label_cache[(patient_id, vertebra)] = image_labels, sample_weights, is_unlabeled

    def adjust_sample_weights_by_oversampling(self, sample_weights: List[float]):
        oversampling_factor = sample_weights[VertebraTasks.MAIN_TASK_IDX]
        if oversampling_factor == 1e-8:
            raise RuntimeError('You can not use relative oversampling when there are missing labels for the main task.')
        for task_idx, task in enumerate(self.tasks):
            sample_weights[task_idx] = sample_weights[task_idx] / oversampling_factor

    def choose_vertebra_for_batch(self, batch_idx, offset) -> Tuple[str, str]:
        if len(self.exclude_patients) > 0 and not self.random_mode:
            raise NotImplementedError('Using exclude_patients without randomness does not work in the current implementation.')

        tries = 0
        patient_id = None
        vertebra = None
        while patient_id is None or patient_id in self.exclude_patients:
            if tries % 100000 == 0 and tries > 0:
                logging.warning(f'Possible performance leak or infinite loop: Unable to find a non-excluded patient id after {tries} tries.')
            if self.random_mode == 'random_vertebra':
                patient_id, vertebra = random.choice(self.all_vertebrae)
            elif self.random_mode == 'random_class':
                class_name = random.choice(self.main_task().class_names())
                patient_id, vertebra = random.choice(self.groups_by_class_by_task[VertebraTasks.MAIN_TASK_IDX].group_for_class(class_name))
            elif not self.random_mode:
                patient_id, vertebra = self.all_vertebrae[(offset + batch_idx) % self.num_samples()]
            else:
                raise ValueError(self.random_mode)
            tries += 1
        assert patient_id is not None
        assert vertebra is not None
        return patient_id, vertebra

    def is_deterministic(self) -> bool:
        def _all_batches_equal(bs1: List[Batch], bs2: List[Batch]) -> bool:
            if len(bs1) != len(bs2):
                return False
            for b1, b2 in zip(bs1, bs2):
                if len(b1) != len(b2):
                    return False
                for l1, l2 in zip(b1, b2):
                    if len(l1) != len(l2):
                        return False
                    for a1, a2 in zip(l1, l2):
                        if a1.shape != a2.shape:
                            return False
                        if not numpy.allclose(a1, a2, equal_nan=True):
                            return False
            return True

        # execute generator twice and check if results are the same
        return _all_batches_equal([next(self) for _ in range(self.steps_per_epoch())],
                                  [next(self) for _ in range(self.steps_per_epoch())])


class AnnotatedPatchesGeneratorPseudo3D(AnnotatedPatchesGenerator):
    OVERWRITE_INVALID_ARGS = False
    if OVERWRITE_INVALID_ARGS:
        print('WARNING: AnnotatedPatchesGenerator2D is in OVERWRITE_INVALID_ARGS mode. Use only for experimental purposes and not for actual training.')

    def __init__(self, **kwargs):
        if 'pixel_scaling' in kwargs:
            scaling = kwargs['pixel_scaling']
        else:
            scaling = 'divide_by_2k'
        pad_value = self.pick_padding_by_scaling(scaling)
        force_values = {
            # if we set the pad value low enough, the padding will be overwritten by the value in the single slice
            'random_project_x': 1.0,
            'projection_methods': [numpy.max],
            'pixel_scaling': scaling,
            'pad_value': pad_value,
        }
        for k, v in force_values.items():
            if not self.OVERWRITE_INVALID_ARGS:
                if k in kwargs and kwargs[k] != v:
                    raise ValueError(f'You can not set {k} to {kwargs[k]} because it is forced to {v}.')
            kwargs[k] = v
        super().__init__(**kwargs)

    def pick_padding_by_scaling(self, scaling):
        if scaling.endswith('_prescale'):
            scaling = scaling[:-len('_prescale')]
        if scaling == 'divide_by_2k':
            pad_value = -1024 / 2048
        elif scaling == 'range01':
            pad_value = 0.
        else:
            raise NotImplementedError(scaling)
        return pad_value


class AnnotatedPatchesGenerator2D(AnnotatedPatchesGeneratorPseudo3D):
    AXIS_IDX = 2  # X

    def __init__(self, **kwargs):
        for r in kwargs['patch_requests']:
            r: PatchRequest
            assert r.size_px()[self.AXIS_IDX] == 1
        super().__init__(**kwargs)

    def axis_idx_in_zyx(self):
        return self.AXIS_IDX

    def axis_idx_in_bzyxc(self):
        return self.AXIS_IDX + 1

    def fill_batch_with_data(self, batch: AnnotatedBatch, offset: int):
        super().fill_batch_with_data(batch, offset)
        for x in batch.xs:
            assert len(x.shape) == 5  # Pseudo-3D
        batch.xs = [x.squeeze(self.axis_idx_in_bzyxc())
                    for x in batch.xs]
        for x in batch.xs:
            assert len(x.shape) == 4  # Real 2D, BZYC


@main_wrapper
def main(generate_images=False, augment=0, generate_histograms=False, save_as_nifti=False, subdir_by_counts=False,
         datasets: List[hiwi.ImageList] = None):
    # random.seed(42)

    def shorten_name(name):
        import re
        name = re.sub(r'\s+', r' ', str(name))
        name = name.replace(', ', ',')
        name = name.replace(', ', ',')
        name = name.replace(' ', '_')
        return re.sub(r'([A-Za-z])[a-z]*_?', r'\1', str(name))

    print('augment:', augment)
    pixel_scalings = [
        # 'divide_by_2k',
        'range01',
    ]
    from ct_dirs import DERIVED_PATH
    if datasets is None:
        datasets = [
            # os.path.join('spacing_3_1_1', 'x_vert_seg_data1.iml'),
            # os.path.join('spacing_3_1_1', 'clean_patients.iml'),

            os.path.join('mros_scouts', 't_with_manually_annotated_coordinates.iml'),
            os.path.join('diagbilanz_scouts', 'clean_patients.iml'),

            # os.path.join('original_spacing', 'x_vert_seg_data1.iml'),
            # os.path.join('original_spacing', 'clean_patients.iml'),

            # os.path.join('verse', 'verse_validation.iml'),
            # os.path.join('verse', 'verse_test.iml'),
            # os.path.join('verse', 'verse_train.iml'),
        ]
    patch_requestss = [
        # [
        #     PatchRequest(spacing=(3, 1, 1),
        #                  size_mm=(60, 50, 40)),
        # ]
        # [
        #     PatchRequest.from_input_size_px_and_spacing(spacing=(2, 1, 1),
        #                                                 input_size_px=(31, 47, 47)),
        #     PatchRequest.from_input_size_px_and_spacing(spacing=(2, 2, 2),
        #                                                 input_size_px=(31, 47, 47)),
        #                                                 ],
        # [
        #     PatchRequest.from_input_size_px_and_spacing(spacing=(1, 1, 1),
        #                                                 input_size_px=(31, 31, 31)),
        # ],
        [
            PatchRequest.from_input_size_px_and_spacing(spacing=(2, 1, 1),
                                                        input_size_px=(31, 47, 47)),
        ],
        # [
        #     PatchRequest.from_input_size_px_and_spacing(spacing=(1, 1, 1),
        #                                                 input_size_px=(19, 19, 19)),
        #     PatchRequest.from_input_size_px_and_spacing(spacing=(2, 2, 2),
        #                                                 input_size_px=(19, 19, 19)),
        #     PatchRequest.from_input_size_px_and_spacing(spacing=(3, 3, 3),
        #                                                 input_size_px=(19, 19, 19)),
        #     PatchRequest.from_input_size_px_and_spacing(spacing=(5, 5, 5),
        #                                                 input_size_px=(19, 19, 19)),
        # ]
    ]
    for dataset_path in datasets:
        dataset = hiwi.ImageList.load(os.path.join(DERIVED_PATH, dataset_path))
        for scaling in pixel_scalings:
            for patch_requests in patch_requestss:
                print('Now doing', scaling, len(dataset), patch_requests)
                if augment:
                    # noise = 0
                    # shift = 0
                    # pad_augment_ratios = (0., 0., 0.)
                    # pad_augment_probabilities = (0., 0., 0.)

                    shift = 4
                    pad_augment_ratios = (0.4, 0.2, 0.)
                    pad_augment_probabilities = (0.25, 0.25, 0.)
                    rotate_range_deg = 18
                    random_project_probability = 0.1

                    # shift = (40 // 3, 40, 5)  # order X, Y, Z
                    # pad_augment_probabilities = (0.2, 0.2, 0)
                    # pad_augment_ratios = (0.2, 0.2, 0.)
                    # rotate_range_deg = 0
                    # cylindrical_crop_probability = 0.25
                else:
                    shift = 0
                    pad_augment_ratios = (0., 0., 0.)
                    pad_augment_probabilities = (0., 0., 0.)
                    rotate_range_deg = 0
                    # cylindrical_crop_probability = 0
                    random_project_probability = 0.
                # only_some_examples = 100
                only_some_examples = None
                generator = AnnotatedPatchesGeneratorPseudo3D(image_list=dataset,
                                                              patch_requests=patch_requests,
                                                              tasks=VertebraTasks([DummyTask()]),
                                                              batch_size=math.inf if only_some_examples is None else only_some_examples,
                                                              auto_tune_batch_size=only_some_examples is None,
                                                              random_mode=False,
                                                              weighted=True,
                                                              pixel_scaling=scaling,
                                                              random_flip_lr=bool(augment),
                                                              random_shift_mm=shift,
                                                              random_project_x=1.0,
                                                              random_project_y=random_project_probability,
                                                              random_project_z=random_project_probability,
                                                              pad_augment_ratios=pad_augment_ratios,
                                                              pad_augment_probabilities=pad_augment_probabilities,
                                                              rotate_range_deg=rotate_range_deg,
                                                              cache_vertebra_volumes_on_disk=False,
                                                              cache_vertebra_volumes_in_ram=False,
                                                              vertebra_size_adjustment=None, )
                for step_idx in range(generator.steps_per_epoch()):
                    print('Loading batch...')
                    batch = generator.get_batch(step_idx)
                    batch.serialize(
                        generate_images=generate_images,
                        save_as_nifti=save_as_nifti,
                        generate_histograms=generate_histograms,
                        relative_dir_name=dataset_path,
                        progress_bar=True,
                    )
                print()


if __name__ == '__main__':
    # noinspection PyBroadException
    try:
        main(generate_images=True,
             generate_histograms=True,
             save_as_nifti=True,
             augment=False,
             subdir_by_counts=False)
    except Exception:
        print_exc_plus()
        exit(-1)
