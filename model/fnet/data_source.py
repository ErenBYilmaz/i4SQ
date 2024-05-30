import copy
import os
from typing import Tuple, Optional, List, Callable, Dict, Set, Any

import cachetools

import hiwi
from hiwi import ImageList
from lib.my_logger import logging
from lib.util import EBC, val_fold_by_test_fold
from load_data.random_dataset_split import random_dataset_split
from tasks import VertebraTask


class FNetDataSource(EBC):
    USE_RAM_CACHE_FOR_CV_SETS = True
    cv_sets_ram_cache: Dict[tuple, hiwi.ImageList] = {}

    def num_folds(self) -> int:
        raise NotImplementedError('Abstract method')

    def cv_sets_cache_key(self, fold_id: int):
        return (type(self), fold_id)

    def folds_for_cv_setting(self) -> List[int]:
        return list(range(self.num_folds()))

    def load_fold_by_id(self, fold_id: int) -> hiwi.ImageList:
        raise NotImplementedError('Abstract method')

    def ndim(self) -> int:
        raise NotImplementedError('Abstract method')

    def _cached_load_fold_by_id(self, fold_id: int) -> hiwi.ImageList:
        if self.USE_RAM_CACHE_FOR_CV_SETS:
            k = self.cv_sets_cache_key(fold_id)
            if k not in self.cv_sets_ram_cache:
                self.cv_sets_ram_cache[k] = self.load_fold_by_id(fold_id)
            return self.cv_sets_ram_cache[k]
        else:
            return self.load_fold_by_id(fold_id)

    def load_test_dataset_by_fold_id(self, fold_id: int) -> hiwi.ImageList:
        return self._cached_load_fold_by_id(fold_id)

    def load_fold_by_ids(self, fold_ids: List[int]) -> hiwi.ImageList:
        iml = ImageList()
        for i in fold_ids:
            iml.extend(self._cached_load_fold_by_id(i))
        name = f'{self.name()}_fold_'
        for f in fold_ids:
            name += str(f)
        iml.name = name
        return iml

    def name(self) -> str:
        raise NotImplementedError('Abstract method')

    def val_fold_by_test_fold(self, test_fold_id: int) -> int:
        return val_fold_by_test_fold(test_fold_id, self.num_folds())

    def cv_training_and_validation_folds(self, test_fold_id: int, verbose=False, val_fold_id: Optional[int] = None) -> Tuple[hiwi.ImageList, hiwi.ImageList]:
        f = test_fold_id
        if val_fold_id is None:
            val_fold_id = self.val_fold_by_test_fold(f)
        assert val_fold_id in range(self.num_folds())
        train_fold_indices = [i for i in range(self.num_folds()) if i not in [test_fold_id, val_fold_id]]
        if verbose:
            logging.info(f'Testing on fold {f}')
            logging.info('Using fold {0} as validation set and folds {1} for training'.format(val_fold_id,
                                                                                              train_fold_indices))
        validation_dataset = self._cached_load_fold_by_id(val_fold_id)
        train_dataset = self.load_fold_by_ids(train_fold_indices)
        return train_dataset, validation_dataset

    def whole_dataset(self):
        raise NotImplementedError('Abstract method')

    def split_by_site(self, verbose=False) -> Tuple[hiwi.ImageList, hiwi.ImageList, hiwi.ImageList]:
        raise NotImplementedError('Abstract method')

    def official_test_train_validation_split(self, verbose=False) -> Tuple[hiwi.ImageList, hiwi.ImageList, hiwi.ImageList]:
        raise NotImplementedError('Abstract method')

    def randomly_split_training_and_validation_data_into_new_training_and_validation_sets(self, cv_test_fold: int,
                                                                                          training_samples: Optional[Tuple[str, str]] = None,
                                                                                          verbose=False) -> Tuple[hiwi.ImageList, hiwi.ImageList]:
        if verbose:
            logging.info('Using random train/validation split')
        whole_dataset = ImageList()
        for i in range(self.num_folds()):
            if i != cv_test_fold:
                whole_dataset.extend(self._cached_load_fold_by_id(i))
        if training_samples is None:
            train_dataset, validation_dataset = random_dataset_split(whole_dataset=whole_dataset,
                                                                     first_part_fraction=2 / 3)
            return train_dataset, validation_dataset
        else:
            train_dataset = ImageList()
            validation_dataset = ImageList()
            training_patient_ids = set()
            if verbose:
                print('Not splitting randomly despite `random_validation_set = True`, but instead loading the training samples from training info '
                      'and using the remaining patients as validation set.')
            for patient_id, _ in training_samples:
                training_patient_ids.add(patient_id)
            for img in whole_dataset:
                if img['patient_id'] in training_patient_ids:
                    train_dataset.append(img)
                else:
                    validation_dataset.append(img)
            return train_dataset, validation_dataset


class DiagnostikBilanzDataSource(FNetDataSource):
    def num_folds(self) -> int:
        return 4

    def name(self) -> str:
        return 'diagnostik_bilanz'

    def load_fold_by_id(self, fold_id: int) -> hiwi.ImageList:
        from ct_dirs import DERIVED_PATH
        return hiwi.ImageList.load(os.path.join(DERIVED_PATH, self.iml_subdir(), f'clean_fold_{fold_id}.iml'))

    def ndim(self) -> int:
        return 3

    def iml_subdir(self):
        return 'original_spacing'

    def whole_dataset(self):
        from ct_dirs import DERIVED_PATH
        return hiwi.ImageList.load(os.path.join(DERIVED_PATH, self.iml_subdir(), f'clean_patients.iml'))

    def split_by_site(self, verbose=False) -> Tuple[hiwi.ImageList, hiwi.ImageList, hiwi.ImageList]:
        from ct_dirs import DERIVED_PATH
        sites_1_and_5 = hiwi.ImageList.load(os.path.join(DERIVED_PATH, self.iml_subdir(), 'clean_training_patients.iml'))
        train_dataset, validation_dataset = random_dataset_split(whole_dataset=sites_1_and_5,
                                                                 first_part_fraction=2 / 3)
        test_dataset = hiwi.ImageList.load(os.path.join(DERIVED_PATH, self.iml_subdir(), f'clean_validation_patients.iml'))
        if verbose:
            logging.info('Testing on sites 3, 4, 6, 8, 9 and training on sites 1, 5.')
        return test_dataset, train_dataset, validation_dataset

    def official_test_train_validation_split(self, verbose=False) -> Tuple[hiwi.ImageList, hiwi.ImageList, hiwi.ImageList]:
        return self.split_by_site(verbose)


class DiagnostikBilanzScoutDataSource(DiagnostikBilanzDataSource):
    def iml_subdir(self):
        return 'diagbilanz_scouts'

    def name(self):
        return 'diagbilanz_scouts'

    def ndim(self) -> int:
        return 2


class MrOsScoutDataSource(DiagnostikBilanzDataSource):
    def __init__(self,
                 manual_coordinates_only=False,
                 use_localizer: str = None):
        self.manual_coordinates_only = manual_coordinates_only
        self.use_localizer = use_localizer

    def cv_sets_cache_key(self, fold_id: int):
        return (type(self), fold_id, self.manual_coordinates_only)

    def num_folds(self) -> int:
        return 4

    def name(self) -> str:
        suffix = '_annotated_coordinates' if self.manual_coordinates_only else ''
        suffix += self.localizer_suffix()
        return self.iml_subdir() + suffix

    def localizer_suffix(self):
        return f'_{self.use_localizer}' if self.use_localizer is not None else ''

    def load_fold_by_id(self, fold_id: int) -> hiwi.ImageList:
        iml_name = f'train_{fold_id}{self.localizer_suffix()}.iml'
        return self.load_iml(iml_name)

    def load_iml(self, iml_name: str):
        logging.info(f'Loading {iml_name} ...')
        from ct_dirs import DERIVED_PATH
        base_iml = hiwi.ImageList.load(os.path.join(DERIVED_PATH, self.iml_subdir(), iml_name))
        if self.manual_coordinates_only:
            annotated_ids = self.annotated_patient_ids()
            result = hiwi.ImageList([img for img in base_iml if img['patient_id'] in annotated_ids])
            result.name = base_iml.name + '_annotated_coordinates'
            return result
        else:
            return base_iml

    def ndim(self) -> int:
        return 2

    @cachetools.cached(cache=cachetools.LRUCache(maxsize=1), key=lambda self: type(self))
    def annotated_patient_ids(self) -> Set[str]:
        from ct_dirs import DERIVED_PATH
        return set(img['patient_id']
                   for img in hiwi.ImageList.load(os.path.join(DERIVED_PATH, self.iml_subdir(), 't_with_manually_annotated_coordinates.iml')))

    def iml_subdir(self):
        return 'mros_scouts'

    @cachetools.cached(cachetools.LRUCache(maxsize=1), lambda self: type(self))
    def whole_dataset(self):
        return self.load_iml(f't_full{self.localizer_suffix()}.iml')

    def load_test_dataset(self):
        from ct_dirs import DERIVED_PATH
        logging.info(f'Loading test data from {self.iml_subdir()} ...')
        base_iml = hiwi.ImageList.load(os.path.join(DERIVED_PATH, self.iml_subdir(), f'test{self.localizer_suffix()}.iml'))
        if self.manual_coordinates_only:
            annotated_ids = self.annotated_patient_ids()
            return hiwi.ImageList([img for img in base_iml if img['patient_id'] in annotated_ids])
        else:
            return base_iml

    def split_by_site(self, verbose=False) -> Tuple[hiwi.ImageList, hiwi.ImageList, hiwi.ImageList]:
        raise NotImplementedError('TODO')

    def official_test_train_validation_split(self, verbose=False) -> Tuple[hiwi.ImageList, hiwi.ImageList, hiwi.ImageList]:
        raise ValueError('Official split is the cross-validation split')


class AGESScoutDataSource(MrOsScoutDataSource):
    def iml_subdir(self):
        return 'ages_scouts'

    @cachetools.cached(cache=cachetools.LRUCache(maxsize=1), key=lambda self: type(self))
    def annotated_patient_ids(self) -> Set[str]:
        raise NotImplementedError('Dont have coordinates on the AGES dataset (yet). '
                                  'When we have them we can probably just use the same method as for MrOS.')



class ImageListDataSource(FNetDataSource):
    def __init__(self, image_list: hiwi.ImageList, iml_name: str, num_dims: int):
        self.iml_name = iml_name
        self.image_list = image_list
        self.num_dims = num_dims

    def cv_sets_cache_key(self, fold_id: int):
        return (type(self), fold_id, self.iml_name)

    def num_folds(self) -> int:
        return 1

    def name(self) -> str:
        return self.iml_name

    def load_fold_by_id(self, fold_id: int) -> hiwi.ImageList:
        assert fold_id == 0
        return self.image_list

    def ndim(self) -> int:
        return self.num_dims

    def whole_dataset(self):
        return self.image_list

    def split_by_site(self, verbose=False) -> Tuple[hiwi.ImageList, hiwi.ImageList, hiwi.ImageList]:
        raise NotImplementedError('Abstract method')

    def official_test_train_validation_split(self, verbose=False) -> Tuple[hiwi.ImageList, hiwi.ImageList, hiwi.ImageList]:
        raise NotImplementedError('Abstract method')


class TraumaDataSource(FNetDataSource):
    def num_folds(self) -> int:
        return 5

    def folds_for_cv_setting(self):
        return [0, 1, 2, 3]

    def name(self) -> str:
        return 'trauma'

    def official_test_train_validation_split(self, verbose=False) -> Tuple[hiwi.ImageList, hiwi.ImageList, hiwi.ImageList]:
        raise NotImplementedError()

    @staticmethod
    def dataset():
        from annotation_loader import load_spine_dataset
        return load_spine_dataset()

    def whole_dataset(self):
        return self.dataset().as_image_list()

    def load_fold_by_id(self, fold_id: int) -> hiwi.ImageList:
        return self.dataset().folds_as_image_lists()[fold_id]

    def ndim(self) -> int:
        return 3

    def load_test_dataset_by_fold_id(self, fold_id: int) -> hiwi.ImageList:
        return self._cached_load_fold_by_id(4)

    def val_fold_by_test_fold(self, test_fold_id: int) -> int:
        return test_fold_id

    def cv_training_and_validation_folds(self, test_fold_id: int, verbose=False) -> Tuple[hiwi.ImageList, hiwi.ImageList]:
        # unlike other datasets, we do not rotate the test fold here but instead use the 5th fold as test fold
        # thus there are three training folds and the validation fold is the one determined by the `test_fold_id` parameter
        val_fold_id, test_fold_id = test_fold_id, 4
        assert val_fold_id != test_fold_id
        train_fold_indices = [i for i in range(self.num_folds()) if i not in [test_fold_id, val_fold_id]]
        if verbose:
            logging.info(f'Testing on fold 4')
            logging.info(f'Using fold {val_fold_id} as validation set and folds {train_fold_indices} for training')
        validation_dataset = self._cached_load_fold_by_id(val_fold_id)
        train_dataset = self.load_fold_by_ids(train_fold_indices)
        return train_dataset, validation_dataset

    def split_by_site(self, verbose=False) -> Tuple[hiwi.ImageList, hiwi.ImageList, hiwi.ImageList]:
        raise RuntimeError('There is only a single site in the trauma dataset.')


class VerseDataSource(FNetDataSource):
    def num_folds(self) -> int:
        raise NotImplementedError()

    def name(self) -> str:
        return 'verse'

    def load_fold_by_id(self, fold_id: int) -> hiwi.ImageList:
        raise NotImplementedError()

    def ndim(self) -> int:
        return 3

    def load_test_fold(self):
        raise NotImplementedError()

    def whole_dataset(self):
        result = ImageList()
        for s in self.official_test_train_validation_split():
            result.extend(s)
        return result

    def split_by_site(self, verbose=False) -> Tuple[hiwi.ImageList, hiwi.ImageList, hiwi.ImageList]:
        raise NotImplementedError()

    def official_test_train_validation_split(self, verbose=False) -> Tuple[hiwi.ImageList, hiwi.ImageList, hiwi.ImageList]:
        from ct_dirs import DERIVED_PATH

        test_dataset = hiwi.ImageList.load(os.path.join(DERIVED_PATH, 'verse', 'verse_test.iml'))
        train_dataset = hiwi.ImageList.load(os.path.join(DERIVED_PATH, 'verse', 'verse_train.iml'))
        validation_dataset = hiwi.ImageList.load(os.path.join(DERIVED_PATH, 'verse', 'verse_validation.iml'))

        return test_dataset, train_dataset, validation_dataset


class TrainAndTestOnSeparateDataSources(FNetDataSource):
    def __init__(self,
                 training_data_source: FNetDataSource,
                 test_data_source: FNetDataSource,
                 validation_from_training: bool):
        super().__init__()
        self.training_data_source = training_data_source
        self.test_data_source = test_data_source
        self.validation_from_training = validation_from_training

    def _cached_load_fold_by_id(self, fold_id: int) -> hiwi.ImageList:
        raise NotImplementedError('Cross validation with multiple datasets is not possible (yet?).')

    def num_folds(self) -> int:
        raise NotImplementedError('Cross validation with multiple datasets is not possible (yet?).')

    def load_fold_by_id(self, fold_id: int) -> hiwi.ImageList:
        raise NotImplementedError('Cross validation with multiple datasets is not possible (yet?).')

    def load_test_dataset_by_fold_id(self, fold_id: int) -> hiwi.ImageList:
        raise NotImplementedError('Cross validation with multiple datasets is not possible (yet?).')

    def ndim(self) -> int:
        assert self.training_data_source.ndim() == self.test_data_source.ndim()
        return self.training_data_source.ndim()

    def name(self) -> str:
        return 'train_on_{}_and_validate_on_{}'.format(self.training_data_source.name(), self.test_data_source.name())

    def official_test_train_validation_split(self, verbose=False) -> Tuple[hiwi.ImageList, hiwi.ImageList, hiwi.ImageList]:
        if self.validation_from_training:
            test_dataset_1, train_dataset_1, validation_dataset_1 = self.training_data_source.official_test_train_validation_split()

            train_dataset = train_dataset_1 + test_dataset_1
            test_dataset = self.test_data_source.whole_dataset()
            validation_dataset = validation_dataset_1
        else:
            test_dataset, train_dataset_1, validation_dataset_1 = self.test_data_source.official_test_train_validation_split()
            train_dataset = self.training_data_source.whole_dataset()
            validation_dataset = train_dataset_1 + validation_dataset_1

        return test_dataset, train_dataset, validation_dataset

    def whole_dataset(self):
        whole_dataset = ImageList()
        for source in self.data_sources():
            whole_dataset += source.whole_dataset()
        return whole_dataset

    def split_by_site(self, verbose=False) -> Tuple[hiwi.ImageList, hiwi.ImageList, hiwi.ImageList]:
        raise NotImplementedError('Split by site does not make sense here. No need to ever implement.')

    def load_test_fold(self):
        test_dataset, _, _ = self.test_data_source.official_test_train_validation_split()
        return test_dataset

    def data_sources(self):
        return [self.training_data_source, self.test_data_source]


class RestrictedDataSource(FNetDataSource):
    def num_folds(self) -> int:
        return self.base_source.num_folds()

    def load_fold_by_id(self, fold_id: int) -> hiwi.ImageList:
        return self.filter_iml(self.base_source.load_fold_by_id(fold_id))

    def ndim(self) -> int:
        return self.base_source.ndim()

    def filter_iml(self, iml: hiwi.ImageList):
        iml = copy.deepcopy(iml)
        for img in iml:
            for vertebra_name in list(img.parts):
                if not self.include_case(img, vertebra_name):
                    del img.parts[vertebra_name]
        return iml

    def filter_iml_tuple(self, imls: Tuple[hiwi.ImageList, ...]):
        return tuple(
            self.filter_iml(iml)
            for iml in imls
        )

    def name(self) -> str:
        return self.base_source.name() + '_' + self.restriction_name

    def whole_dataset(self):
        return self.filter_iml(self.base_source.whole_dataset())

    def split_by_site(self, verbose=False) -> Tuple[hiwi.ImageList, hiwi.ImageList, hiwi.ImageList]:
        return self.filter_iml_tuple(self.base_source.split_by_site(verbose=verbose))

    def official_test_train_validation_split(self, verbose=False) -> Tuple[hiwi.ImageList, hiwi.ImageList, hiwi.ImageList]:
        return self.filter_iml_tuple(
            self.base_source.official_test_train_validation_split(verbose=verbose)
        )

    def __init__(self,
                 base_source: FNetDataSource,
                 restriction_name: str):
        self.base_source = base_source
        self.restriction_name = restriction_name

    def cv_sets_cache_key(self, fold_id: int):
        return (type(self), self.base_source.cv_sets_cache_key(fold_id))

    def include_case(self, img: hiwi.Image, vertebra_name: str) -> bool:
        raise NotImplementedError('Abstract method')


class ClassRestrictedDataSource(RestrictedDataSource):
    def __init__(self,
                 base_source: FNetDataSource,
                 task: VertebraTask,
                 exclude_class_names: List[str]):
        if len(exclude_class_names) == 0:
            raise ValueError(exclude_class_names)
        super().__init__(base_source, restriction_name='excl_' + '_'.join(exclude_class_names))
        self.task = task
        self.exclude_class_names = exclude_class_names

    def cv_sets_cache_key(self, fold_id: int):
        return (type(self), self.base_source.cv_sets_cache_key(fold_id), self.task.output_layer_name(), *sorted(self.exclude_class_names))

    def include_case(self, img: hiwi.Image, vertebra_name: str):
        try:
            label = self.task.y_true_from_hiwi_image_and_vertebra(img, vertebra_name)
        except self.task.UnlabeledVertebra:
            return True
        class_name = self.task.class_name_of_label(label)
        return class_name not in self.exclude_class_names


class DataSourceCombination(FNetDataSource):
    def num_folds(self) -> int:
        fold_counts = set(len(ds.folds_for_cv_setting()) for ds in self.data_sources)
        if len(fold_counts) != 1:
            raise RuntimeError(fold_counts)
        return next(iter(fold_counts))

    def folds_for_cv_setting(self) -> List[int]:
        foldss = set(tuple(ds.folds_for_cv_setting()) for ds in self.data_sources)
        if len(foldss) != 1:
            raise RuntimeError(foldss)
        return list(next(iter(foldss)))

    def name(self) -> str:
        return '_plus_'.join([ds.name() for ds in self.data_sources])

    def official_test_train_validation_split(self, verbose=False) -> Tuple[hiwi.ImageList, hiwi.ImageList, hiwi.ImageList]:
        test_dataset_for_all_sources = ImageList()
        train_dataset_for_all_sources = ImageList()
        validation_dataset_for_all_sources = ImageList()
        for evaluator in self.data_sources:
            test_dataset, train_dataset, validation_dataset = evaluator.official_test_train_validation_split(verbose=verbose)
            test_dataset_for_all_sources += test_dataset
            train_dataset_for_all_sources += train_dataset
            validation_dataset_for_all_sources += validation_dataset
        return test_dataset_for_all_sources, train_dataset_for_all_sources, validation_dataset_for_all_sources

    def load_fold_by_id(self, fold_id: int) -> hiwi.ImageList:
        dataset = ImageList()
        for source in self.data_sources:
            dataset += source.load_fold_by_id(fold_id)
        dataset.name = f'{self.name()}_fold_{fold_id}'
        return dataset

    def ndim(self) -> int:
        assert len(set(ds.ndim() for ds in self.data_sources)) == 1, set(ds.ndim() for ds in self.data_sources)
        return self.data_sources[0].ndim()

    def load_test_dataset_by_fold_id(self, fold_id: int) -> hiwi.ImageList:
        dataset = ImageList()
        for source in self.data_sources:
            dataset += source.load_test_dataset_by_fold_id(fold_id)
        dataset.name = f'{self.name()}_test_fold_{fold_id}'
        return dataset

    def cv_training_and_validation_folds(self, test_fold_id: int, verbose=False) -> Tuple[hiwi.ImageList, hiwi.ImageList]:
        training_dataset_for_all_sources = ImageList()
        validation_dataset_for_all_sources = ImageList()
        for source in self.data_sources:
            training_dataset, validation_dataset = source.cv_training_and_validation_folds(test_fold_id, verbose=verbose)
            training_dataset_for_all_sources += training_dataset
            validation_dataset_for_all_sources += validation_dataset
        training_dataset_for_all_sources.name = f'{self.name()}_train_sets_{test_fold_id}'
        validation_dataset_for_all_sources.name = f'{self.name()}_val_sets_{test_fold_id}'
        return training_dataset_for_all_sources, validation_dataset_for_all_sources

    def split_by_site(self, verbose=False) -> Tuple[hiwi.ImageList, hiwi.ImageList, hiwi.ImageList]:
        test_dataset_for_all_evaluators = ImageList()
        train_dataset_for_all_evaluators = ImageList()
        validation_dataset_for_all_evaluators = ImageList()
        for evaluator in self.data_sources:
            test_dataset, train_dataset, validation_dataset = evaluator.split_by_site(verbose=verbose)
            test_dataset_for_all_evaluators += test_dataset
            train_dataset_for_all_evaluators += train_dataset
            validation_dataset_for_all_evaluators += validation_dataset
        return test_dataset_for_all_evaluators, train_dataset_for_all_evaluators, validation_dataset_for_all_evaluators

    def __init__(self, data_sources: List[FNetDataSource]):
        super().__init__()
        self.data_sources = data_sources

    def cv_sets_cache_key(self, fold_id: int):
        return (type(self), *[s.cv_sets_cache_key(fold_id) for s in self.data_sources])

    def whole_dataset(self):
        whole_dataset = ImageList()
        for evaluator in self.data_sources:
            whole_dataset += evaluator.whole_dataset()
        whole_dataset.name = self.name()
        return whole_dataset
