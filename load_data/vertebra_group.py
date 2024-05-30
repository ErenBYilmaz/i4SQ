import copy
from math import inf
from typing import List

import numpy

from hiwi import Image, ImageList
from lib.util import EBC
from load_data import VERTEBRAE
from tasks import VertebraClassificationTask, VertebraTasks, AgeRegressionTask

MetricName = GroupName = str


class Group(EBC):
    def __init__(self, name: GroupName, above_patient_level: bool):
        self.above_patient_level = above_patient_level
        self.name = name

    def contains_vertebra(self, img: Image, vertebra_name: str) -> bool:
        raise NotImplementedError

    def filter_iml(self, image_list: ImageList) ->ImageList:
        results = copy.deepcopy(image_list)
        for img_idx in reversed(range(len(results))):
            img = results[img_idx]
            for vertebra in list(img.parts):
                if not self.contains_vertebra(img, vertebra):
                    del img.parts[vertebra]
            if len(img.parts) == 0:
                del results[img_idx]
        if hasattr(results, 'name'):
            results.name = f'{results.name}_{self.name}'
        return results

    @staticmethod
    def defaults() -> List['Group']:
        return [
            WholeDataset()
        ]


class WholeDataset(Group):
    def __init__(self, name='Whole Dataset'):
        super().__init__(name=name, above_patient_level=True)

    def contains_vertebra(self, img: Image, vertebra_name: str) -> bool:
        return True


class VertebraLevel(Group):
    def __init__(self, vertebra: str):
        super().__init__(name=vertebra, above_patient_level=False)
        self.vertebra = vertebra

    def contains_vertebra(self, img: Image, vertebra_name: str) -> bool:
        return vertebra_name == self.vertebra

    @staticmethod
    def all_groups() -> List['Group']:
        return [
            VertebraLevel(vertebra)
            for vertebra in VERTEBRAE
        ]


class MultipleVertebraLevels(Group):
    def __init__(self, vertebrae: List[str]):
        super().__init__(name=self.group_name(vertebrae), above_patient_level=False)
        for v in vertebrae:
            if v not in VERTEBRAE:
                raise ValueError
        self.vertebrae = vertebrae

    @staticmethod
    def group_name(vertebrae):
        return ' + '.join(vertebrae)

    def contains_vertebra(self, img: Image, vertebra_name: str) -> bool:
        return vertebra_name in self.vertebrae


class Sex(Group):
    def __init__(self, sex: str):
        super().__init__(name=sex, above_patient_level=True)

    def normalize(self, name: str):
        return name.lower().strip()

    def contains_vertebra(self, img: Image, vertebra_name: str) -> bool:
        if 'dicom_metadata' in img:
            if self.normalize(img['dicom_metadata']['0010|0040']) == self.normalized_name():
                return True
        if '0010|0040' in img:
            if self.normalize(img['0010|0040']) == self.normalized_name():
                return True
        if 'PatientSex' in img and self.normalize(img['PatientSex']) == self.normalized_name():
            return True
        return False

    def normalized_name(self):
        return self.normalize(self.name)

    @staticmethod
    def all_sexes():
        return [
            Sex('O'),
            Sex('F'),
            Sex('W'),
            Sex('M'),
        ]


class AgeRange(Group):
    UPPER_AGE_BOUND = 150

    def __init__(self, start_age, stop_age):
        super().__init__(name=self.group_name(start_age, stop_age), above_patient_level=True)
        self.start_age = start_age
        self.stop_age = stop_age
        self.task = AgeRegressionTask(1)

    @staticmethod
    def group_name(start_age, stop_age):
        return f'{start_age}Y <= Age < {stop_age}Y'

    def contains_vertebra(self, img: Image, vertebra_name: str) -> bool:
        try:
            age = self.task.y_true_from_hiwi_image_and_vertebra(img, vertebra_name).item()
        except self.task.UnlabeledVertebra:
            return False
        return self.start_age <= age < self.stop_age

    @staticmethod
    def groups_of_size(size: int):
        return [AgeRange(-inf, 0)] + [
            AgeRange(min_age, min_age + size)
            for min_age in range(0, AgeRange.UPPER_AGE_BOUND, size)
        ]


class VertebraRange(MultipleVertebraLevels):
    def __init__(self, start: str, end: str, step: int = 1):
        self.start = start
        self.end = end
        self.step = step
        super().__init__(VERTEBRAE[VERTEBRAE.index(start):VERTEBRAE.index(end) + 1:step])

    def group_name(self, vertebrae):
        assert self.start in vertebrae
        assert self.end in vertebrae
        result = f'{self.start} - {self.end}'
        if self.step != 1:
            result += f' (step {self.step})'
        return result

    def contains_vertebra(self, img: Image, vertebra_name: str) -> bool:
        return vertebra_name in self.vertebrae

    @staticmethod
    def relevant_groups():
        return [
            VertebraRange('C1', 'L6'),  # very different from other vertebrae
            VertebraRange('C1', 'C2'),  # very different from other vertebrae
            VertebraRange('C3', 'L6'),  # excluding above
            VertebraRange('T4', 'L4'),  # range in the Diagnostik-Bilanz dataset
            VertebraRange('C1', 'T4'),  # rather small compared to other vertebrae
            VertebraRange('C1', 'C7'),
            VertebraRange('T1', 'T12'),
            VertebraRange('L1', 'L6'),
            VertebraRange('T1', 'L6'),
        ]


class VertebraClass(Group):
    def __init__(self, task: VertebraClassificationTask, class_name):
        super().__init__(name=self.group_name(class_name), above_patient_level=False)
        self.class_name = class_name
        self.task = task

    def contains_vertebra(self, img: Image, vertebra_name: str) -> bool:
        try:
            label = self.task.y_true_from_hiwi_image_and_vertebra(img, vertebra_name)
        except self.task.UnlabeledVertebra:
            return False
        try:
            return self.task.class_name_of_label(label) == self.class_name
        except self.task.LabelWithoutClass:
            return False

    @staticmethod
    def group_name(class_name):
        return f'{class_name} (Vertebra level)'

    @staticmethod
    def from_tasks(tasks: VertebraTasks) -> List['Group']:
        return [
            VertebraClass(task, class_name)
            for task in tasks
            if isinstance(task, VertebraClassificationTask)
            for class_name in task.class_names()
        ]


class PatientClass(Group):
    def __init__(self, task: VertebraClassificationTask, class_name):
        super().__init__(name=self.group_name(class_name), above_patient_level=True)
        self.class_name = class_name
        self.task = task

    def contains_vertebra(self, img: Image, vertebra_name: str) -> bool:
        labels = []
        for name in img.parts:
            try:
                labels.append(self.task.y_true_from_hiwi_image_and_vertebra(img, name))
            except self.task.UnlabeledVertebra:
                continue
            except self.task.ExcludedVertebra:
                return False
        if len(labels) == 0:
            return False # no annotated vertebrae?
        patient_level_label = self.task.patient_level_aggregation(numpy.array(labels))
        try:
            patient_level_label_name = self.task.class_name_of_label(patient_level_label)
            if patient_level_label_name in ['1', '3']:
                raise RuntimeError('A patient level genant score should only take the values 0 or 2 '
                                   f'meaning "<2" and ">=2", respectively. Was {patient_level_label_name}')
            return patient_level_label_name == self.class_name
        except self.task.UnlabeledVertebra as e:
            raise e.ThrownByClassIdxOfLabel()
        except self.task.LabelWithoutClass:
            return False

    @staticmethod
    def group_name(class_name):
        return f'{class_name} (Patient level)'

    @staticmethod
    def from_tasks(tasks: VertebraTasks) -> List['Group']:
        return [
            PatientClass(task, class_name)
            for task in tasks
            if isinstance(task, VertebraClassificationTask)
            for class_name in task.class_names()
        ]


class TraumaCVFold(Group):
    def __init__(self, fold_idx: int):
        super().__init__(name=f'Trauma Fold {fold_idx}', above_patient_level=True)
        self.fold_idx = int(fold_idx)

    def contains_vertebra(self, img: Image, vertebra_name: str) -> bool:
        from annotation_loader import CVFoldLoader
        return CVFoldLoader.cv_fold_of_study(study_description=img['patient_id']) == self.fold_idx

    @staticmethod
    def all_folds():
        try:
            from annotation_loader import CVFoldLoader
        except ImportError:
            return []
        return [
            TraumaCVFold(fold_idx)
            for fold_idx in CVFoldLoader.cv_folds
        ]


class DiagBilanzCVFold(Group):
    def __init__(self, fold_idx: int):
        super().__init__(name=f'DiagBilanz Fold {fold_idx}', above_patient_level=True)
        self.fold_idx = int(fold_idx)

    def contains_vertebra(self, img: Image, vertebra_name: str) -> bool:
        from train_test_split import CROSS_VALIDATION_SETS, patient_number_from_long_string
        for dcm_path in CROSS_VALIDATION_SETS[self.fold_idx]:
            if patient_number_from_long_string(dcm_path) == img['patient_id']:
                return True
        return False

    @staticmethod
    def all_folds():
        from train_test_split import CROSS_VALIDATION_SETS
        return [
            DiagBilanzCVFold(fold_idx)
            for fold_idx in range(len(CROSS_VALIDATION_SETS))
        ]
