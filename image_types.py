import os
from typing import List


def is_dcm_dir(dir_name):
    if not os.path.isdir(dir_name):
        return False
    for filename in os.listdir(dir_name):
        if os.path.isfile(os.path.join(dir_name, filename)) and filename.endswith('.dcm'):
            return True
    return False


def is_mhd_file(path: str) -> bool:
    return os.path.isfile(path) and path.endswith('.mhd')


def is_dcm_file(path: str) -> bool:
    return os.path.isfile(path) and path.endswith('.dcm')


def is_ct_dataset(dataset: List[str]) -> bool:
    return all(is_dcm_dir(ct_path) for ct_path in dataset)


def is_mhd_dataset(dataset: List[str]) -> bool:
    return all(is_mhd_file(ct_path) for ct_path in dataset)
