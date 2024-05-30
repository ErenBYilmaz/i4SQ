import os
from typing import List, Dict
from lib.tuned_cache import TunedMemory

metadata_cache = TunedMemory(location='./.cache/metadata', verbose=0)


@metadata_cache.cache
def image_files_for_ct(ct_dirs: List[str]) -> Dict[str, List[str]]:
    """
    calculates a dictionary to get all paths to the ct images

    :param ct_dirs: a dataset
    :return: a dictionary with the path of the ct images as key and the name of ct_dir + image name as the value of the
    dictionary
    """
    images_files_by_ct = {
        ct_dir: [ct_dir] if os.path.isfile(ct_dir) else [
            os.path.join(ct_dir, file_name)
            for file_name in next(os.walk(ct_dir))[2]  # returns all files in ct_dir
        ]
        for ct_dir in ct_dirs
    }
    return images_files_by_ct
