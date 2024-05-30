"""
.. include:: ../README.md
   :start-line: 3
"""


__all__ = (
    'batchize',
    'dist',
    'change_anatomical_orientation',
    'find_local_max',
    'find_local_max_recursively',
    'refine_local_peak',
    'guess_image_shape',
    'itk_image_to_sitk_image',
    'find_anatomical_orientation',
    'load_image',
    'place_gaussian',
    'resample_image',
    'save_image',
    'sitk_image_to_itk_image',
    'show_logs',
    'write_logs',
    'write_pointset_file',
    'transform_elastically',
    'AvgEvaluation',
    'Dashboard',
    'Evaluation',
    'Image',
    'ImageList',
    'LocalMaxLocator',
    'Object',
    'PatchExtractor',
    'Watchdog',
    'WorkingDirectory'
)


__pdoc__ = {
    'cli': False,
    'dashboard': False,
    'eval': False,
    'feature': False,
    'image': False,
    'training': False,
    'utils': False,
    'watchdog': False
}


from .dashboard import Dashboard
from .eval import AvgEvaluation, Evaluation
from .feature import find_local_max, find_local_max_recursively
from .feature import refine_local_peak, LocalMaxLocator
from .image import Image, ImageList, Object, load_image, save_image
from .image import find_anatomical_orientation, resample_image
from .image import change_anatomical_orientation
from .image import sitk_image_to_itk_image, itk_image_to_sitk_image
from .training import PatchExtractor, batchize, place_gaussian
from .training import transform_elastically
from .utils import WorkingDirectory, guess_image_shape, show_logs, \
    write_logs, dist, write_pointset_file
from .watchdog import Watchdog
