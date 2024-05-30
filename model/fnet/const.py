from typing import Optional

from lib import dl_backend
from model.fnet.data_source import DiagnostikBilanzDataSource, TraumaDataSource, DiagnostikBilanzScoutDataSource, \
    MrOsScoutDataSource, DataSourceCombination, AGESScoutDataSource
from tasks import VertebraTasks, tensorflow_available

# dl_backend.b().limit_memory_usage()



RECORD_PREDICTIONS = False
DATA_SOURCES = [
    # DiagnostikBilanzScoutDataSource(),
    # DiagnostikBilanzDataSource(),
    # TraumaDataSource(),
    # MrOsScoutDataSource(),
    # AGESScoutDataSource(use_localizer='yolo_20240429'),
    DataSourceCombination([MrOsScoutDataSource(manual_coordinates_only=True),
                           DiagnostikBilanzScoutDataSource()]),
]

TASKS = VertebraTasks([
    # *VertebraTasks.tasks_for_which_diagnostik_bilanz_has_annotations(),
    *VertebraTasks.tasks_for_which_diagnostik_bilanz_and_mros_scouts_have_annotations(),
    # *VertebraTasks.tasks_for_which_trauma_dataset_has_annotations()
])

LARGER_RESULT_IS_BETTER = False
# EXPERIMENT_NAME: str = 'perceiver_from_scratch'
# EXPERIMENT_NAME: str = 'perceiver_hp_search_step_2_longer_training'
# EXPERIMENT_NAME: str = 'scouts_2d_hyperparameter_search'
# EXPERIMENT_NAME: str = 'scouts_2d_hyperparameter_search_validation'
# EXPERIMENT_NAME: str = 'scouts_2d_20240513_trained_on_db'
# EXPERIMENT_NAME: str = 'scouts_2d_20240513_trained_on_mros'
EXPERIMENT_NAME: str = 'keras_model_zoo_20240522'
# EXPERIMENT_NAME: str = 'scouts_2d_20240513_trained_on_mros_plus_db'
# EXPERIMENT_NAME: str = 'ct_3d_dropout'
# EXPERIMENT_NAME: str = 'scouts_2d_dropout'
# EXPERIMENT_NAME: str = 'diagbilanz_hp_validation_fixed_rotation'
# EXPERIMENT_NAME: str = 'diagbilanz_scouts_scaling_tf'
assert EXPERIMENT_NAME is not None

FIXED_TRAIN_EPOCHS: Optional[int] = None  # use this in hyperparameter searches to prevent short training runs
