import importlib
import os
import random
from typing import List

from joblib import delayed

from lib.profiling_tools import fix_yappi_crash_before_loading_tensorflow
from model.fnet.model_analysis.build_model_application_cache import BuildModelApplicationCache
from model.fnet.model_analysis.dump_results_table import DumpResultsTable
from model.fnet.model_analysis.list_difficult_vertebrae import ListDifficultVertebrae, DifficultVertebraePlotter
from model.fnet.model_analysis.plot_augmentation_robustness import AugmentationRobustnessDrawer
from model.fnet.model_analysis.plot_calibration import CalibrationCurvePlotter
from model.fnet.model_analysis.plot_confusion import PlotConfusionMatrices, PlotConfusionOnPatientLevel
from model.fnet.model_analysis.plot_grouped_metrics import PlotPatientLevelGroupedMetrics, PlotGroupedMetrics
from model.fnet.model_analysis.plot_models import PlotModels
from model.fnet.model_analysis.plot_output_as_text_on_sagittal_slice import PlottingConfig, PlotOutputAsTextOnSagittalSlice
from model.fnet.model_analysis.plot_outputs import OutputsPlotter, PlotOutputsOnPatientLevel
from model.fnet.model_analysis.plot_roc_curves import PlotROCCurvesOnPatientLevel, PlotROCCurves, PlotFROCCurves
from model.fnet.model_analysis.plot_tsne import TSNEPlotter
from model.fnet.model_analysis.plot_tta import PlotTestTimeAugmentation, RetainedDataPlottingConfiguration
from model.fnet.model_analysis.store_mcdo_models import StoreMCDOModels
from model.fnet.model_analysis.store_raw_models import StoreRawModels

fix_yappi_crash_before_loading_tensorflow()

import lib.main_wrapper
import lib.scheduling
from lib.my_logger import logging
from lib.scheduling import BeforeDispatch
from lib.util import in_debug_mode
from model.fnet.const import EXPERIMENT_NAME, DATA_SOURCES
from model.fnet.data_source import MrOsScoutDataSource, DiagnostikBilanzScoutDataSource, DiagnostikBilanzDataSource, \
    DataSourceCombination, TraumaDataSource, AGESScoutDataSource
from model.fnet.evaluate import FNetParameterEvaluator
from model.fnet.model_analysis.analyze_trained_models import ModelAnalyzer, UseValidationSet, UseTrainSet, UseWholeSet, \
    UseTestSet
from model.fnet.model_analysis.evaluation_result_serializer import FileSystemSerializer
from model.fnet.model_analysis.export_outputs_for_survival_analysis import ExportOutputsForSurvivalAnalysis, BinaryClassificationTasksFilter, SpecificTaskFilter, \
    NonArtificialTasksFilter, AndFilter, NoTasksFilter
from tasks import IncidentSpineFractureClassification, TimeToFirstSpineFractureRegression, AgeRegressionTask, BMIRegressionTask, IncidentHipFractureClassification, TimeToFirstHipFractureRegression, \
    TimeToFirstFractureRegression, IncidentFractureClassification, IncidentHipOrSpineFractureClassification, TimeToFirstHipOrSpineFractureRegression

for model_module in ['model.fnet.fnet', 'model.pytorch_model.mnist_model']:
    try:
        importlib.import_module(model_module)  # might be needed for loading from configuration files
    except ImportError as e:
        logging.info(f'Not importing {model_module} due to {type(e).__name__}: {e}')

try:
    from annotation_supplement_ey import LIKELY_DIFFICULTIES
except ImportError:
    LIKELY_DIFFICULTIES = None

try:
    from config import N_JOBS
except ImportError:
    N_JOBS = 3
if in_debug_mode():
    N_JOBS = 1
RANDOM_ANALYSIS_ORDER = N_JOBS > 1
RESULT_SERIALIZER = FileSystemSerializer()


@lib.main_wrapper.MainWrapper(re_raise_unkown_exceptions=True)
def main():
    FNetParameterEvaluator.import_known_model_types()
    ModelAnalyzer.DISABLE_INTERMEDIATE_ANALYSES = True
    ModelAnalyzer.RANDOM_MODEL_ORDER = RANDOM_ANALYSIS_ORDER
    ModelAnalyzer.RANDOM_SUBDIRECTORY_ORDER = RANDOM_ANALYSIS_ORDER
    # ModelAnalyzer.SKIP_EXISTING_BY_DEFAULT = RANDOM_ANALYSIS_ORDER
    # ModelAnalyzer.SKIP_EXISTING_BY_DEFAULT = False
    ModelAnalyzer.SKIP_EXISTING_BY_DEFAULT = True
    ModelAnalyzer.RESULT_SERIALIZER = RESULT_SERIALIZER
    use_datasets = [
        UseValidationSet(),
        # UseTrainSet(),
        # UseTestSet(),
        # UseWholeSet(),
    ]
    threshold_methods = [
        # 'svm',
        # 'sens_spec_prod',
        0.5,
        # 0.1,
        # 0.05,
    ]
    exclude_site_6 = [
        # True,
        False,
    ]
    csvs = {
        # r'C:\Users\Eren\Programme\philips-spinetool\wrapper\results\philips_round_3\summary\real_world_coordinates.csv': 'philips_round_3',
        None: None,  # COORDINATE_CSV,
    }
    # ModelAnalyzer.ONLY_TASKS = VertebraTasks.tasks_for_which_mros_has_annotations().output_layer_names()
    ModelAnalyzer.ONLY_TASKS = None

    assert len(DATA_SOURCES) == 1
    evaluators = [
        # FNetParameterEvaluator(DATA_SOURCES[0]),
        # FNetParameterEvaluator(AGESScoutDataSource(use_localizer='yolo_20240424')),
        # FNetParameterEvaluator(MrOsScoutDataSource(use_localizer='yolo_20240424')),
        FNetParameterEvaluator(MrOsScoutDataSource(manual_coordinates_only=True)),
        FNetParameterEvaluator(DiagnostikBilanzScoutDataSource()),
        # FNetParameterEvaluator(TraumaDataSource()),
        # FNetParameterEvaluator(DiagnostikBilanzDataSource()),
        FNetParameterEvaluator(DataSourceCombination([MrOsScoutDataSource(manual_coordinates_only=True), DiagnostikBilanzScoutDataSource()])),
    ]
    if RANDOM_ANALYSIS_ORDER:
        random.shuffle(threshold_methods)
        random.shuffle(use_datasets)
        random.shuffle(exclude_site_6)
        random.shuffle(evaluators)
    for evaluator in evaluators:
        for ModelAnalyzer.use_dataset in use_datasets:
            for ModelAnalyzer.EXCLUDE_SITE_6 in exclude_site_6:
                for csv_path, csv_name in csvs.items():
                    for ModelAnalyzer.DEFAULT_THRESHOLD in threshold_methods:
                        model_dir = fr'models/scouts_2d_20240513'
                        # model_dir = fr'models/{EXPERIMENT_NAME}'
                        analyses: List[ModelAnalyzer] = [
                            # *BuildModelApplicationCache.both(evaluator),
                            # DumpResultsTable(evaluator=evaluator),


                            StoreRawModels(evaluator=evaluator),
                            PlotModels(evaluator=evaluator),
                            PlotConfusionMatrices(evaluator=evaluator, model_level_plotting=False),
                            PlotOutputsOnPatientLevel(evaluator=evaluator, model_level_plotting=False),
                            PlotConfusionOnPatientLevel(evaluator=evaluator, model_level_plotting=False),
                            PlotROCCurves(evaluator=evaluator, model_level_plotting=False),
                            OutputsPlotter(evaluator=evaluator, model_level_plotting=False),
                            PlotROCCurvesOnPatientLevel(evaluator=evaluator, model_level_plotting=False),
                            # PlotOutputAsTextOnSagittalSlice(evaluator=evaluator, model_level_plotting=False,
                            #                                 plotting_config=PlottingConfig.for_asbmr_2023()),
                            # PlotGroupedMetrics(evaluator=evaluator),
                            # PlotPatientLevelGroupedMetrics(evaluator=evaluator, only_json=True),
                            CalibrationCurvePlotter(evaluator=evaluator, model_level_plotting=False),
                            *[
                                ListDifficultVertebrae(evaluator, likely_difficulties=LIKELY_DIFFICULTIES, task_idx=task_idx)
                                for task_idx in range(ListDifficultVertebrae.max_tasks_per_model)
                                # for task_idx in range(2)
                            ],

                            # ----------------------------------------

                            # PlotOutputAsTextOnSagittalSlice(evaluator=evaluator, model_level_plotting=False),
                            # DifficultVertebraePlotter(evaluator,
                            #                           to_subdir='all_vertebrae',
                            #                           difficult_vertebrae_table=None,
                            #                           add_output_info_for_task_idx=None,
                            #                           binary_threshold_or_method=0.5),
                            # PlotTestTimeAugmentation(evaluator=evaluator, model_level_plotting=False,
                            #                          retained_data_plots=[],
                            #                          generator_modifiers=PlotTestTimeAugmentation.default_modifiers_for_2d()),

                            # AugmentationRobustnessDrawer(evaluator=evaluator, generator_modifiers=AugmentationRobustnessDrawer.mcdo_modifiers_for_2d()),
                            # AugmentationRobustnessDrawer(evaluator=evaluator, generator_modifiers=AugmentationRobustnessDrawer.default_modifiers_for_2d()),

                            # StoreMCDOModels(evaluator=evaluator),
                            # PlotTestTimeAugmentation(evaluator=evaluator, model_level_plotting=True,
                            #                          retained_data_plots=RetainedDataPlottingConfiguration.example_configurations(),
                            #                          generator_modifiers=PlotTestTimeAugmentation.best_known_modifiers()),
                            # PlotTestTimeAugmentation(evaluator=evaluator, model_level_plotting=False,
                            #                          retained_data_plots=[],
                            #                          generator_modifiers=PlotTestTimeAugmentation.default_modifiers()),
                            # PlotOutputAsTextOnSagittalSlice(evaluator=evaluator, model_level_plotting=False,
                            #                                 plotting_config=PlottingConfig.for_spie()),
                            # *[ExportOutputsForSurvivalAnalysis(evaluator,
                            #                                    patient_level_binary_prognosis_task_for_event_flag=output_task_pair[0],
                            #                                    patient_level_tte_task=output_task_pair[1],
                            #                                    task_exclusion_filter=task_filter,
                            #                                    skip_existing=False,
                            #                                    extra_input_tasks=extra_inputs,
                            #                                    standardize_features=False,
                            #                                    group_by_patient=group_by_patient,
                            #                                    patient_level_aggregation_method='max')
                            #   for group_by_patient in [True, False]
                            #   for task_filter in [NonArtificialTasksFilter(),
                            #                       AndFilter([NonArtificialTasksFilter(), BinaryClassificationTasksFilter()]),
                            #                       SpecificTaskFilter('ifo'),
                            #                       NoTasksFilter(),
                            #                       SpecificTaskFilter('gsdz0v123'),
                            #                       SpecificTaskFilter('gsdz01v23'),
                            #                       SpecificTaskFilter('gsdz012v3'), ]
                            #   for extra_inputs in [
                            #       [AgeRegressionTask(1.0)],
                            #       [AgeRegressionTask(1.0), BMIRegressionTask(1.0)],
                            #       [],
                            #   ]
                            #   for output_task_pair in [
                            #         (IncidentSpineFractureClassification(1.0), TimeToFirstSpineFractureRegression(1.0)),
                            #         (IncidentHipFractureClassification(1.0), TimeToFirstHipFractureRegression(1.0)),
                            #         (IncidentFractureClassification(1.0), TimeToFirstFractureRegression(1.0)),
                            #         (IncidentHipOrSpineFractureClassification(1.0), TimeToFirstHipOrSpineFractureRegression(1.0)),
                            #         # (IncidentNonTraumaticSpineFractureClassification(1.0), TimeToFirstNonTraumaticSpineFractureRegression(1.0)),
                            #   ]],
                            # TSNEPlotter(evaluator, also_plot_at_input_layer=False, exclude_binary_tasks=False),
                            # CompareOutputs(evaluator=evaluator),
                            # PlotFROCCurves(evaluator=evaluator),
                            # *[
                            #     DifficultVertebraePlotter(evaluator,
                            #                               to_subdir='all_vertebrae',
                            #                               save_as_nifti=False,
                            #                               difficult_vertebrae_table=None,
                            #                               add_output_info_for_task_idx=task_idx)
                            #     for task_idx in range(ListDifficultVertebrae.max_tasks_per_model)
                            # ],
                            # GeneratePseudoLabels(evaluator=FNetParameterEvaluator.default_evaluator(), dataset=FNetParameterEvaluator(DiagnostikBilanzDataSource()).whole_dataset(), name='clean_patients'),
                            # *[
                            #     ListVertebraUsefulness(evaluator, likely_difficulties=None, task_idx=task_idx)
                            #     for task_idx in range(ListDifficultVertebrae.max_tasks_per_model)
                            # ],
                            # AugmentationRobustnessDrawer(evaluator=evaluator),
                        ]
                        if RANDOM_ANALYSIS_ORDER:
                            analyses = analyses.copy()
                            random.shuffle(analyses)
                        for analysis in analyses:
                            if N_JOBS == 1:
                                logging.info(f'Running {analysis.name()}')
                            analysis.coordinate_csv_path = csv_path
                            analysis.coordinate_csv_name = csv_name
                            analysis.analyze_subdirectories(model_dir)


if __name__ == '__main__':
    # lib.main_wrapper.ENABLE_PROFILING = True
    # ModelAnalyzer.clear_directory_level_cache_unless_debugging()
    # ModelAnalyzer.clear_caches()
    print(f'Starting Parallel backend with {N_JOBS} jobs')
    schedule = [delayed(main)() for job_idx in range(N_JOBS)]
    with lib.scheduling.MyParallel(n_jobs=N_JOBS, prefer='processes', verbose=20, batch_size=1, max_nbytes=None,
                                   reuse=True, before_dispatch=BeforeDispatch(N_JOBS).before_dispatch) as parallel:
        parallel(schedule)
    RESULT_SERIALIZER.commit()
