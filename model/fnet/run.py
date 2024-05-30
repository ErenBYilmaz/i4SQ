import copy
import functools
import joblib.externals.loky
from math import inf
from numpy import DataSource
from typing import List

from lib.profiling_tools import fix_yappi_crash_before_loading_tensorflow

fix_yappi_crash_before_loading_tensorflow()

import lib.memory_control
import lib.scheduling
import lib.util
from lib import parameter_search, util
from lib.image_processing_tool import TrainableImageProcessingTool
from lib.main_wrapper import MainWrapper, main_wrapper
from lib.my_logger import logging
from lib.progress_bar import ProgressBar
from lib.scheduling import SerializableJob, BeforeDispatch
from model.fnet.const import EXPERIMENT_NAME, LARGER_RESULT_IS_BETTER, DATA_SOURCES, TASKS
from model.fnet.data_source import DataSourceCombination, MrOsScoutDataSource, DiagnostikBilanzScoutDataSource, FNetDataSource
from model.fnet.evaluate import FNetParameterEvaluator
from model.fnet.hyperparameter_set import FNetDefaultHyperparameterSet, FNetHyperparameterSetForScoutScans, SimplePytorchModelConfig, HyperparameterSet, PerceiverConfigPretrained, \
    PerceiverConfigFromScratch, FNetConfigForSendingToFaraz20231123, FNetWithBatchNorm, FNetFor3Epochs, SwinConfig, FNet2DForScoutScans, FNetFor45Epochs
from model.fnet.parameter_modifier import UsingDifferentParameters, UsingSameParameters, UsingDifferentParameterValue, MultipleDifferentParameterValues, ExperimentNameSuffix, \
    UsingAnotherHyperparameterSet
from model.keras_zoo_model.hyperparameter_set import KerasZooModelConfig
from tasks import VertebraTasks

JOBS_SUBDIR = 'fnet_run'
CLEAR_OLD_JOBS = True
FORCE_ADD_NEW_JOBS_EVEN_IF_THERE_ARE_OLD_JOBS_REMAINING = False
try:
    from config import N_JOBS
except ImportError:
    N_JOBS = 1  # if lib.util.in_debug_mode() else 12

RESULTS_ONLY = False
MIN_TRIES_BEFORE_DISPLAY = 4
NO_MUTATIONS_PROBABILITY = 1 / 10
MUTATION_PROBABILITY = 1 / 10
MAX_DISPLAY_RESULTS = 0
MAX_TABLE_ROW_COUNT = 50
print(f'''
CLEAR_OLD_JOBS = {CLEAR_OLD_JOBS}
RESULTS_ONLY = {RESULTS_ONLY}
MIN_TRIES_BEFORE_DISPLAY = {MIN_TRIES_BEFORE_DISPLAY}
NO_MUTATIONS_PROBABILITY = {NO_MUTATIONS_PROBABILITY}
MUTATION_PROBABILITY = {MUTATION_PROBABILITY}
MAX_DISPLAY_RESULTS = {MAX_DISPLAY_RESULTS}
MAX_TABLE_ROW_COUNT = {MAX_TABLE_ROW_COUNT}
N_JOBS = {N_JOBS}
FORCE_ADD_NEW_JOBS_EVEN_IF_THERE_ARE_OLD_JOBS_REMAINING = {FORCE_ADD_NEW_JOBS_EVEN_IF_THERE_ARE_OLD_JOBS_REMAINING}
''')


@MainWrapper(skip_memory_limit=False)
def main():
    if CLEAR_OLD_JOBS and not RESULTS_ONLY:
        logging.info('Removing old jobs...')
        SerializableJob.clear_jobs(JOBS_SUBDIR)

    # parameter_set = FNetParametersAsUSedForSPIE()
    # parameter_set = HyperparametersOptimizedForTraumaDataset()
    # parameter_set = FNetFor45Epochs()
    # parameter_set = FNetDefaultHyperparameterSet()
    # parameter_set = FNetWithBatchNorm()
    # parameter_set = SwinConfig()
    # parameter_set = PerceiverConfigFromScratch()
    # parameter_set = PerceiverConfigPretrained()
    # parameter_set = FNetFor3Epochs()
    # parameter_set = SimplePytorchModelConfig()
    # parameter_set = FNetConfigForSendingToFaraz20231123()
    # parameter_set = FNetHyperparameterSetForScoutScans()
    parameter_set = KerasZooModelConfig()
    # parameter_set = FNet2DForScoutScans()

    parameter_set.model_type()  # make sure the model definition is available for import

    def print_results_table(parameter_set: HyperparameterSet):
        parameter_search.finish_experiments(EXPERIMENT_NAME,
                                            params=parameter_set.hyper_parameters(),
                                            metric_names=parameter_set.metrics_names(),
                                            # metric_names=['val_ifo_wacc', 'val_ddxgs_wcacc', 'early_stopped_after', 'num_params',
                                            #               'early_stopped_during_last_2_epochs', 'val_ifo_ce'],
                                            optimize='val_' + parameter_set.optimization_criterion(),
                                            larger_result_is_better=LARGER_RESULT_IS_BETTER,
                                            print_results_table=True,
                                            max_table_row_count=MAX_TABLE_ROW_COUNT,
                                            max_display_results=MAX_DISPLAY_RESULTS,
                                            filter_results_table=f'epochs = {parameter_set.num_epochs()} '
                                                                 f'AND swa_start_batch IS NOT NULL '
                                                                 f'AND tasks = "{TASKS.serialize()}"',
                                            # f'AND early_stopped_after IS NOT NULL',
                                            single_cell_values=True,
                                            ignore_constant_columns=True,
                                            # plot_metrics_by_metrics=True,
                                            )

    def schedule_for_all_folds(num_experiments_per_fold_,
                               param_changes_: UsingDifferentParameters = None,
                               using_specific_data_source: FNetDataSource = None,
                               parameter_searching=False):
        if using_specific_data_source is None:
            using_specific_data_source = DataSourceCombination(DATA_SOURCES)
        folds: List[int] = using_specific_data_source.folds_for_cv_setting()
        for fold_idx in folds:
            if isinstance(num_experiments_per_fold_, dict):
                num_experiments_this_fold = num_experiments_per_fold_[fold_idx]
            else:
                num_experiments_this_fold = num_experiments_per_fold_
            for _ in range(num_experiments_this_fold):
                schedule_for_single_fold(fold_idx, param_changes_, using_specific_data_source, parameter_searching=parameter_searching)

    def schedule_for_single_fold(fold_idx: int,
                                 param_changes_: UsingDifferentParameters = None,
                                 using_specific_data_source: DataSource = None,
                                 parameter_searching=False):
        from joblib import delayed
        if param_changes_ is None:
            param_changes_ = UsingSameParameters()
        with param_changes_:
            parameter_set_after_changes = copy.deepcopy(parameter_set)
        evaluate = MainWrapper(beep=False, re_raise_unkown_exceptions=True)(parameter_search.evaluate_with_initial_params)
        assert parameter_set_after_changes.model_type() in TrainableImageProcessingTool.SUBCLASSES_BY_NAME
        job = (
            delayed
            (evaluate)
            (EXPERIMENT_NAME,
             params=parameter_set_after_changes.hyper_parameters(),
             evaluate=functools.partial(
                 FNetParameterEvaluator.default_evaluator(using_specific_data_source,
                                                          parameter_searching=parameter_searching).evaluate_parameters_on_specific_fold,
                 fold_id=fold_idx,
                 beep=True,
             ),
             metric_names=parameter_set_after_changes.metrics_names(),
             optimize='val_' + parameter_set_after_changes.optimization_criterion(),
             num_experiments=1,
             larger_result_is_better=LARGER_RESULT_IS_BETTER,
             experiment_count='tries')
        )
        SerializableJob(*job, subdir=JOBS_SUBDIR).serialize()

    def schedule_hyperparameter_search():
        from joblib import delayed
        evaluate = MainWrapper(beep=False, re_raise_unkown_exceptions=True)(parameter_search.random_parameter_search)
        job = (
            delayed
            (evaluate)
            (EXPERIMENT_NAME,
             params=parameter_set.hyper_parameters(),
             evaluate=FNetParameterEvaluator.default_evaluator(parameter_searching=True).evaluate_parameters_on_all_folds,
             mutation_probability=MUTATION_PROBABILITY,
             no_mutations_probability=NO_MUTATIONS_PROBABILITY,
             metric_names=parameter_set.metrics_names(),
             optimize='val_' + parameter_set.optimization_criterion(),
             larger_result_is_better=LARGER_RESULT_IS_BETTER,
             ignore_configuration_condition=f'epochs != {parameter_set.num_epochs()} OR swa_start_batch IS NULL ',
             runs_per_configuration=20,
             initial_experiments=1,
             # max_num_experiments=1
             )
        )
        SerializableJob(*job, subdir=JOBS_SUBDIR, run_repeatedly=True).serialize()

    def schedule_hyperparameter_searches():
        for _ in range(N_JOBS):
            schedule_hyperparameter_search()

    try:
        if len(SerializableJob.remaining_jobs(JOBS_SUBDIR)) == 0 or FORCE_ADD_NEW_JOBS_EVEN_IF_THERE_ARE_OLD_JOBS_REMAINING:
            num_experiments_per_fold = 5
            logging.info(f'num_experiments_per_fold {num_experiments_per_fold}')

            # for name, tasks in param_changess.items():
            #     if isinstance(tasks, VertebraTask):
            #         tasks = VertebraTasks([tasks])
            #     param_changes = UsingDifferentParameterValue(
            #         default_param=parameter_set.tasks_parameter,
            #         value=tasks.serialize(),
            #         value_name=name,
            #         verbose=False
            #     )
            #     schedule_for_all_folds(num_experiments_per_fold_=num_experiments_per_fold,
            #                            param_changes_=param_changes,
            #                            using_specific_data_source=data_source_changes[name])

            # base_param = parameter_set.pixel_scaling_parameter

            param_changes = [
                MultipleDifferentParameterValues([
                    UsingDifferentParameterValue(parameter_set.backbone_name, value=n, verbose=False, ),
                    UsingDifferentParameterValue(parameter_set.pretrained, value=p, verbose=False, ),
                ],
                    name=f'_{n}{"_p" if p else ""}',
                    model_subdir_parameter=parameter_set.model_subdir_parameter,
                    verbose=False)
                for n in parameter_set.backbone_name.possible_values # [:2]
                for p in parameter_set.pretrained.possible_values
            ]
            for c in ProgressBar(param_changes):
                schedule_for_all_folds(num_experiments_per_fold_=num_experiments_per_fold,
                                       param_changes_=c)
            # schedule_for_all_folds(num_experiments_per_fold_=num_experiments_per_fold)
            # for f_idx in [1, 2, 3,
            #               1, 2, 3,
            #               1, 2, 3,
            #               1, 2, 3, ]:
            #     schedule_for_single_fold(f_idx)
            # scout_data_sources = {
            #     'trained_on_mros': MrOsScoutDataSource(manual_coordinates_only=True),
            #     'trained_on_db_mros': DataSourceCombination([MrOsScoutDataSource(manual_coordinates_only=True), DiagnostikBilanzScoutDataSource()]),
            #     'trained_on_db': DiagnostikBilanzScoutDataSource(),
            # }
            # scout_configs = {
            #     '2d': FNet2DForScoutScans,
            #     'pseudo_3d': FNetHyperparameterSetForScoutScans,
            # }

            # for hp_name, hyperparameter_set in scout_configs.items():
            #     hyperparameter_set_with_db_tasks = hyperparameter_set()
            #     hyperparameter_set_with_db_tasks.tasks_parameter.initial_value = VertebraTasks.tasks_for_which_diagnostik_bilanz_has_annotations().serialize()
            #     schedule_for_all_folds(num_experiments_per_fold_=num_experiments_per_fold,
            #                            param_changes_=UsingAnotherHyperparameterSet(
            #                                original_hyperparameter_set=parameter_set,
            #                                new_hyperparameter_set=hyperparameter_set_with_db_tasks,
            #                                name=hp_name + '_db_extra_tasks',
            #                                model_subdir_parameter=parameter_set.model_subdir_parameter
            #                            ))
            # schedule_hyperparameter_searches()

        if not RESULTS_ONLY:
            remaining_jobs = len(SerializableJob.remaining_jobs(subdir=JOBS_SUBDIR))
            print(f'Starting Parallel backend with {N_JOBS} workers, {remaining_jobs} jobs')
            joblib.externals.loky.set_loky_pickler('dill')
            before_dispatch = BeforeDispatch().before_dispatch
            with lib.scheduling.MyParallel(n_jobs=N_JOBS, prefer='processes', verbose=20, batch_size=1, max_nbytes=None,
                                           before_dispatch=before_dispatch, reuse=False) as parallel:
                parallel(j.delayed() for j in SerializableJob.remaining_jobs_iterator(subdir=JOBS_SUBDIR,
                                                                                      shuffle=N_JOBS > 1))

        print_results_table(parameter_set)

    except lib.util.DontSaveResultsError:
        print('WARNING: No results saved this run!')

    util.beep(2000, 500)


if __name__ == '__main__':
    main()
