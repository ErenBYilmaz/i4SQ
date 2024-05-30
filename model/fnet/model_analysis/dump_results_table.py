import codecs
import os.path
from contextlib import redirect_stdout
from typing import List, Dict

from lib import parameter_search
from model.fnet.const import LARGER_RESULT_IS_BETTER
from model.fnet.evaluate import FNetParameterEvaluator
from model.fnet.hyperparameter_set import FNetDefaultHyperparameterSet
from model.fnet.model_analysis.analyze_trained_models import SingleModelEvaluationResult, ModelAnalyzer


class DumpResultsTable(ModelAnalyzer):
    def analyze_batch(self, batch, y_preds, names):
        raise NotImplementedError('Not needed for this analyzer. Do not call.')

    def __init__(self, evaluator: FNetParameterEvaluator, skip_existing=None):
        super().__init__(evaluator=evaluator, analysis_needs_xs=False, skip_existing=skip_existing, model_level_caching=False,
                         binary_threshold_or_method=0.5)

    def to_subdir(self):
        return 'results_table'

    def to_dir(self, short=False, subdir: str = None) -> str:
        if subdir is None:
            if hasattr(self, 'model_dir'):
                subdir = os.path.relpath(self.model_dir, "models")
            else:
                subdir = ''
        return os.path.join(
            '' if short else self.BASE_IMG_DIR,
            '' if short else subdir,
            self.coordinate_csv_name if self.coordinate_csv_name is not None else '',
        )

    def analyze_single_model(self, model_path: str, ) -> SingleModelEvaluationResult:
        raise NotImplementedError('Not needed for this analyzer. Do not call.')

    def analyze_subdirectories(self, model_dir) -> Dict[str, List[List[SingleModelEvaluationResult]]]:
        experiment_name = os.path.basename(model_dir)
        to_file = os.path.join(self.to_dir(subdir=experiment_name), f'results_table.log')
        self.serializer().create_directory_if_not_exists(os.path.dirname(to_file))

        if self.skip_existing and self.serializer().isfile(to_file):
            results = []
        else:
            parameter_set = FNetDefaultHyperparameterSet()
            with codecs.open(to_file, 'w', "utf-8") as f:
                with redirect_stdout(f):
                    parameter_search.finish_experiments(experiment_name,
                                                        params=parameter_set.hyper_parameters(),
                                                        metric_names=parameter_set.metrics_names(),
                                                        optimize='val_' + parameter_set.optimization_criterion(),
                                                        larger_result_is_better=LARGER_RESULT_IS_BETTER,
                                                        print_results_table=True,
                                                        max_table_row_count=2048,
                                                        max_display_results=0,
                                                        filter_results_table=f'epochs = (SELECT MAX(EPOCHS) FROM {experiment_name}) '
                                                                             f'AND swa_start_batch IS NOT NULL '
                                                                             f'AND early_stopped_after IS NOT NULL',
                                                        single_cell_values=True,
                                                        ignore_constant_columns=True,
                                                        show_progress_bar=False)
            print(f'Wrote {os.path.abspath(to_file)}')
            results = []
        return self.after_multiple_directories(results)
