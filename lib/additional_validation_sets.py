import time
from builtins import DeprecationWarning
from math import nan, inf
from typing import Dict, List, Tuple, Union, Optional, TYPE_CHECKING

from lib.callbacks import AnyCallback
from lib.image_processing_tool import TrainableImageProcessingTool

if TYPE_CHECKING:
    from wandb.apis.public import Run
else:
    Run = None

from lib.clone_compiled_model import clone_model_or_tool

MetricName = str


class AdditionalValidationSets(AnyCallback):
    def __init__(self, validation_sets, verbose=1, batch_size=None, record_original_history=True,
                 record_predictions=False,
                 keep_best_model_by_metric: Optional[MetricName] = None,
                 larger_result_is_better: Optional[bool] = None,
                 evaluate_on_best_model_by_metric: bool = False,
                 keep_history=False,
                 log_using_wandb_run: Run = None):
        """
        :param validation_sets:
        a list of
        2-tuples ((validation_generator, validation_steps), validation_set_name) or
        3-tuples (validation_data, validation_targets, validation_set_name) or
        4-tuples (validation_data, validation_targets, sample_weights, validation_set_name)
        :param verbose:
        verbosity mode, 1 or 0
        :param batch_size:
        batch size to be used when evaluating on the additional datasets
        :param keep_best_model_by_metric:
        some kind of early stopping: you specify a metric and the best model
        according to this metric will be available after training in the .best_model attribute
        note that some additional work may be required if you are using custom layers in your model,
        as this feature requires cloning the model.
        :param evaluate_on_best_model_by_metric:
        if keep_best_model_by_metric is True then this parameter determines if evaluation
        should be done on the "best" found model or the most recent one
        :param keep_history:
        whether or not to keep the history for the next training run
        """
        super(AdditionalValidationSets, self).__init__()
        if larger_result_is_better is None and keep_best_model_by_metric is not None:
            raise ValueError('If you want to keep the best model you need to specify if you '
                             'want to keep the model with the largest or smallest metric '
                             '(parameter larger_result_is_better).')
        self.keep_history = keep_history
        self.evaluate_on_best_model_by_metric = evaluate_on_best_model_by_metric
        self.keep_best_model_by_metric = keep_best_model_by_metric
        self.larger_result_is_better = larger_result_is_better
        self.record_predictions = record_predictions
        if record_predictions:
            raise DeprecationWarning('record_predictions is not supported anymore')
        self.validation_sets = validation_sets
        for validation_set in self.validation_sets:
            if len(validation_set) not in [2, 3, 4]:
                raise ValueError()
        self.epoch = []
        self.metrics_to_be_logged_to_wandb: Dict[str, float] = {}
        self.history: Dict[str, List[Union[float, Dict]]] = {}
        self.verbose = verbose
        self.batch_size = batch_size
        self.record_original_history = record_original_history
        self.best_model: Optional[TrainableImageProcessingTool] = None
        self.best_metric = self.worst_possible_metric()
        self.t = time.perf_counter()
        self.log_using_wandb_run = log_using_wandb_run
        self._supports_tf_logs = not record_original_history
        if record_original_history:
            print('WARNING: Possible performance loss while recording history.')

    def worst_possible_metric(self):
        if self.larger_result_is_better is None:
            return None
        if self.larger_result_is_better:
            return -inf
        else:
            return inf

    def on_train_begin(self, logs=None):
        if not self.keep_history:
            self.epoch = []
            self.history = {}
        if self.keep_best_model_by_metric is not None:
            self.best_metric = self.worst_possible_metric()
            if self.model_to_evaluate() is not None:
                self.best_model = clone_model_or_tool(self.model)
        else:
            self.best_model = self.model

    def metric_names(self):
        return self.model.metrics_names

    def validation_set_by_name(self, name: str):
        for validation_set in self.validation_sets:
            if validation_set[-1] == name:
                return validation_set

    def on_epoch_end(self, epoch, logs=None):
        self.epoch.append(epoch)

        if self.record_original_history:
            # record the same values as History() as well
            logs = logs or {}
            for k, v in logs.items():
                self.record_metric_value(k, v)

        if len(self.validation_sets) == 0:
            return

        # evaluate on the additional validation sets
        model: TrainableImageProcessingTool = self.model_to_evaluate()
        stop_training_before = self.model.stop_training

        if self.keep_best_model_by_metric:
            if all(not self.keep_best_model_by_metric.endswith(metric)
                   for metric in self.metric_names()):
                raise ValueError(f'Unknown metric name: {self.keep_best_model_by_metric}. Available: {self.metric_names()}')

        for validation_set in self.validation_sets:
            assert len(validation_set) == 2
            (validation_generator, validation_steps), validation_set_name = validation_set
            if model is not None:
                assert validation_generator is not None
                results = model.evaluate(validation_generator,
                                         verbose=0,
                                         steps=validation_steps)
            else:
                results = [nan for _ in self.metric_names()]

            for i, result in enumerate(results):
                value_name = self.prefix() + validation_set_name + '_' + self.metric_names()[i]
                self.record_metric_value(value_name, result)

        if self.log_to_wandb():
            self.wandb_commit(epoch)

        if self.keep_best_model_by_metric and model is not None:
            if self.best_model is None:
                self.best_model = clone_model_or_tool(model)
            if self.keep_best_model_by_metric in self.history:
                last_metric = self.history[self.keep_best_model_by_metric][-1]
            elif self.prefix() + self.keep_best_model_by_metric in self.history:
                last_metric = self.history[self.prefix() + self.keep_best_model_by_metric][-1]
            else:
                raise ValueError(f'Unknown metric name: {self.keep_best_model_by_metric}. '
                                 f'Available are {set(self.history).union(k[len(self.prefix()):] for k in self.history)}')

            if last_metric is not None:
                if self.larger_result_is_better and last_metric > self.best_metric:
                    self.best_metric = last_metric
                    self.best_model.set_weights(model.get_weights())
                elif not self.larger_result_is_better and last_metric < self.best_metric:
                    self.best_metric = last_metric
                    self.best_model.set_weights(model.get_weights())

                self.best_metric = self.worst_possible_metric()
            else:
                self.best_model = self.model

        if self.verbose and model is not None:
            metric_strings = [f'{round(time.perf_counter() - self.t)}s']
            metric_strings += [f'{metric}: {values[-1]:#.4g}'
                               for metric, values in self.history.items()
                               if metric not in logs and '_predictions' not in metric]
            print(' - '.join(metric_strings))
            # headers = []
            # table = [[]]
            # for metric, values in self.history.items():
            #     if metric in logs or '_predictions' in metric:
            #         continue
            #     headers.append(metric)
            #     table[0].append(values[-1])
            # print(lib.util.my_tabulate(table, headers=headers, tablefmt='plain'))

        self.t = time.perf_counter()
        if stop_training_before:
            self.model.stop_training = stop_training_before

    def record_metric_value(self, value_name, value):
        self.history.setdefault(value_name, []).append(value)
        if self.log_to_wandb():
            assert value_name not in self.metrics_to_be_logged_to_wandb
            self.metrics_to_be_logged_to_wandb[value_name] = value

    def log_to_wandb(self):
        return self.log_using_wandb_run is not None

    def wandb_commit(self, epoch):
        assert self.log_to_wandb()
        self.log_using_wandb_run.log({**self.metrics_to_be_logged_to_wandb, 'epoch': epoch},
                                     step=epoch)
        self.metrics_to_be_logged_to_wandb = {}

    def model_to_evaluate(self) -> Optional[TrainableImageProcessingTool]:
        """
        To be overridden by subclasses, i was once using this to implement
        Stochastic Weight Averaging (https://arxiv.org/abs/1803.05407)
        which builds a separate model in callbacks, that is then also evaluated.
        If `None` is returned, no evaluation is done (in that epoch).
        """
        return self.model

    # noinspection PyMethodMayBeStatic
    def prefix(self):
        """
        To be overridden by subclasses, the value returned here will be prepended to the keys in the list of metrics
        """
        return ''

    def results(self):
        """
        I actually don't remember what this method was used for, looks like it returns the results of the last epoch only
        :return: list of pairs (set_name:str, last_results:float)
        """
        if self.history == {}:
            return None
        else:
            results: List[Tuple[str, float]] = [(key, self.history[key][len(self.history[key]) - 1]) for key in
                                                self.history]
            rs: Dict[str, float] = {key: value for (key, value) in results}
            return rs
