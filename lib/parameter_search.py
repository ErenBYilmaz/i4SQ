"""
made by Eren Yilmaz
"""
import functools
import itertools
import math
import re
import sys
from copy import deepcopy

from lib.tuned_cache import TunedMemory
from lib import util
from lib.my_logger import logging
from lib.progress_bar import ProgressBar
import pickle

from matplotlib.axes import Axes
import os
import random
import sqlite3
from datetime import datetime
from math import inf, nan, ceil, sqrt
from timeit import default_timer
from typing import Dict, Any, List, Callable, Optional, Tuple, Iterable, Union

import numpy
import pandas
import scipy.stats

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates

from lib.util import my_tabulate, round_to_digits, print_progress_bar, heatmap_from_points, LogicError, shorten_name

score_cache = TunedMemory(location='.cache/scores', verbose=0)
sns.set(style="whitegrid", font_scale=1.5)

# set up db for results
connection: sqlite3.Connection = sqlite3.connect('random_search_results.db')
connection.cursor().execute('PRAGMA foreign_keys = 1')
connection.cursor().execute('PRAGMA journal_mode = WAL')
connection.cursor().execute('PRAGMA synchronous = NORMAL')

Parameters = Dict[str, Any]
MetricValue = float
Metrics = Dict[str, MetricValue]
History = List[MetricValue]

suppress_intermediate_beeps = False


class Prediction:
    def __init__(self, dataset: str, y_true, y_pred, name: str):
        self.y_pred = y_pred
        self.y_true = y_true
        self.name = name
        if not isinstance(name, str):
            self.name = str(name)
        else:
            self.name = name
        if not isinstance(dataset, str):
            raise TypeError
        self.dataset = dataset

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__class__.__name__ + repr({k: v for k, v in self.__dict__.items() if k != 'predictions'})


class EvaluationResult:
    def __init__(self,
                 results: Dict[str, MetricValue],
                 parameters: Parameters = None,
                 predictions: Optional[List[Prediction]] = None):
        self.predictions = predictions
        if self.predictions is not None:
            self.predictions: Optional[List[Prediction]] = self.predictions.copy()
        else:
            self.predictions: Optional[List[Prediction]] = []
        if parameters is None:
            self.parameters = parameters
        else:
            self.parameters = parameters.copy()
        self.results: Dict[str, MetricValue] = results.copy()

    def __iter__(self):
        yield self

    def __eq__(self, other):
        return (isinstance(other, EvaluationResult)
                and self.parameters == other.parameters
                and self.predictions == other.predictions
                and self.results == other.results)

    def __str__(self):
        return '{0}{1}'.format(self.__class__.__name__,
                               {k: getattr(self, k) for k in ['results', 'parameters', 'predictions']})

    def __repr__(self):
        return self.__class__.__name__ + repr(self.__dict__)


assert list(EvaluationResult({}, {})) == list([EvaluationResult({}, {})])

EvaluationFunction = Callable[[Parameters], Union[List[EvaluationResult], EvaluationResult, List[float], float]]


class Parameter:
    def __init__(self, name: str, initial_value, larger_value, smaller_value, first_try_increase=False):
        self.name = name
        self.initial_value = initial_value
        self.larger_value = larger_value
        self.smaller_value = smaller_value
        self.first_try_increase = first_try_increase

    def __repr__(self):
        return self.__class__.__name__ + repr(self.__dict__)

    def copy(self, new_name=None):
        result: Parameter = deepcopy(self)
        if new_name is not None:
            result.name = new_name
        return result


class BoundedParameter(Parameter):
    def __init__(self, name, initial_value, larger_value, smaller_value, minimum=-inf, maximum=inf,
                 first_try_increase=False):
        self.minimum = minimum
        self.maximum = maximum

        super().__init__(name,
                         initial_value,
                         lambda x: self._bounded(larger_value(x)),
                         lambda x: self._bounded(smaller_value(x)),
                         first_try_increase=first_try_increase)

        if self.initial_value < self.minimum:
            raise ValueError('Initial value is lower than minimum value.')
        if self.initial_value > self.maximum:
            raise ValueError('Initial value is larger than maximum value.')

    def _bounded(self, y):
        y = max(self.minimum, y)
        y = min(self.maximum, y)
        return y


class ListParameter(Parameter):
    def __init__(self, name, initial_value, possible_values: List, first_try_increase=False, circle=False):
        self.possible_values = possible_values.copy()
        if initial_value not in self.possible_values:
            raise ValueError()
        if len(set(self.possible_values)) != len(self.possible_values):
            print('WARNING: It seems that there are duplicates in the list of possible values for {0}'.format(name))
        length = len(self.possible_values)
        if circle:
            smaller = lambda x: self.possible_values[(self.current_value_idx(x) + 1) % length]
            larger = lambda x: self.possible_values[(self.current_value_idx(x) - 1) % length]
        else:
            smaller = lambda x: self.possible_values[min(self.current_value_idx(x) + 1, length - 1)]
            larger = lambda x: self.possible_values[max(self.current_value_idx(x) - 1, 0)]

        super().__init__(name,
                         initial_value,
                         smaller,
                         larger,
                         first_try_increase=first_try_increase)

    def current_value_idx(self, x):
        try:
            return self.possible_values.index(x)
        except ValueError:
            return 0


class ConstantParameter(Parameter):
    def __init__(self, name, value):
        super().__init__(name,
                         initial_value=value,
                         larger_value=lambda x: self.initial_value,
                         smaller_value=lambda x: self.initial_value)

    @staticmethod
    def from_param(param: Parameter):
        return ConstantParameter(param.name, param.initial_value)


class BinaryParameter(ListParameter):
    def __init__(self, name, value1, value2):
        super().__init__(name,
                         value1,
                         possible_values=[value1, value2])


class BooleanParameter(BinaryParameter):
    def __init__(self, name, initial_value: bool):
        super().__init__(name,
                         bool(initial_value),
                         not bool(initial_value))


class TernaryParameter(ListParameter):
    def __init__(self, name, value1, value2, value3):
        super().__init__(name,
                         value1,
                         possible_values=[value1, value2, value3],
                         circle=True)


class ExponentialParameter(BoundedParameter):
    def __init__(self, name, initial_value, base, minimum=-inf, maximum=inf, first_try_increase=False):
        super().__init__(name,
                         initial_value,
                         lambda x: float(x * base),
                         lambda x: float(x / base),
                         minimum,
                         maximum,
                         first_try_increase=first_try_increase)
        self.plot_scale = 'log'


class ExponentialIntegerParameter(BoundedParameter):
    def __init__(self, name, initial_value, base, minimum=-inf, maximum=inf, first_try_increase=False):
        if minimum != -inf:
            minimum = round(minimum)

        if maximum != inf:
            maximum = round(maximum)

        super().__init__(name,
                         round(initial_value),
                         lambda x: round(x * base),
                         lambda x: round(x / base),
                         minimum,
                         maximum,
                         first_try_increase=first_try_increase)
        self.plot_scale = 'log'


class LinearParameter(BoundedParameter):
    def __init__(self, name, initial_value, summand, minimum=-inf, maximum=inf, first_try_increase=False):
        super().__init__(name,
                         initial_value,
                         lambda x: float(x + summand),
                         lambda x: float(x - summand),
                         minimum,
                         maximum,
                         first_try_increase=first_try_increase)


class LinearIntegerParameter(BoundedParameter):
    def __init__(self, name, initial_value, summand, minimum=-inf, maximum=inf, first_try_increase=False):
        super().__init__(name,
                         initial_value,
                         lambda x: x + summand,
                         lambda x: x - summand,
                         minimum,
                         maximum,
                         first_try_increase=first_try_increase)


class InvalidParametersError(Exception):
    def __init__(self, parameters=None):
        self.parameters = parameters


class BadParametersError(InvalidParametersError):
    pass


class InvalidReturnError(Exception):
    pass


class EmptyTableError(Exception):
    pass


EXAMPLE_PARAMS = [
    ExponentialParameter('learn_rate', 0.001, 10),
    ExponentialIntegerParameter('hidden_layer_size', 512, 2, minimum=1),
    LinearIntegerParameter('hidden_layer_count', 3, 1, minimum=0),
    ExponentialIntegerParameter('epochs', 100, 5, minimum=1),
    LinearParameter('dropout_rate', 0.5, 0.2, minimum=0, maximum=1),
]


def mean_confidence_interval_size(data, confidence=0.95, force_v: Optional[int] = None,
                                  force_sem: Optional[float] = None):
    if len(data) == 0:
        return nan
    if force_sem is None:
        if len(data) == 1:
            return inf
        sem = scipy.stats.sem(data)
    else:
        sem = force_sem
    if sem == 0:
        return 0
    if force_v is None:
        v = len(data) - 1
    else:
        v = force_v
    return numpy.mean(data) - scipy.stats.t.interval(confidence,
                                                     df=v,
                                                     loc=numpy.mean(data),
                                                     scale=sem)[0]


def try_parameters(experiment_name: str,
                   evaluate: EvaluationFunction,
                   params: Dict[str, any],
                   optimize: Optional[str] = None,
                   larger_result_is_better: bool = None, ):
    print('running experiment...')
    params = params.copy()

    if larger_result_is_better is None and optimize is not None:
        raise NotImplementedError(
            'Don\'t know how to optimize {0}. Did you specify `larger_result_is_better`?'.format(optimize))
    assert larger_result_is_better is not None or optimize is None

    worst_score = -inf if larger_result_is_better else inf

    cursor = connection.cursor()
    start = default_timer()
    try:
        result = evaluate(params)
        if not isinstance(result, Iterable):
            result = [result]
        evaluation_results: List[EvaluationResult] = list(result)
    except InvalidParametersError as e:
        frame = sys.exc_info()[2]
        while frame.tb_next is not None:
            frame = frame.tb_next
        logging.error(f'InvalidParameters according to line {frame.tb_lineno} of {frame.tb_frame.f_code.co_filename}')
        del frame
        if optimize is not None:
            bad_results: Dict[str, float] = {
                optimize: worst_score  # default for metrics is NULL
            }
        else:
            bad_results = {}
        if e.parameters is None:
            evaluation_results = [EvaluationResult(
                parameters=params,
                results=bad_results
            )]
        else:
            evaluation_results = [EvaluationResult(
                parameters=e.parameters,
                results=bad_results
            )]
    finally:
        duration = default_timer() - start

    for idx in range(len(evaluation_results)):
        if isinstance(evaluation_results[idx], float):
            evaluation_results[idx] = EvaluationResult(parameters=params,
                                                       results={optimize: evaluation_results[idx]})

    p_count = 0
    for evaluation_result in evaluation_results:
        if evaluation_result.parameters is None:
            evaluation_result.parameters = params
        metric_names = sorted(evaluation_result.results.keys())
        param_names = list(sorted(evaluation_result.parameters.keys()))
        for metric_name in metric_names:
            add_metric_column(experiment_name, metric_name, verbose=1)
        for param_name in param_names:
            add_parameter_column(experiment_name, param_name, evaluation_result.parameters[param_name], verbose=1)

        if not set(param_names).isdisjoint(metric_names):
            raise RuntimeError('Metrics and parameter names should be disjoint')

        if optimize is not None and numpy.isnan(evaluation_result.results[optimize]):
            evaluation_result.results[optimize] = worst_score

        metric_values = [evaluation_result.results[metric_name] for metric_name in metric_names]
        param_names_comma_separated = ','.join('"' + param_name + '"' for param_name in param_names)
        metric_names_comma_separated = ','.join('"' + metric_name + '"' for metric_name in metric_names)
        insert_question_marks = ','.join('?' for _ in range(len(param_names) + len(metric_names)))

        cursor.execute('''
                INSERT INTO {0} ({1}) VALUES ({2})
                '''.format(experiment_name,
                           param_names_comma_separated + ',' + metric_names_comma_separated,
                           insert_question_marks), (*[evaluation_result.parameters[name] for name in param_names],
                                                    *metric_values))
        result_id = cursor.lastrowid
        assert cursor.execute(f'SELECT COUNT(*) FROM {experiment_name}_predictions WHERE result_id = ? LIMIT 1',
                              (result_id,)).fetchone()[0] == 0
        p_count += len(evaluation_result.predictions)
        dataset_names = [(prediction.dataset, prediction.name) for prediction in evaluation_result.predictions]
        if len(set(dataset_names)) != len(dataset_names):
            print('\n'.join(sorted(dsn
                                   for idx, dsn in dataset_names
                                   if dsn in dataset_names[idx:])))
            raise InvalidReturnError(
                'Every combination of name and dataset in a single evaluation result must be unique.'
                'There should be a list of duplicates printed above where the number of occurrences'
                'of an element in the list is the actual number of occurrences minus 1 '
                '(so only duplicates are listed).')
        # noinspection SqlResolve
        cursor.executemany('''
            INSERT INTO {0}_predictions (dataset, y_true, y_pred, result_id, name) 
            VALUES (?, ?, ?, ?, ?)
            '''.format(experiment_name), [(prediction.dataset,
                                           pickle.dumps(prediction.y_true),
                                           pickle.dumps(prediction.y_pred),
                                           result_id,
                                           prediction.name) for prediction in evaluation_result.predictions])

    connection.commit()
    print('saved', len(evaluation_results), 'results and', p_count, 'predictions to db')
    if not suppress_intermediate_beeps:
        util.beep(1000, 500)

    if optimize is not None:
        scores = [r.results[optimize] for r in evaluation_results]
        if larger_result_is_better:
            best_score = max(scores)
        else:
            best_score = min(scores)
        print('  finished in', duration, 'seconds, best loss/score:', best_score)

    for r in evaluation_results:
        if list(sorted(r.results.keys())) != list(sorted(metric_names)):
            raise InvalidReturnError("""
            Wrong metric names were returned by `evaluate`:
            Expected metric_names={0}
            but was {1}.
            The result was saved to database anyways, possibly with missing values.
            """.format(list(sorted(metric_names)),
                       list(sorted(r.results.keys()))))

    return evaluation_results


def config_dict_from_param_list(params: List[Parameter]):
    return {
        p.name: p.initial_value
        for p in params
    }


def evaluate_with_initial_params(experiment_name: str,
                                 params: List[Parameter],
                                 evaluate: EvaluationFunction,
                                 optimize: str,
                                 larger_result_is_better: bool,
                                 metric_names=None,
                                 num_experiments=1,
                                 experiment_count='db_tries_initial'):
    random_parameter_search(experiment_name=experiment_name,
                            params=params,
                            evaluate=evaluate,
                            optimize=optimize,
                            larger_result_is_better=larger_result_is_better,
                            mutation_probability=1.,
                            no_mutations_probability=0.,
                            max_num_experiments=num_experiments,
                            metric_names=metric_names,
                            initial_experiments=num_experiments,
                            experiment_count=experiment_count, )


def random_parameter_search(experiment_name: str,
                            params: List[Parameter],
                            evaluate: EvaluationFunction,
                            optimize: str,
                            larger_result_is_better: bool,
                            mutation_probability: float = None,
                            no_mutations_probability: float = None,
                            allow_multiple_mutations=False,
                            max_num_experiments=inf,
                            metric_names=None,
                            initial_experiments=1,
                            runs_per_configuration=inf,
                            initial_runs=1,
                            ignore_configuration_condition='0',
                            experiment_count='tries', ):
    logging.info(f'experiment name: {experiment_name}')
    logging.info(f'Trying to {("maximize" if larger_result_is_better else "minimize")} {optimize}')
    if metric_names is None:
        metric_names = [optimize]
    if optimize not in metric_names:
        raise ValueError('trying to optimize {0} but only metrics available are {1}'.format(optimize, metric_names))
    params = sorted(params, key=lambda p: p.name)
    validate_parameter_set(params)

    param_names = [param.name for param in params]

    cursor = connection.cursor()
    create_experiment_tables_if_not_exists(experiment_name, params, metric_names)

    def max_tries_reached(ps):
        return len(result_ids_for_parameters(experiment_name, ps)) >= runs_per_configuration

    def min_tries_reached(ps):
        return len(result_ids_for_parameters(experiment_name, ps)) >= initial_runs

    def try_(ps) -> bool:
        tried = False
        if not max_tries_reached(ps):
            try_parameters(experiment_name=experiment_name,
                           evaluate=evaluate,
                           params=ps,
                           optimize=optimize,
                           larger_result_is_better=larger_result_is_better, )
            tried = True
        else:
            logging.info('Skipping because maximum number of tries is already reached.')
            if no_mutations_probability == 1:
                return tried
        while not min_tries_reached(ps):
            logging.info('Repeating because minimum number of tries is not reached.')
            try_parameters(experiment_name=experiment_name,
                           evaluate=evaluate,
                           params=ps,
                           optimize=optimize,
                           larger_result_is_better=larger_result_is_better, )
            tried = True
        return tried

    if mutation_probability is None:
        mutation_probability = 1 / (len(params) + 1)
    if no_mutations_probability is None:
        no_mutations_probability = (1 - 1 / len(params)) / 4
    initial_params = {param.name: param.initial_value for param in params}
    logging.info(f'initial parameters: {str(initial_params)}')

    def skip():
        best_scores, best_mean, best_std, best_conf = get_results_for_params(optimize, experiment_name,
                                                                             best_params, 0.99)
        try_scores, try_mean, try_std, try_conf = get_results_for_params(optimize, experiment_name,
                                                                         try_params, 0.99)
        if larger_result_is_better:
            if best_mean - best_conf > try_mean + try_conf:
                return True
        else:
            if best_mean + best_conf < try_mean - try_conf:
                return True
        return False

    # get best params
    initial_params = {param.name: param.initial_value for param in params}
    try:
        best_params = get_best_params(experiment_name,
                                      larger_result_is_better,
                                      optimize,
                                      param_names,
                                      additional_condition=f'NOT ({ignore_configuration_condition})')
    except EmptyTableError:
        best_params = initial_params
    try_params = best_params.copy()

    def results_for_params(ps):
        return get_results_for_params(
            metric=optimize,
            experiment_name=experiment_name,
            parameters=ps
        )

    if experiment_count == 'tries':
        num_experiments = 0
    elif experiment_count == 'results':
        num_experiments = 0
    elif experiment_count == 'db_total':
        num_experiments = cursor.execute('SELECT COUNT(*) FROM {0} WHERE NOT ({1})'.format(experiment_name,
                                                                                           ignore_configuration_condition)).fetchone()[
            0]
    elif experiment_count == 'db_tries_best':
        num_experiments = len(result_ids_for_parameters(experiment_name, best_params))
    elif experiment_count == 'db_tries_initial':
        num_experiments = len(result_ids_for_parameters(experiment_name, initial_params))
    else:
        raise ValueError('Invalid argument for experiment_count')
    last_best_score = results_for_params(best_params)[1]
    while num_experiments < max_num_experiments:

        if num_experiments < initial_experiments:
            try_params = initial_params.copy()
        else:
            try:
                last_best_params = best_params
                best_params = get_best_params(experiment_name,
                                              larger_result_is_better,
                                              optimize,
                                              param_names,
                                              additional_condition=f'NOT ({ignore_configuration_condition})')
                best_scores, best_score, _, best_conf_size = results_for_params(best_params)
                if last_best_score is not None and best_score is not None:
                    if last_best_params != best_params:
                        if last_best_score < best_score and larger_result_is_better or last_best_score > best_score and not larger_result_is_better:
                            print(' --> Parameters were improved by this change!')
                        if last_best_score > best_score and larger_result_is_better or last_best_score < best_score and not larger_result_is_better:
                            print(' --> Actually other parameters are better...')
                last_best_score = best_score

                # print('currently best parameters:', best_params)
                changed_params = {k: v for k, v in best_params.items() if best_params[k] != initial_params[k]}
                print('currently best parameters (excluding unchanged parameters):', changed_params)
                print('currently best score:', best_score, 'conf.', best_conf_size, 'num.', len(best_scores))
            except EmptyTableError:
                best_params = {param.name: param.initial_value for param in params}
                best_conf_size = inf
            try_params = best_params.copy()
            verbose = 1
            if random.random() > no_mutations_probability:
                modify_params_randomly(mutation_probability, params, try_params, verbose,
                                       allow_multiple_mutations=allow_multiple_mutations)

        if num_experiments < initial_experiments:
            try_params = initial_params.copy()
        else:
            # check if this already has a bad score
            if skip():
                print('skipping because this set of parameters is known to be worse with high probability.')
                print()
                continue

        # print('trying parameters', {k: v for k, v in try_params.items() if try_params[k] != initial_params[k]})

        done_experiment = try_(try_params)
        if not done_experiment and no_mutations_probability == 1:
            return
        if experiment_count == 'tries':
            num_experiments += 1
        elif experiment_count == 'results':
            num_experiments += int(done_experiment)
        elif experiment_count == 'db_total':
            num_experiments = cursor.execute('SELECT COUNT(*) FROM {0}'.format(experiment_name)).fetchone()[0]
        elif experiment_count == 'db_tries_best':
            num_experiments = len(result_ids_for_parameters(experiment_name, best_params))
        elif experiment_count == 'db_tries_initial':
            num_experiments = len(result_ids_for_parameters(experiment_name, initial_params))
        else:
            raise LogicError('It is not possible that this is reached.')


def mutation_test(experiment_name: str,
                  base_params: List[Parameter],
                  mutations: Dict[str, Any],
                  evaluate: EvaluationFunction,
                  optimize: str,
                  larger_result_is_better: bool,
                  metric_names=None,
                  experiments_per_mutation=1, ):
    for param_to_change in mutations:
        for p in base_params:
            if p.name == param_to_change:
                break
        else:
            raise ValueError(param_to_change)
    for param_to_change, value in mutations.items():
        params_this_experiment = deepcopy(base_params)
        for p in params_this_experiment:
            if p.name == param_to_change:
                p.initial_value = value
        evaluate_with_initial_params(experiment_name=experiment_name,
                                     params=params_this_experiment,
                                     evaluate=evaluate,
                                     optimize=optimize,
                                     larger_result_is_better=larger_result_is_better,
                                     metric_names=metric_names,
                                     num_experiments=experiments_per_mutation)


def diamond_parameter_search(experiment_name: str,
                             diamond_size: int,
                             params: List[Parameter],
                             evaluate: EvaluationFunction,
                             optimize: str,
                             larger_result_is_better: bool,
                             runs_per_configuration=inf,
                             initial_runs=1,
                             metric_names=None,
                             filter_results_condition='1'):
    print('experiment name:', experiment_name)
    if metric_names is None:
        metric_names = [optimize]
    if optimize not in metric_names:
        raise ValueError('trying to optimize {0} but only metrics available are {1}'.format(optimize, metric_names))
    print('Optimizing metric', optimize)
    if runs_per_configuration > initial_runs:
        print(
            f'WARNING: You are using initial_runs={initial_runs} and runs_per_configuration={runs_per_configuration}. '
            f'This may lead to unexpected results if you dont know what you are doing.')
    params_in_original_order = params
    params = sorted(params, key=lambda p: p.name)
    validate_parameter_set(params)

    create_experiment_tables_if_not_exists(experiment_name, params, metric_names)

    initial_params = {param.name: param.initial_value for param in params}
    print('initial parameters:', initial_params)

    # get best params
    initial_params = {param.name: param.initial_value for param in params}
    try:
        best_params = get_best_params_and_compare_with_initial(experiment_name, initial_params, larger_result_is_better,
                                                               optimize,
                                                               additional_condition=filter_results_condition)
    except EmptyTableError:
        best_params = initial_params

    def max_tries_reached(ps):
        return len(result_ids_for_parameters(experiment_name, ps)) >= runs_per_configuration

    def min_tries_reached(ps):
        return len(result_ids_for_parameters(experiment_name, ps)) >= initial_runs

    def try_(ps) -> bool:
        tried = False
        if not max_tries_reached(ps):
            try_parameters(experiment_name=experiment_name,
                           evaluate=evaluate,
                           params=ps,
                           optimize=optimize,
                           larger_result_is_better=larger_result_is_better, )
            tried = True
        else:
            print('Skipping because maximum number of tries is already reached.')
        while not min_tries_reached(ps):
            print('Repeating because minimum number of tries is not reached.')
            try_parameters(experiment_name=experiment_name,
                           evaluate=evaluate,
                           params=ps,
                           optimize=optimize,
                           larger_result_is_better=larger_result_is_better, )
            tried = True
        return tried

    last_best_score = results_for_params(optimize, experiment_name, best_params)[1]
    modifications_steps = [
        {'param_name': param.name, 'direction': direction}
        for param in params_in_original_order
        for direction in ([param.larger_value, param.smaller_value] if param.first_try_increase
                          else [param.smaller_value, param.larger_value])
    ]
    while True:  # repeatedly iterate parameters
        restart_scheduled = False
        any_tries_done_this_iteration = False
        for num_modifications in range(diamond_size + 1):  # first try small changes, later larger changes
            modification_sets = itertools.product(*(modifications_steps for _ in range(num_modifications)))
            for modifications in modification_sets:  # which modifications to try this time
                while True:  # repeatedly modify parameters in this direction
                    improvement_found_in_this_iteration = False
                    try_params = best_params.copy()
                    for modification in modifications:
                        try_params[modification['param_name']] = modification['direction'](
                            try_params[modification['param_name']])
                    for param_name, param_value in try_params.items():
                        if best_params[param_name] != param_value:
                            print(f'Setting {param_name} = {param_value} for the next run.')
                    if try_params == best_params:
                        print('Repeating experiment with best found parameters.')

                    if try_(try_params):  # if the experiment was actually conducted
                        any_tries_done_this_iteration = True
                        best_params = get_best_params_and_compare_with_initial(experiment_name, initial_params,
                                                                               larger_result_is_better, optimize,
                                                                               filter_results_condition)

                        last_best_params = best_params
                        best_scores, best_score, _, best_conf_size = results_for_params(optimize, experiment_name,
                                                                                        best_params)

                        changed_params = {k: v for k, v in best_params.items() if best_params[k] != initial_params[k]}
                        print('currently best parameters (excluding unchanged parameters):', changed_params)
                        print('currently best score:', best_score, 'conf.', best_conf_size, 'num.', len(best_scores))
                    else:
                        last_best_params = best_params
                        _, best_score, _, best_conf_size = results_for_params(optimize, experiment_name, best_params)

                    if last_best_score is not None and best_score is not None:
                        if last_best_params != best_params:
                            if last_best_score < best_score and larger_result_is_better or last_best_score > best_score and not larger_result_is_better:
                                print(' --> Parameters were improved by this change!')
                                improvement_found_in_this_iteration = True
                                if num_modifications > 1:
                                    # two or more parameters were modified and this improved the results -> first try to modify them again in the same direction,
                                    # then restart the search from the best found configuration
                                    restart_scheduled = True
                            elif last_best_score > best_score and larger_result_is_better or last_best_score < best_score and not larger_result_is_better:
                                print(' --> Actually other parameters are better...')
                    if not improvement_found_in_this_iteration:
                        break  # stop if no improvement was found in this direction
                if restart_scheduled:
                    break
            if restart_scheduled:
                break
        if restart_scheduled:
            continue
        if not any_tries_done_this_iteration:
            break  # parameter search finished (converged in some sense)


cross_parameter_search = functools.partial(diamond_parameter_search, diamond_size=1)
cross_parameter_search.__name__ = 'cross_parameter_search'


def get_best_params_and_compare_with_initial(experiment_name, initial_params, larger_result_is_better, optimize,
                                             additional_condition='1'):
    best_params = get_best_params(experiment_name, larger_result_is_better, optimize, list(initial_params),
                                  additional_condition=additional_condition)
    changed_params = {k: v for k, v in best_params.items() if best_params[k] != initial_params[k]}
    best_scores, best_score, _, best_conf_size = results_for_params(optimize, experiment_name, best_params)
    print('currently best parameters (excluding unchanged parameters):', changed_params)
    print('currently best score:', best_score, 'conf.', best_conf_size, 'num.', len(best_scores))
    return best_params


def results_for_params(optimize, experiment_name, ps):
    return get_results_for_params(
        metric=optimize,
        experiment_name=experiment_name,
        parameters=ps
    )


def modify_params_randomly(mutation_probability, params, try_params, verbose, allow_multiple_mutations=False):
    for param in params:
        while random.random() < mutation_probability:
            next_value = random.choice([param.smaller_value, param.larger_value])
            old_value = try_params[param.name]
            try:
                try_params[param.name] = round_to_digits(next_value(try_params[param.name]), 4)
            except TypeError:  # when the parameter is not a number
                try_params[param.name] = next_value(try_params[param.name])
            if verbose and try_params[param.name] != old_value:
                print('setting', param.name, '=', try_params[param.name], 'for this run')
            if not allow_multiple_mutations:
                break


def finish_experiments(experiment_name: str,
                       params: List[Parameter],
                       optimize: str,
                       larger_result_is_better: bool,
                       metric_names=None,
                       filter_results_table='1',
                       max_display_results=None,
                       print_results_table=False,
                       max_table_row_count=inf,
                       plot_metrics_by_metrics=False,
                       plot_metric_over_time=False,
                       ignore_constant_columns=False,
                       plot_metrics_by_parameters=False,
                       single_cell_values=True,
                       thin_headers=True,
                       show_progress_bar=True):
    if single_cell_values and (plot_metrics_by_parameters or plot_metrics_by_metrics):
        raise NotImplementedError
    if max_display_results is inf:
        max_display_results = None
    if metric_names is None:
        metric_names = [optimize]
    # get the best parameters
    cursor = connection.cursor()
    params = sorted(params, key=lambda param: param.name)
    param_names = sorted(set(param.name for param in params))
    param_names_comma_separated = ','.join('"' + param_name + '"' for param_name in param_names)

    best_params = get_best_params(experiment_name, larger_result_is_better, optimize, param_names,
                                  additional_condition=filter_results_table, )
    best_score = get_results_for_params(
        metric=optimize,
        experiment_name=experiment_name,
        parameters=best_params
    )
    initial_params = {param.name: param.initial_value for param in params}

    # get a list of all results with mean std and conf
    if print_results_table or plot_metrics_by_parameters or plot_metrics_by_metrics:
        concatenated_metric_names = ','.join(
            'GROUP_CONCAT(IFNULL(\"{0}\", \'nan\'), \'@\') AS "{0}"'.format(metric_name)
            for metric_name in metric_names)
        worst_score = '-1e999999' if larger_result_is_better else '1e999999'
        limit_string = f'LIMIT {max_table_row_count}' if max_table_row_count is not None and max_table_row_count < inf else ''
        # noinspection SqlAggregates
        cursor.execute('''
        SELECT COUNT(*) AS "#results", {1}, {4}
        FROM {0} AS params
        GROUP BY {1}
        HAVING ({5})
        ORDER BY AVG(CASE WHEN params.{3} IS NULL THEN {6} ELSE params.{3} END) {2}
        {7}
        '''.format(experiment_name,
                   param_names_comma_separated,
                   'DESC' if larger_result_is_better else 'ASC',
                   optimize,
                   concatenated_metric_names,
                   filter_results_table,
                   worst_score,
                   limit_string))
        all_results = cursor.fetchall()

        column_description = list(cursor.description)

        for idx, row in enumerate(all_results):
            all_results[idx] = list(row)

        if ignore_constant_columns and len(all_results) > 1:
            constant_columns = [column_idx for column_idx in reversed(range(len(all_results[0])))
                                if len(set(row[column_idx]
                                           for row in all_results)) == 1
                                if column_idx != 0]
            for column_idx in constant_columns:
                for row in all_results:
                    del row[column_idx]
                del column_description[column_idx]

        # prepare results table
        if print_results_table or plot_metrics_by_metrics or plot_metrics_by_parameters:
            iterations = 0
            print('Generating table of parameters')
            for column_index, column in list(enumerate(column_description))[::-1]:  # reverse
                if show_progress_bar:
                    print_progress_bar(iterations, len(metric_names))
                column_name: str = column[0]
                is_metric = column_name in metric_names
                if thin_headers:
                    column_name = re.sub(r'(.{15,}?)([_A-Z])', r'\1\n\2', column_name)
                    column_name = re.sub(r'\n_', r'\n', column_name)
                    # column_name = column_name.replace('_', '\n')
                column_description[column_index] = column
                if is_metric:
                    if max_display_results is None or max_display_results > 0:
                        column_description[column_index] = column_name + ' values'
                    if single_cell_values:
                        column_description.insert(column_index + 1, column_name)
                    else:
                        column_description.insert(column_index + 1, column_name + ' mean')
                        column_description.insert(column_index + 2, column_name + ' std')
                        column_description.insert(column_index + 3, column_name + ' conf')
                    list_row: List
                    for list_row in all_results:
                        string_values: str = list_row[column_index]
                        if string_values is None:
                            metric_values: List[float] = [nan]
                        else:
                            metric_values = list(map(float, string_values.split('@')))
                        assert all(isinstance(v, float) for v in metric_values)
                        list_row[column_index] = [round_to_digits(x, 3) for x in
                                                  (metric_values if max_display_results is None else metric_values[
                                                                                                     :max_display_results])]
                        sigma = numpy.std(metric_values)
                        mu = numpy.mean(metric_values)
                        conf = mean_confidence_interval_size(metric_values)
                        if single_cell_values:
                            list_row.insert(column_index + 1,
                                            f'µ={mu : <11.6g} ' +
                                            (f'σ={sigma : <9.4g} ' if sigma > 0 else ' ' * 10) +
                                            (f'conf={conf : <9.4g}' if conf != inf and conf != 0 else ' ' * 10), )
                        else:
                            list_row.insert(column_index + 1, mu)
                            list_row.insert(column_index + 2, sigma)
                            list_row.insert(column_index + 3, conf)
                    if all(len(list_row[column_index]) == 0 for list_row in all_results):
                        del column_description[column_index]
                        for list_row in all_results:
                            del list_row[column_index]
                    iterations += 1
                else:
                    column_description[column_index] = column_name
                if show_progress_bar:
                    print_progress_bar(iterations, len(metric_names))

        if print_results_table:  # actually print the table
            print()
            table = my_tabulate(all_results,
                                headers=column_description,
                                tablefmt='pipe')

            print(table)
            cursor.execute('''
            SELECT COUNT(*)
            FROM {0}
            '''.format(experiment_name))
            print('Total number of rows, experiments, cells in this table:',
                  (len(all_results), cursor.fetchone()[0], len(all_results) * len(all_results[0])))
            print('Best parameters:', best_params)
            changed_params = {k: v for k, v in best_params.items() if best_params[k] != initial_params[k]}
            print('Best parameters (excluding unchanged parameters):', changed_params)
            print('loss/score for best parameters (mean, std, conf):', best_score[1:])

        if plot_metrics_by_parameters or plot_metrics_by_metrics:
            print('Loading data...')
            if single_cell_values:
                metric_suffix = ''
                data_columns = ['#results'] + param_names + metric_names
            elif max_display_results == 0:
                metric_suffix = '_mean'
                data_columns = ['#results'] + param_names + [x for name in metric_names
                                                             for x in [
                                                                 name + metric_suffix,
                                                                 name + '_std',
                                                                 name + '_conf'
                                                             ]]
            else:
                metric_suffix = '_mean'
                data_columns = ['#results'] + param_names + [x for name in metric_names
                                                             for x in [
                                                                 name + '_values',
                                                                 name + metric_suffix,
                                                                 name + '_std',
                                                                 name + '_conf'
                                                             ]]
            df = pandas.DataFrame.from_records(all_results, columns=data_columns)
            if plot_metrics_by_parameters:
                print('Plotting metrics by parameter...')
                plots = [
                    (param.name,
                     getattr(param, 'plot_scale', None),
                     param.smaller_value if isinstance(param, BoundedParameter) else None,
                     param.larger_value if isinstance(param, BoundedParameter) else None)
                    for param in params
                ]

                iterations = 0
                for metric_name in metric_names:
                    dirname = 'img/results/{0}/{1}/'.format(experiment_name, metric_name)
                    os.makedirs(dirname, exist_ok=True)
                    for plot, x_scale, min_mod, max_mod in plots:
                        print_progress_bar(iterations, len(metric_names) * len(plots))
                        if min_mod is None:
                            min_mod = lambda x: x
                        if max_mod is None:
                            max_mod = lambda x: x
                        if df[plot].nunique() <= 1:
                            iterations += 1
                            continue
                        grid = sns.relplot(x=plot, y=metric_name + metric_suffix, data=df)
                        if x_scale is not None:
                            if x_scale == 'log' and min_mod(df.min(axis=0)[plot]) <= 0:
                                x_min = None
                            else:
                                x_min = min_mod(df.min(axis=0)[plot])
                            grid.set(xscale=x_scale,
                                     xlim=(x_min,
                                           max_mod(df.max(axis=0)[plot]),))
                        plt.savefig(dirname + '{0}.png'.format(plot))
                        plt.clf()
                        plt.close()
                        iterations += 1
                        print_progress_bar(iterations, len(metric_names) * len(plots))

            if plot_metrics_by_metrics:
                print('Plotting metrics by metrics...')
                dirname = 'img/results/{0}/'.format(experiment_name)
                os.makedirs(dirname, exist_ok=True)

                # Generate some plots, metric by metric
                iterations = 0
                print('Plotting metric by metric, grouped')
                for metric_name in metric_names:
                    for metric_2 in metric_names:
                        if metric_name == metric_2:
                            iterations += 1
                            print_progress_bar(iterations, len(metric_names) ** 2)
                            continue
                        print_progress_bar(iterations, len(metric_names) ** 2)
                        sns.relplot(x=metric_name + metric_suffix, y=metric_2 + metric_suffix, data=df)
                        plt.savefig(dirname + '{0}_{1}_grouped.png'.format(metric_name, metric_2))
                        plt.clf()
                        plt.close()
                        heatmap_from_points(x=df[metric_name + metric_suffix], y=df[metric_2 + metric_suffix])
                        plt.xlabel(f'mean {metric_name}')
                        plt.ylabel(f'mean {metric_2}')
                        plt.savefig(dirname + '{0}_{1}_heatmap.png'.format(metric_name, metric_2))
                        plt.clf()
                        plt.close()
                        iterations += 1
                        print_progress_bar(iterations, len(metric_names) ** 2)

                df = pandas.read_sql_query('SELECT * FROM {0}'.format(experiment_name),
                                           connection)
                df['dt_created'] = pandas.to_datetime(df['dt_created'])

            if plot_metric_over_time:
                # Generate some plots, metric over time
                dirname = 'img/results/{0}/'.format(experiment_name)
                os.makedirs(dirname, exist_ok=True)
                print('Plotting metric over time')
                iterations = 0
                for metric_name in metric_names:
                    if not df[metric_name].any():
                        continue
                    print_progress_bar(iterations, len(metric_names))
                    ax = df.plot(x='dt_created', y=metric_name, style='.')
                    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d %H:00'))
                    plt.savefig(dirname + 'dt_created_{0}.png'.format(metric_name))
                    plt.clf()
                    plt.close()
                    iterations += 1
                    print_progress_bar(iterations, len(metric_names))

                # plot optimize grouped over time
                assert df['dt_created'].is_monotonic  # sorting should not be a problem but we are lazy
                y_means = []
                df = df.drop_duplicates(subset='dt_created')
                timestamps = pandas.DatetimeIndex(df.dt_created).asi8 // 10 ** 9
                iterations = 0
                print('Preparing plot {0} over time'.format(optimize))
                for x in timestamps:
                    print_progress_bar(iterations, len(timestamps))
                    not_after_x = 'CAST(strftime(\'%s\', dt_created) AS INT) <= {0}'.format(x)
                    param = get_best_params(additional_condition=not_after_x,
                                            param_names=param_names,
                                            experiment_name=experiment_name,
                                            larger_result_is_better=larger_result_is_better,
                                            optimize=optimize)
                    scores, mean, std, conf = get_results_for_params(optimize, experiment_name, param,
                                                                     additional_condition=not_after_x)
                    y_means.append(mean)
                    iterations += 1
                    print_progress_bar(iterations, len(timestamps))
                df['score'] = y_means

                ax = df.plot(x='dt_created', y='score')
                ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%m-%d %H:00'))
                plt.savefig(dirname + '{0}_over_time.png'.format(optimize))
                plt.clf()
                plt.close()

    return best_params, best_score


def predictions_for_parameters(experiment_name: str, parameters, show_progress=False):
    result_ids = result_ids_for_parameters(experiment_name, parameters)
    if not show_progress:
        return [
            predictions_for_result_id(experiment_name, result_id)
            for result_id in result_ids
        ]
    else:
        return [
            predictions_for_result_id(experiment_name, result_id)
            for result_id in ProgressBar(result_ids)
        ]


def result_ids_for_parameters(experiment_name, parameters: Dict[str, Any]):
    condition, parameters = only_specific_parameters_condition(parameters)
    cursor = connection.cursor()
    cursor.execute('''
    SELECT rowid FROM {0}
    WHERE {1}
    ORDER BY rowid
    '''.format(experiment_name, condition), parameters)
    result_ids = [row[0] for row in cursor.fetchall()]
    return result_ids


def creation_times_for_parameters(experiment_name, parameters):
    condition, parameters = only_specific_parameters_condition(parameters)
    cursor = connection.cursor()
    cursor.execute('''
    SELECT dt_created FROM {0}
    WHERE {1}
    ORDER BY rowid
    '''.format(experiment_name, condition), parameters)
    creation_times = [row[0] for row in cursor.fetchall()]
    return creation_times


def predictions_for_result_id(experiment_name: str, result_id):
    cursor = connection.cursor()
    cursor.execute('''
    SELECT name, dataset, y_pred, y_true FROM {0}_predictions
    WHERE result_id = ?
    '''.format(experiment_name, ), (result_id,))
    predictions = [{
        'name': row[0],
        'dataset': row[1],
        'y_pred': row[2],
        'y_true': row[3],
    } for row in cursor.fetchall()]

    return predictions


def list_difficult_samples(experiment_name,
                           loss_functions,
                           dataset,
                           max_losses_to_average=20,
                           additional_condition='1',
                           additional_parameters=(),
                           also_print=False):
    names = all_sample_names(dataset, experiment_name)
    cursor = connection.cursor()
    if 'epochs' in additional_condition:
        try:
            print('Creating index to fetch results faster (if not exists)...')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS {0}_by_name_epochs_dataset
                ON {0} (name, epochs, dataset)'''.format(experiment_name))
        except Exception as e:  # TODO check error type
            print(e)
            pass
    cursor = connection.cursor()
    table = []
    print('Fetching results for names...')
    for name in ProgressBar(names):
        if additional_condition == '1':
            additional_join = ''
        else:
            additional_join = 'JOIN {0} ON {0}.rowid = result_id'.format(experiment_name)
        if isinstance(max_losses_to_average, int) is not None and max_losses_to_average != inf:
            limit_string = f'LIMIT ?'
            limit_args = [max_losses_to_average]
        elif max_losses_to_average is None or max_losses_to_average == inf:
            limit_string = ''
            limit_args = []
        else:
            raise ValueError
        cursor.execute('''
            SELECT y_pred, y_true
            FROM {0}
             CROSS JOIN {0}_predictions ON {0}.rowid = result_id
            WHERE name = ? AND dataset = ? AND ({1})
            {3}'''.format(experiment_name, additional_condition, ..., limit_string), (name,
                                                                                      dataset,
                                                                                      *additional_parameters,
                                                                                      *limit_args,))
        data = cursor.fetchall()
        if len(data) > 0:
            def aggregate(xs):
                if len(set(xs)) == 1:
                    return xs[0]
                else:
                    return numpy.mean(xs)

            table.append((*[aggregate([loss_function(y_pred=y_pred, y_true=y_true, name=name)
                                       for y_pred, y_true in data])
                            for loss_function in loss_functions],
                          name, len(data)))
    print('sorting table...')
    table.sort(reverse=True)
    if also_print:
        print('stringifying table...')
        print(my_tabulate(table,
                          headers=[loss_function.__name__ for loss_function in loss_functions] + ['name', '#results'],
                          tablefmt='pipe'))
    return table


def all_sample_names(dataset, experiment_name):
    cursor = connection.cursor()
    print('Creating index to have faster queries by name (if not exists)...')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS {0}_predictions_by_name_and_dataset
        ON {0}_predictions (dataset, name)'''.format(experiment_name))
    print('Fetching all names...')
    names = []
    last_found = ''  # smaller than all other strings
    while True:
        cursor.execute('SELECT name '
                       'FROM {0}_predictions '
                       'WHERE dataset = ? AND name > ?'
                       'LIMIT 1'.format(experiment_name), (dataset, last_found))
        row = cursor.fetchone()
        if row is None:
            break
        names.append(row[0])
        last_found = row[0]
    return names


def only_specific_parameters_condition(parameters: Dict[str, Any]) -> Tuple[str, Tuple]:
    items = list(parameters.items())  # to have the correct ordering
    return '(' + ' AND '.join(f'"{name}" IS ?' for name, _ in items) + ')', \
           tuple(value for name, value in items)


def only_best_parameters_condition(experiment_name: str,
                                   larger_result_is_better: bool,
                                   optimize: str,
                                   param_names: List[str],
                                   additional_condition: str = '1') -> Tuple[str, Tuple]:
    parameters = get_best_params(experiment_name=experiment_name,
                                 larger_result_is_better=larger_result_is_better,
                                 optimize=optimize,
                                 param_names=param_names,
                                 additional_condition=additional_condition)
    return only_specific_parameters_condition(parameters)


def get_results_for_params(metric, experiment_name, parameters, confidence=0.95,
                           additional_condition='1'):
    param_names = list(parameters.keys())
    cursor = connection.cursor()
    params_equal = '\nAND '.join('"' + param_name + '" IS ?' for param_name in param_names)
    cursor.execute(
        '''
        SELECT {0}
        FROM {1}
        WHERE {2} AND ({3})
        '''.format(metric,
                   experiment_name,
                   params_equal,
                   additional_condition),
        tuple(parameters[name] for name in param_names)
    )
    # noinspection PyShadowingNames
    scores = [row[0] if row[0] is not None else nan for row in cursor.fetchall()]
    if len(scores) == 0:
        return scores, nan, nan, nan
    return scores, numpy.mean(scores), numpy.std(scores), mean_confidence_interval_size(scores, confidence)


def num_results_for_params(param_names, experiment_name, parameters,
                           additional_condition='1'):
    cursor = connection.cursor()
    params_equal = '\nAND '.join('"' + param_name + '" IS ?' for param_name in param_names)
    cursor.execute(
        '''
        SELECT COUNT(*)
        FROM {0}
        WHERE {1} AND ({2})
        '''.format(experiment_name,
                   params_equal,
                   additional_condition),
        tuple(parameters[name] for name in param_names)
    )
    return cursor.fetchone()[0]


def get_best_params(experiment_name: str,
                    larger_result_is_better: bool,
                    optimize: str,
                    param_names: List[str],
                    additional_condition='1') -> Optional[Parameters]:
    cursor = connection.cursor()
    param_names_comma_separated = ','.join('"' + param_name + '"' for param_name in param_names)
    worst_score = '-1e999999' if larger_result_is_better else '1e999999'
    # noinspection SqlAggregates
    cursor.execute('''
            SELECT * FROM {0} AS params
            GROUP BY {1}
            HAVING ({4})
            ORDER BY AVG(CASE WHEN params.{3} IS NULL THEN {5} ELSE params.{3} END) {2}, MIN(rowid) ASC
            LIMIT 1
            '''.format(experiment_name,
                       param_names_comma_separated,
                       'DESC' if larger_result_is_better else 'ASC',
                       optimize,
                       additional_condition,
                       worst_score, ))
    row = cursor.fetchone()
    if row is None:
        raise EmptyTableError()
    else:
        return params_from_row(cursor.description, row, param_names=param_names)


def params_from_row(description, row, param_names=None) -> Parameters:
    best_params = {}
    for idx, column_description in enumerate(description):
        column_name = column_description[0]
        if param_names is None or column_name in param_names:
            best_params[column_name] = row[idx]
    return best_params


def create_experiment_tables_if_not_exists(experiment_name, params, metric_names):
    cursor = connection.cursor()
    param_names = set(param.name for param in params)
    initial_params = {param.name: param.initial_value for param in params}
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS {0}(
        rowid INTEGER PRIMARY KEY,
        dt_created DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    '''.format(experiment_name))
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS {0}_predictions(
        rowid INTEGER PRIMARY KEY,
        dataset TEXT NOT NULL,
        y_true BLOB,
        y_pred BLOB,
        name TEXT NOT NULL, -- used to identify the samples
        result_id INTEGER NOT NULL REFERENCES {0}(rowid),
        UNIQUE(result_id, dataset, name) -- gives additional indices
    )
    '''.format(experiment_name))
    connection.commit()

    for param_name in param_names:
        default_value = initial_params[param_name]
        add_parameter_column(experiment_name, param_name, default_value)

    for metric_name in metric_names:
        add_metric_column(experiment_name, metric_name)


def add_metric_column(experiment_name, metric_name, verbose=0):
    cursor = connection.cursor()
    try:
        cursor.execute('ALTER TABLE {0} ADD COLUMN "{1}" NUMERIC DEFAULT NULL'.format(experiment_name,
                                                                                      metric_name))
    except sqlite3.OperationalError as e:
        if 'duplicate column name' not in e.args[0]:
            raise
    else:
        if verbose:
            print(f'WARNING: created additional column {metric_name}. This may or may not be intentional')
    connection.commit()


def add_parameter_column(experiment_name, param_name, default_value, verbose=0):
    cursor = connection.cursor()
    try:
        if isinstance(default_value, str):
            # default_value = default_value.replace("\\", "\\\\")
            default_value = default_value.replace("'", "''")
            default_value = "'" + default_value + "'"
        if default_value is None:
            default_value = 'NULL'
        if default_value == math.inf:
            default_value = '1e999999999'
        if default_value == -math.inf:
            default_value = '-1e999999999'
        query = 'ALTER TABLE {0} ADD COLUMN "{1}" BLOB DEFAULT {2}'.format(experiment_name, param_name, default_value)
        cursor.execute(query)
    except sqlite3.OperationalError as e:
        if 'duplicate column name' not in e.args[0]:
            raise
    else:
        if verbose:
            print(f'WARNING: created additional column {param_name} with default value {default_value}. '
                  f'This may or may not be intentional')
    connection.commit()


def markdown_table(all_results, sort_by):
    rows = [list(result['params'].values()) + [result['mean'], result['std'], result['conf'], result['all']] for result
            in all_results]
    rows.sort(key=sort_by)
    table = my_tabulate(rows, headers=list(all_results[0]['params'].keys()) + ['mean', 'std', 'conf', 'results'],
                        tablefmt='pipe')
    return table


def validate_parameter_set(params):
    if len(params) == 0:
        raise ValueError('Parameter set empty')
    for i, param in enumerate(params):
        # noinspection PyUnusedLocal
        other_param: Parameter
        for other_param in params[i + 1:]:
            if param.name == other_param.name and param.initial_value != other_param.initial_value:
                msg = '''
                A single parameter cant have multiple initial values. 
                Parameter "{0}" has initial values "{1}" and "{2}"
                '''.format(param.name, param.initial_value, other_param.initial_value)
                raise ValueError(msg)


def run_name(parameters=None) -> str:
    if parameters is None:
        parameters = {}
    shorter_parameters = {
        shorten_name(k): shorten_name(v)
        for k, v in parameters.items()
    }
    return ((str(datetime.now()) + str(shorter_parameters).replace(' ', ''))
            .replace("'", '')
            .replace('"', '')
            .replace(":", '_')
            .replace(",", '')
            .replace("_", '')
            .replace("<", '')
            .replace(">", '')
            .replace("{", '')
            .replace("}", ''))


def plot_experiment(metric_names,
                    experiment_name: str,
                    plot_name: str,
                    param_names: List[str],
                    params_list: List[Parameters],
                    evaluate: EvaluationFunction,
                    ignore: List[str] = None,
                    plot_shape=None,
                    metric_limits: Dict = None,
                    titles=None,
                    natural_metric_names: Dict[str, str] = None,
                    min_runs_per_params=0,
                    single_plot_width=6.4,
                    single_plot_height=4.8, ):
    if natural_metric_names is None:
        natural_metric_names = {}
    for parameters in params_list:
        if 'epochs' not in parameters:
            raise ValueError('`plot_experiment` needs the number of epochs to plot (`epochs`)')

    if metric_limits is None:
        metric_limits = {}
    if ignore is None:
        ignore = []
    if titles is None:
        titles = [None for _ in params_list]

    if plot_shape is None:
        width = ceil(sqrt(len(params_list)))
        plot_shape = (ceil(len(params_list) / width), width,)
    else:
        width = plot_shape[1]
    plot_shape_offset = 100 * plot_shape[0] + 10 * plot_shape[1]
    axes: Dict[int, Axes] = {}
    legend: List[str] = []
    results_dir = 'img/results/{0}/over_time/'.format(experiment_name)
    os.makedirs(results_dir, exist_ok=True)

    metric_names = sorted(metric_names, key=lambda m: (metric_limits.get(m, ()), metric_names.index(m)))
    print(metric_names)
    plotted_metric_names = []
    iterations = 0

    for plot_std in [False, True]:
        plt.figure(figsize=(single_plot_width * plot_shape[1], single_plot_height * plot_shape[0]))
        for idx, metric in enumerate(metric_names):
            print_progress_bar(iterations, 2 * (len(metric_names) - len(ignore)))
            limits = metric_limits.get(metric, None)
            try:
                next_limits = metric_limits.get(metric_names[idx + 1], None)
            except IndexError:
                next_limits = None
            if metric in ignore:
                continue
            sqlite_infinity = '1e999999'
            metric_is_finite = '{0} IS NOT NULL AND {0} != {1} AND {0} != -{1}'.format(metric, sqlite_infinity)
            for plot_idx, parameters in enumerate(params_list):
                while num_results_for_params(param_names=param_names,
                                             experiment_name=experiment_name,
                                             parameters=parameters, ) < min_runs_per_params:
                    print('Doing one of the missing experiments for the plot:')
                    print(parameters)
                    results = try_parameters(experiment_name=experiment_name,
                                             evaluate=evaluate,
                                             params=parameters, )
                    assert any(result.parameters == parameters for result in results)

                contains_avg_over = 'average_over_last_epochs' in parameters

                total_epochs = parameters['epochs']
                history = []
                lower_conf_limits = []
                upper_conf_limits = []
                for epoch_end in range(total_epochs):
                    current_parameters = parameters.copy()
                    if contains_avg_over:
                        current_parameters['average_over_last_epochs'] = None
                    current_parameters['epochs'] = epoch_end + 1
                    scores, mean, std, conf = get_results_for_params(
                        metric=metric,
                        experiment_name=experiment_name,
                        parameters=current_parameters,
                        additional_condition=metric_is_finite
                    )
                    history.append(mean)
                    if plot_std:
                        lower_conf_limits.append(mean - 1.959964 * std)
                        upper_conf_limits.append(mean + 1.959964 * std)
                    else:
                        lower_conf_limits.append(mean - conf)
                        upper_conf_limits.append(mean + conf)
                x = list(range(len(history)))
                if plot_shape_offset + plot_idx + 1 not in axes:
                    # noinspection PyTypeChecker
                    ax: Axes = plt.subplot(plot_shape_offset + plot_idx + 1)
                    assert isinstance(ax, Axes)
                    axes[plot_shape_offset + plot_idx + 1] = ax
                ax = axes[plot_shape_offset + plot_idx + 1]
                ax.plot(x, history)
                ax.fill_between(x, lower_conf_limits, upper_conf_limits, alpha=0.4)
                if titles[plot_idx] is not None:
                    ax.set_title(titles[plot_idx])
                if limits is not None:
                    ax.set_ylim(limits)
                ax.set_xlim(0, max(total_epochs, ax.get_xlim()[1]))
                current_row = plot_idx // width
                if current_row == plot_shape[0] - 1:
                    ax.set_xlabel('Epoch')
            natural_name = natural_metric_names.get(metric, metric)
            if plot_std:
                legend += ['mean ' + natural_name, '1.96σ of {0}'.format(natural_name)]
            else:
                legend += ['mean ' + natural_name, '95% conf. of mean {0}'.format(natural_name)]

            plotted_metric_names.append(metric)
            if limits is None or next_limits is None or limits != next_limits:
                legend = legend[0::2] + legend[1::2]
                for ax in axes.values():
                    ax.legend(legend)
                if plot_std:
                    plt.savefig(results_dir + plot_name + '_' + ','.join(plotted_metric_names) + '_std' + '.png')
                else:
                    plt.savefig(results_dir + plot_name + '_' + ','.join(plotted_metric_names) + '.png')
                plt.clf()
                plt.close()
                plt.figure(figsize=(single_plot_width * plot_shape[1], single_plot_height * plot_shape[0]))
                axes = {}
                plotted_metric_names = []
                legend = []
            iterations += 1
            print_progress_bar(iterations, 2 * (len(metric_names) - len(ignore)))
    plt.clf()
    plt.close()
