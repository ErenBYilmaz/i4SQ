from contextlib import ContextDecorator
from typing import List

from lib.my_logger import logging
from lib.parameter_search import Parameter
from lib.util import EBC
from model.fnet.hyperparameter_set import HyperparameterSet


class UsingDifferentParameters(EBC, ContextDecorator):
    def __enter__(self):
        raise NotImplementedError('Abstract')

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError('Abstract')


class UsingSameParameters(UsingDifferentParameters):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class UsingDifferentParameterValue(UsingDifferentParameters):
    def __init__(self, default_param: Parameter, value, value_name: str = None, ignore_unchanged=False,
                 also_update_model_subdir=False, verbose=True, model_subdir_parameter: Parameter = None):
        if value_name is None:
            value_name = value
        self.value_name = value_name
        self.ignore_unchanged = ignore_unchanged
        self.value = value
        self._value_before = None
        self.default_param = default_param
        self.model_subdir_parameter = model_subdir_parameter
        self.maybe_patch_model_name: UsingDifferentParameters
        if also_update_model_subdir:
            self.maybe_patch_model_name = UsingDifferentParameterValue(
                default_param=self.model_subdir_parameter,
                value=f'{self.model_subdir_parameter.initial_value}_{self.value_name}',
                also_update_model_subdir=False,
                verbose=verbose
            )
        else:
            self.maybe_patch_model_name = UsingSameParameters()
        self.verbose = verbose

    def __enter__(self):
        assert self._value_before is None
        self._value_before = self.default_param.initial_value
        if self._value_before != self.value or not self.ignore_unchanged:
            self.maybe_patch_model_name.__enter__()
            if self.verbose:
                logging.info(f'Setting {self.default_param.name} = {self.value}')
            self.default_param.initial_value = self.value

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._value_before != self.value or not self.ignore_unchanged:
            assert self.default_param.initial_value == self.value
            self.default_param.initial_value = self._value_before
            self.maybe_patch_model_name.__exit__(exc_type, exc_val, exc_tb)
        self._value_before = None

    @staticmethod
    def with_different_settings(default_param: Parameter, values):
        for v in values:
            u = UsingDifferentParameterValue(default_param, v)
            with u:
                yield u


class MultipleDifferentParameterValues(UsingDifferentParameters):
    def __init__(self, using_different_parameter_values: List[UsingDifferentParameters], name=None, verbose=True,
                 model_subdir_parameter: Parameter = None):
        self.name = name
        self.using_different_parameter_values = using_different_parameter_values
        self.model_subdir_parameter = model_subdir_parameter
        if self.name is not None:
            self.maybe_patch_model_name = UsingDifferentParameterValue(
                default_param=self.model_subdir_parameter,
                value=f'{self.model_subdir_parameter.initial_value}_{name}',
                also_update_model_subdir=False,
                verbose=verbose
            )
        else:
            self.maybe_patch_model_name = UsingSameParameters()
        self.using_different_parameter_values.append(self.maybe_patch_model_name)
        self.verbose = verbose

    def __enter__(self):
        for u in self.using_different_parameter_values:
            u.__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        for u in reversed(self.using_different_parameter_values):
            u.__exit__(exc_type, exc_val, exc_tb)


class ExperimentNameSuffix(MultipleDifferentParameterValues):
    def __init__(self, name=None,
                 model_subdir_parameter: Parameter = None):
        super().__init__([], name=name, verbose=False, model_subdir_parameter=model_subdir_parameter)


class UsingAnotherHyperparameterSet(MultipleDifferentParameterValues):
    def __init__(self, original_hyperparameter_set: HyperparameterSet,
                 new_hyperparameter_set: HyperparameterSet, name=None,
                 verbose=False,
                 model_subdir_parameter=None):
        self.original_hyperparameter_set = original_hyperparameter_set
        self.new_hyperparameter_set = new_hyperparameter_set
        new_values = new_hyperparameter_set.default_values()
        super().__init__([
            UsingDifferentParameterValue(p, new_values[p.name], ignore_unchanged=True, also_update_model_subdir=False, verbose=False)
            for p in original_hyperparameter_set.hyper_parameters()
        ], name=name, verbose=verbose, model_subdir_parameter=model_subdir_parameter)
