from abc import abstractmethod
from typing import Optional, List, Type, Tuple


class DeepLearningBackendInterface():
    @abstractmethod
    def metrics_for_binary_task(self, threshold: Optional[float] = 0.5) -> list:
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def weighted_metrics_for_binary_task(self) -> list:
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def multi_task_weighted_metrics(self, num_classes: int) -> list:
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def multi_task_metrics(self, num_classes: int) -> list:
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def genant_score_custom_loss_metrics(self) -> list:
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def weighted_genant_score_regression_accuracy_metrics(self) -> list:
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def regression_metrics(self) -> list:
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def regression_weighted_metrics(self) -> list:
        raise NotImplementedError('Abstract method')

    def genant_score_custom_loss(self):
        return self.genant_score_custom_loss_metrics()[0]

    @abstractmethod
    def gpu_error_classes(self) -> Tuple[Type[Exception], ...]:
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def empty_model(self, loss):
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def clear_session(self):
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def limit_memory_usage(self):
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def categorical_crossentropy_loss(self):
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def binary_crossentropy_loss(self):
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def mse_loss(self):
        raise NotImplementedError('Abstract method')

    @staticmethod
    def one_hot_encode(num_classes: int, class_idx: int) -> List[int]:
        result = [0] * num_classes
        result[class_idx] = 1
        return result

    def memory_leak_cleanup(self):
        pass

    @abstractmethod
    def reset_model_metrics(self, model, only_metrics: Optional[List[str]] = None):
        raise NotImplementedError('Abstract method')

    @abstractmethod
    def reuse_training_workers(self) -> bool:
        raise NotImplementedError('Abstract method')
