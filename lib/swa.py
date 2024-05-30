from typing import Optional

from lib.additional_validation_sets import AdditionalValidationSets, MetricName
from lib.clone_compiled_model import clone_model_or_tool
from lib.image_processing_tool import TrainableImageProcessingTool


class SWA(AdditionalValidationSets):
    def __init__(self,
                 start_batch: int,
                 alpha_1,
                 alpha_2,
                 cycle_length: int = 1,
                 validation_sets=None,
                 verbose=1,
                 batch_size=None,
                 record_original_history=False,
                 record_predictions=False,
                 keep_best_model_by_metric: Optional[MetricName] = None,
                 larger_result_is_better: Optional[bool] = None,
                 log_using_wandb_run=None,):
        if validation_sets is None:
            validation_sets = []
        if start_batch is None or cycle_length is None:
            raise ValueError
        super(SWA, self).__init__(validation_sets,
                                  verbose,
                                  batch_size,
                                  record_original_history=record_original_history,
                                  record_predictions=record_predictions,
                                  keep_best_model_by_metric=keep_best_model_by_metric,
                                  larger_result_is_better=larger_result_is_better,
                                  log_using_wandb_run=log_using_wandb_run)
        self.start_batch = start_batch
        self.c = cycle_length
        self.w_swa = None
        self.batch = None
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        if self.c == 1 and alpha_2 is not alpha_1:
            raise ValueError('There is no cycle so why are there two different learning rates?')
        if self.c == 0:
            raise ValueError('Cycle length must be positive.')

        self.swa_model: Optional[TrainableImageProcessingTool] = None
        self.weights_changed = False

    def on_train_begin(self, logs=None):
        self.batch = 0
        if 'samples' in self.params:
            num_batches = (self.params['samples'] // self.params['batch_size']) * self.params['epochs']
            if self.start_batch >= num_batches:
                raise ValueError('Can not start SWA after training is finished')
        # validation stuff
        self.swa_model: TrainableImageProcessingTool = clone_model_or_tool(self.model)
        self.weights_changed = False
        super(SWA, self).on_train_begin(logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        super(SWA, self).on_epoch_end(epoch=epoch, logs=logs)

    def model_to_evaluate(self) -> Optional[TrainableImageProcessingTool]:
        if self.batch < self.start_batch:
            return None
        if self.weights_changed and self.w_swa is not None:  # TODO test
            self.swa_model.set_weights(self.w_swa)
            self.weights_changed = False
        return self.swa_model

    def prefix(self):
        return 'swa_c' + str(self.c) + '_a1' + str(self.alpha_1) + '_a2' + str(self.alpha_2) + '_si' + str(
            self.start_batch) + '_'

    def on_batch_begin(self, batch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        if self.batch > self.start_batch:
            i = self.batch - self.start_batch
            # set learning rate
            if self.c == 1:
                t = 0
            else:
                t = 1 / (self.c - 1) * ((i - 1) % self.c)
            lr = (1 - t) * self.alpha_1 + t * self.alpha_2
            from tensorflow import keras
            keras.backend.set_value(self.model.optimizer.lr, lr)

    def on_batch_end(self, batch, logs=None):
        if self.batch == self.start_batch:
            self.w_swa = self.model.get_weights()  # Initialize weights
        elif self.batch > self.start_batch:
            i = self.batch - self.start_batch
            # Update average
            if i % self.c == 0:
                n_models = (i / self.c)
                iterate = range(len(self.w_swa)) if isinstance(self.w_swa, list) else self.w_swa.keys()
                for j in iterate:  # iterate all weight layers
                    self.w_swa[j] = (self.w_swa[j] * n_models + self.model.get_weights()[j]) / (n_models + 1)
        self.weights_changed = True
        self.batch += 1


class SWAWithoutLrChange(SWA):
    def __init__(self,
                 start_batch,
                 cycle_length=1,
                 validation_sets=None,
                 verbose=1,
                 batch_size=None,
                 record_original_history=False,
                 record_predictions=False,
                 keep_best_model_by_metric: Optional[MetricName] = None,
                 larger_result_is_better: Optional[bool] = None,
                 log_using_wandb_run=None):
        super(SWAWithoutLrChange, self).__init__(
            start_batch=start_batch,
            alpha_1=None,
            alpha_2=None,
            cycle_length=cycle_length,
            validation_sets=validation_sets,
            verbose=verbose,
            batch_size=batch_size,
            record_original_history=record_original_history,
            record_predictions=record_predictions,
            keep_best_model_by_metric=keep_best_model_by_metric,
            larger_result_is_better=larger_result_is_better,
            log_using_wandb_run=log_using_wandb_run,
        )

    def on_batch_begin(self, batch, logs=None):
        pass  # dont change learning rate

    def prefix(self):
        return 'swa_c' + str(self.c) + '_sb' + str(self.start_batch) + '_'
