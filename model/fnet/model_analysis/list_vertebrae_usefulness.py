import json
import os.path
import os.path
from typing import List

from lib.my_logger import logging
from model.fnet.model_analysis.list_difficult_vertebrae import ListDifficultVertebrae
from vertebra_usefulness.result import UsefulnessResult
from vertebra_usefulness.results_table import UsefulnessResultsTable
from vertebra_usefulness.summary_table import VertebraUsefulnessSummaryTable

try:
    from annotation_supplement_ey import LIKELY_DIFFICULTIES
except ImportError:
    LIKELY_DIFFICULTIES = {}
from lib.callbacks import RecordBatchLosses
from model.fnet.model_analysis.analyze_trained_models import ModelAnalyzer, tasks_of_any_model, split_name
from tasks import VertebraClassificationTask


class ListVertebraUsefulness(ListDifficultVertebrae):

    # noinspection PyAttributeOutsideInit
    def before_model(self, model_path: str, ):
        super().before_model(model_path)
        self.table = UsefulnessResultsTable()

    def analyze_batch(self, batch, y_preds, names):
        _, ys, _ = batch
        record_batch_losses = RecordBatchLosses()
        self.dummy_loss_model.evaluate(x=y_preds[self.task_idx],
                                       y=ys[self.task_idx],
                                       verbose=0,
                                       callbacks=[record_batch_losses],
                                       batch_size=1,
                                       steps=len(names),
                                       return_dict=True)
        metrics = record_batch_losses.metrics
        assert len(metrics) == len(names)
        # (output,), (ground_truth,) = next(val_gen)
        training_samples = json.loads(self.config['training_info'])['training_samples']
        test_samples = []
        test_losses = []
        ground_truths = []
        for sample_idx in range(len(names)):
            # output = y_preds[self.task_idx][sample_idx]
            ground_truth = ys[self.task_idx][sample_idx]
            patient, vertebra = split_name(names[sample_idx])
            test_samples.append((patient, vertebra))
            test_losses.append(metrics[sample_idx]['loss'])
            ground_truths.append(self.task().class_name_of_label(ground_truth))
        new_row = UsefulnessResult(
            training_samples=training_samples,
            test_samples=test_samples,
            test_losses=test_losses,
            ground_truths=ground_truths,
        )
        self.table.append(new_row)

    def after_multiple_models(self, results: List[UsefulnessResultsTable]):
        tasks = tasks_of_any_model(self.model_files_being_analyzed, self.evaluator, only_tasks=self.ONLY_TASKS)
        if len(tasks) > self.max_tasks_per_model:
            raise RuntimeError(f'Please increase the max_tasks_per_model to at least {len(tasks)}.')
        try:
            task = tasks[self.task_idx]
        except IndexError:
            raise self.InvalidTaskIndex
        if not isinstance(task, VertebraClassificationTask):
            raise self.InvalidTaskIndex
        if not isinstance(task, VertebraClassificationTask):
            raise self.InvalidTaskIndex
        logging.info(f'Summarizing vertebra usefulness results for task {task.output_layer_name()}...')
        vertebra_usefulness_table = VertebraUsefulnessSummaryTable.from_results_table(UsefulnessResultsTable.concatenate(results),
                                                                                      likely_difficulties=self.likely_difficulties,
                                                                                      task=task,
                                                                                      model_dir=self.model_dir,
                                                                                      compute_confidence=False)
        ModelAnalyzer.after_multiple_models(self, results)
        self.write_summary_to_disk(vertebra_usefulness_table)
        return vertebra_usefulness_table

    def after_model_directory(self, results: VertebraUsefulnessSummaryTable):
        return ModelAnalyzer.after_model_directory(self, results)

    def write_summary_to_disk(self, results):
        base_name = os.path.join(self.to_dir(), f'vertebra_usefulness_{results.task.output_layer_name()}')
        self.serializer().create_directory_if_not_exists(self.to_dir())
        print('Writing', base_name + '.*')
        self.serializer().save_dataframe(base_name + '.csv', results.df)
        self.serializer().save_dataframe(results.df)
        self.serializer().save_pkl(base_name + '.pkl', results)

        md_string = self.table_description(results.task)
        md_string += '\n'
        md_string += results.to_markdown()

        self.serializer().save_text(base_name + '.md', text=md_string)

    @staticmethod
    def table_description(task: VertebraClassificationTask):
        return ''

    def columns(self):
        raise NotImplementedError

    @staticmethod
    def sort_results(results, task: VertebraClassificationTask):
        raise NotImplementedError
