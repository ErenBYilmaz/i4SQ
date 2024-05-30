import os.path
from copy import deepcopy
from typing import List, Union, Optional

import numpy

from hiwi import ImageList
from model.fnet.evaluate import FNetParameterEvaluator
from model.fnet.model_analysis.analyze_trained_models import SingleModelEvaluationResult, ModelAnalyzer, UseSpecificDataset, split_name
from tasks import VertebraTask


class GeneratePseudoLabels(ModelAnalyzer):
    def __init__(self, evaluator: FNetParameterEvaluator, dataset: Union[ImageList, str], name: Optional[str] = None, skip_existing=None):
        super().__init__(evaluator=evaluator,
                         analysis_needs_xs=False,
                         skip_existing=skip_existing,
                         model_level_caching=True,
                         )
        self.USE_DATASET_BY_DEFAULT = UseSpecificDataset(dataset, name)
        self.result = None

    def to_subdir(self):
        return 'pseudolabels'

    def before_multiple_models(self, model_files):
        super().before_multiple_models(model_files)
        self.result = None

    def analyze_batch(self, batch, y_preds, names):
        if self.result is not None:
            raise RuntimeError('Already analyzed a different batch for this dataset')
        model_name = os.path.basename(self.model_path)
        to_file = os.path.join(self.to_dir(), f'{model_name}.iml')
        self.serializer().create_directory_if_not_exists(self.to_dir())
        dataset = deepcopy(self.dataset)
        img_by_id = {
            image['patient_id']: image
            for image in dataset
        }
        for sample_idx, name in enumerate(names):
            patient_id, vertebra = split_name(name)
            img = img_by_id[patient_id]
            for task_idx, task in enumerate(self.tasks):
                task: VertebraTask
                label: numpy.ndarray = y_preds[task_idx][sample_idx]
                img.parts[vertebra].setdefault('tool_outputs', {}).setdefault(model_name, {}).setdefault(task.output_layer_name(), []).append(label.tolist())
        dataset.dump(os.path.abspath(to_file), relative_paths=False)
        print(f'Wrote {os.path.abspath(to_file)}')
        self.result = dataset

    def after_dataset(self) -> SingleModelEvaluationResult:
        super().after_dataset()
        result = self.result
        self.result = None
        return result

    def after_multiple_models(self, results: List[SingleModelEvaluationResult]):
        self.serializer().create_directory_if_not_exists(self.to_dir())
        to_file = os.path.join(self.to_dir(), f'combined.iml')

        combined_dataset = self.merge_image_list_pseudo_labels(results)
        combined_dataset.dump(os.path.abspath(to_file))
        print(f'Wrote {os.path.abspath(to_file)}')
        return combined_dataset

    @staticmethod
    def merge_image_list_pseudo_labels(resultss):
        combined_dataset = ImageList()
        img_by_id = {}
        for results in resultss:
            for result in results:
                for new_img in result:
                    if new_img['patient_id'] not in img_by_id:
                        combined_dataset.append(new_img)
                        img_by_id[new_img['patient_id']] = new_img
                    else:
                        img = img_by_id[new_img['patient_id']]
                        for vertebra in new_img.parts:
                            if 'tool_outputs' not in img.parts[vertebra]:
                                continue
                            for model_name in new_img.parts[vertebra]['tool_outputs']:
                                for task, labels in new_img.parts[vertebra]['tool_outputs'][model_name].items():
                                    img.parts[vertebra].setdefault('tool_outputs', {}).setdefault(model_name, {}).setdefault(task, []).extend(labels)
        return combined_dataset
