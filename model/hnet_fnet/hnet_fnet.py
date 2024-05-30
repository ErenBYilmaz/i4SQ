import json
import os

import pandas

from hiwi import ImageList
from lib.image_processing_tool import ImageProcessingPipeline
from lib.main_wrapper import main_wrapper
from lib.my_logger import logging
from model.fnet.evaluate import FNetParameterEvaluator
from model.fnet.fnet import FNet
from model.hnet.hnet import HNet, hnet_by_version
from model.hnet_fnet.fracture_evaluation import classify_fractures, coords_from_iml, metric_summary


class HNetFNetPipeline(ImageProcessingPipeline):
    def __init__(self, hnet: HNet, fnet: FNet, results_subdir: str = ''):
        super(HNetFNetPipeline, self).__init__(tools=[hnet, fnet])
        self.results_subdir = results_subdir
        self.hnet = hnet
        self.fnet = fnet
        self.fnet.use_coordinates = self.hnet.using_predicted_coordinates

    def name(self) -> str:
        return 'HNetFNetPipeline'

    def evaluate_whole_pipeline(self, iml_with_intermediate_predictions_and_evaluations: ImageList):
        if self.whole_pipeline_evaluation_task_name is None:
            logging.info('Cannot evaluate whole pipeline, no task name given')
            return
        with self.hnet.using_predicted_coordinates(iml_with_intermediate_predictions_and_evaluations):
            classify_fractures(
                fracture_model_name=self.fnet.name(),
                test_list=iml_with_intermediate_predictions_and_evaluations,
                coordinates=coords_from_iml(iml_with_intermediate_predictions_and_evaluations),
                verbose=1,
                task=self.fnet.tasks.by_name('01v23_012v3_0v123'),
                results_path=self.results_path(),
                threshold=0.5,
                exclude_center_6=True,
            )

    def results_path(self):
        return os.path.join(self.hnet.tmp_results_dir, self.results_subdir, self.fnet.name())


@main_wrapper
def main():
    skip_existing = True
    whole_results_table = []
    for task_name in ['hufo', 'hafo']:
        for test_mode in ['val', 'test']:
            for exclude_c1_c2 in [True, False]:
                for classification_threshold in [0.5, 0.7]:
                    results_subdir = f'{test_mode}_{task_name}_{str(classification_threshold)}'
                    data_source = TraumaDataSource(exclude_c1_c2=exclude_c1_c2)
                    dataset_name_for_evaluation = data_source.name()
                    results_dir = fr'img/generated/{dataset_name_for_evaluation}/trauma_repetition_no_bc_20230314'
                    model_dir = os.path.join('models', 'trauma_repetition_no_bc_20230314')
                    # model_dir = os.path.join(results_dir, 'models')
                    json_output_files = []
                    for model_name in os.listdir(model_dir):
                        if not model_name.endswith('.h5'):
                            continue
                        pipeline = HNetFNetPipeline(
                            hnet=(HNet(model_exe_path=hnet_by_version(7).exe_path,
                                       results_dir=os.path.join(results_dir, 'whole_pipeline'))),
                            fnet=(FNet(model_path=os.path.join(model_dir, model_name),
                                       dataset_name_for_evaluation=dataset_name_for_evaluation,
                                       use_coordinates='should_be_set-by_pipeline_constructor')),
                            results_subdir=results_subdir,
                            whole_pipeline_evaluation_task_name=task_name,
                            classification_threshold=classification_threshold,
                        )
                        json_output_file = os.path.join(pipeline.results_path(), 'fracture_outputs.json')
                        if os.path.isfile(json_output_file) and skip_existing:
                            print('Using existing file instead of evaluating:', json_output_file)
                        else:
                            config = FNetParameterEvaluator.load_config_for_model_evaluation(pipeline.fnet.model_path)
                            test_fold = config['cv_test_fold']
                            if test_mode == 'test':
                                iml = data_source.load_test_dataset_by_fold_id(test_fold)
                            elif test_mode == 'val':
                                iml = data_source.load_fold_by_id(data_source.val_fold_by_test_fold(test_fold))
                            else:
                                raise ValueError(test_mode)
                            pipeline.evaluate(iml)
                            # output_iml = pipeline.predict(iml)
                            # output_iml_path = os.path.join(pipeline.results_path(), 'pipeline_output_example.iml')
                            # output_iml.dump(output_iml_path)
                            # logging.info(f'Wrote {output_iml_path}')
                        assert os.path.isfile(json_output_file)
                        json_output_files.append(json_output_file)
                    metrics = []
                    for j in json_output_files:
                        with open(j, 'r') as f:
                            metrics.append(json.load(f))
                    metrics_df = pandas.DataFrame.from_records(m['patient_metrics'] for m in metrics)
                    summary_data = {}
                    for column in metrics_df.columns:
                        mean, std, conf = metric_summary(column, metrics_df[column])
                        summary_data[column] = {'mean': float(mean), 'std': std, 'conf': conf, 'values': metrics_df[column].tolist()}

                    summary_data['individual_results_files'] = [j for j in json_output_files]
                    summary_data['confusion'] = {}
                    summary_data['confusion']['tp'] = [[patient_id for patient_id, p in m['patients'].items() if p['y_true'] and p['y_pred']] for m in metrics]
                    summary_data['confusion']['fp'] = [[patient_id for patient_id, p in m['patients'].items() if not p['y_true'] and p['y_pred']] for m in metrics]
                    summary_data['confusion']['fn'] = [[patient_id for patient_id, p in m['patients'].items() if p['y_true'] and not p['y_pred']] for m in metrics]
                    summary_data['confusion']['tn'] = [[patient_id for patient_id, p in m['patients'].items() if not p['y_true'] and not p['y_pred']] for m in metrics]

        summary_json_path = os.path.join(base, 'whole_pipeline', test_mode, 'summary.json')
        print('Writing summary to', os.path.abspath(summary_json_path))
        with open(summary_json_path, 'w') as f:
            json.dump(summary_data, f)



if __name__ == '__main__':
    main()
