import os.path

import shutil

from model.fnet.evaluate import FNetParameterEvaluator
from model.fnet.model_analysis.analyze_trained_models import SingleModelEvaluationResult, ModelAnalyzer


class StoreRawModels(ModelAnalyzer):
    def analyze_batch(self, batch, y_preds, names):
        raise NotImplementedError('Not needed for this analyzer. Do not call.')

    def __init__(self, evaluator: FNetParameterEvaluator, skip_existing=None):
        super().__init__(evaluator=evaluator, analysis_needs_xs=False, skip_existing=skip_existing, model_level_caching=False,
                         binary_threshold_or_method=0.5)

    def to_subdir(self):
        return 'models'

    def analyze_single_model(self, model_path: str, ) -> SingleModelEvaluationResult:
        self.serializer().create_directory_if_not_exists(self.to_dir())
        to_file = os.path.join(self.to_dir(), f'{os.path.basename(model_path)}')
        if self.skip_existing and self.serializer().isfile(to_file):
            return []
        else:
            shutil.copy(model_path, to_file)
            assert model_path.endswith('.h5')
            shutil.copy(model_path[:-len('.h5')] + '.json',
                        to_file[:-len('.h5')] + '.json')
            print(f'Wrote {os.path.abspath(to_file)}')
            return []
